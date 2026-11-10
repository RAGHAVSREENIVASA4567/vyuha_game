"""
litewing_mpu_bridge_raw.py

Replaces the LiteWing phone app with your ESP32+MPU6050 breadboard rig.

Pipeline:
    ESP32+MPU6050  --BT SPP-->  this script  --WiFi/UDP CRTP-->  LiteWing v1.2

WHY THIS VERSION EXISTS (read this before the rest):
    The original bridge used cflib's high-level `Crazyflie` class, which
    requires a full setup handshake before it considers itself "connected":
    link up -> request protocol version -> download log/param Table of
    Contents -> THEN fire the connected callback.

    Debug logging showed the link itself comes up fine (LiteWing answers a
    raw CRTP handshake instantly), but the *protocol version request* -
    part of cflib's platform service, port 13 - never gets a reply from
    LiteWing's firmware. cflib waits forever for that, so `connected` never
    fires, even though the underlying link is completely healthy.

    This bridge only ever needs to SEND commander setpoints - it never reads
    logs, params, or memory - so none of that handshake is actually
    necessary. This version drops cflib's Crazyflie class entirely and
    talks to the link with a small raw CRTP-over-UDP client that:
      1. sends the same null/handshake packet your probe already confirmed
         gets a reply, to register as the drone's client, then
      2. sends COMMANDER (port 3) setpoint packets directly, at
         SEND_RATE_HZ, using the exact same [port][data...][checksum]
         framing cflib itself uses over UDP.

    No TOC download, no protocol-version wait, no chance of stalling on a
    service LiteWing's firmware may not implement.

SAFETY - READ BEFORE RUNNING:
    - Props OFF for your first several runs. Verify setpoints look sane by
      watching the printed values before ever attaching props.
    - This script clamps roll/pitch to +/-20 deg and thrust to a
      conservative ceiling below max by default - raise those only once
      you trust the pipeline.
    - Press Space to zero thrust instantly without disconnecting, or 'k' /
      Ctrl+C at any time to zero-thrust AND disconnect. There is also an
      automatic failsafe: if the serial link from the MPU rig drops, this
      script sends zero-thrust immediately rather than holding the last
      command - keyboard throttle state does not override that failsafe, and
      keyboard thrust is latched back to 0 so it does not resume on reconnect.
    - Connect your Mac's WiFi to the LiteWing's own AP before running this.
      IMPORTANT: this drone appears to lock onto whichever client
      (IP, port) registers first after boot and ignore all others. Power-
      cycle the drone before each real test, and make sure NOTHING else
      (phone app, probe scripts, a previous run of this script) talks UDP
      to it first in the same session.
    - The script binds a FIXED local UDP port (default 2391, --local-port)
      so reruns look like the same client to the drone. Use the same port
      in any probe script, and still power-cycle the drone if a different
      client (or port) reached it first since boot.
    - --ble reads the same CSV over BLE (Nordic UART) for Macs where Classic
      Bluetooth SPP delivers no bytes. Needs `pip install bleak`, the BLE
      firmware, and Bluetooth permission for your terminal app.
    - No arming request is sent. ESP-Drone-derived firmware like this
      typically doesn't need one - it treats "receiving valid commander
      packets" as armed and spins motors once thrust rises above their
      start threshold. Watch/listen to the props (or the disarmed ones,
      for the first tests) rather than expecting an armed/disarmed status.
    - This script puts your terminal into raw/cbreak mode to read single
      keypresses without Enter. It restores your terminal on exit, including
      on crashes, but if it ever leaves your terminal in a weird state, run
      `reset` (or open a new terminal tab).
"""

import re
import sys
import time
import tty
import struct
import socket
import termios
import select
import threading
import argparse
import platform

import serial
import serial.tools.list_ports


# ---------------------------------------------------------------------------
# Configuration - adjust these for your setup
# ---------------------------------------------------------------------------

DRONE_IP = "192.168.43.42"
DRONE_PORT = 2390
BT_PORT = "/dev/tty.AETHON-X"       # macOS BT SPP device - override with --bt-port if it changes
BT_BAUD = 115200
LOCAL_UDP_PORT = 2391               # fixed local source port so every run looks like the
                                     # SAME client to the drone (it locks onto the first
                                     # (IP, port) that registers after boot)

SEND_RATE_HZ = 30                   # CRTP setpoint send rate (keep >= 20 Hz)
CAL_SECONDS = 2.0                   # yaw-rate offset calibration window (rig held still)
CAL_MAX_RANGE_DPS = 12.0            # reject the window if yaw-rate varies more than this
CAL_MIN_SAMPLES = 20                # ...or if too few samples arrived
LINK_TIMEOUT_S = 2.0                # if no MPU sample in this long -> failsafe
                                     # (read timeout is 1.0s per attempt, so
                                     # this allows one missed read before failsafe)

# Safety clamps - raise these gradually as you trust the pipeline
MAX_TILT_DEG = 20.0                 # roll/pitch setpoint clamp, degrees
MAX_YAWRATE_DPS = 90.0              # yaw rate clamp, deg/s
# Conservative bench-test ceiling. The original file's comments said 30000,
# but the actual value was 60000. Keep this low until the firmware's thrust
# scale and failsafe behavior have been verified with propellers removed.
MAX_THRUST = 30000

# Keyboard throttle
THRUST_STEP = 1200                  # change per keypress (or per repeat while held)

# AETHON-X firmware's actual 13-field CSV line (verified against
# aethon_esp32.ino and aethon_visualizer.py's parser):
#   angleX,angleY,angleZ,gxR,gyR,gzR,axg,ayg,waz,thrX,thrY,thrZ,landed
# angleX = pitch (deg), angleY = roll (deg), angleZ = yaw (deg, drifts),
# gxR/gyR/gzR = gyro rates (deg/s). Only the first 6 fields are needed here.
LINE_RE = re.compile(
    r"^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*"
    r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,.*$"
)

CRTP_PORT_COMMANDER = 3


class RawCRTPLink:
    """
    Minimal CRTP-over-UDP client for ESP-Drone/LiteWing.

    Bypasses cflib's Crazyflie class (and its TOC/protocol-version setup
    handshake) entirely. Only implements what this bridge actually needs:
    registering with the drone and sending commander setpoint packets.

    IMPORTANT: The exact UDP encapsulation/registration handshake is
    firmware-specific and must be confirmed against the LiteWing firmware.
    Do not assume that receiving any UDP datagram proves CRTP registration.
    """

    def __init__(self, ip, port, timeout_s=1.0, retries=3, local_port=LOCAL_UDP_PORT,
                 send_disconnect=True):
        self.addr = (ip, port)
        self.send_disconnect = send_disconnect
        self.timeout_s = timeout_s
        self.retries = max(1, int(retries))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if local_port:
            # Fixed source port: reruns reuse the same (IP, port) the drone
            # already registered, instead of looking like a new stranger.
            self.sock.bind(("", local_port))
        self.sock.settimeout(timeout_s)
        self._connected = False

    def connect(self):
        """Try the configured firmware-specific registration packet.

        A UDP socket's connect() does not prove that the remote endpoint is
        alive. This method only reports success after a datagram is received;
        firmware-specific response validation still requires a known-good
        packet/ack specification.
        """
        try:
            self.sock.connect(self.addr)
            for attempt in range(1, self.retries + 1):
                try:
                    self.sock.send(b'\xFF\x01\x01\x01')
                    data = self.sock.recv(1024)
                    print(f"[CRTP] received UDP response: {data!r}")
                    self._connected = True
                    return True
                except socket.timeout:
                    print(f"[CRTP] no response (attempt {attempt}/{self.retries})")
            print(f"[CRTP] no handshake reply within {self.timeout_s:g}s per attempt")
            return False
        except OSError as e:
            print(f"[CRTP] socket error while connecting to {self.addr}: {e}", file=sys.stderr)
            return False

    def _send_frame(self, port, data_bytes, channel=0):
        # CRTP header as cflib builds it: port in bits 7..4, link bits 3..2
        # set to 1s (0x0C), channel in bits 1..0. cflib's UDP driver then
        # appends a 1-byte additive checksum (sum of all bytes mod 256),
        # which ESP-Drone-style firmware validates. Confirm against your
        # working probe if setpoints are accepted but motors do nothing.
        header = ((port & 0x0F) << 4) | 0x0C | (channel & 0x03)
        raw = bytes([header]) + data_bytes
        frame = raw + bytes([sum(raw) & 0xFF])
        try:
            if not self._connected:
                print("[CRTP] refusing to send setpoint before response was received", file=sys.stderr)
                return
            self.sock.send(frame)
        except OSError as e:
            print(f"[CRTP] send error: {e}", file=sys.stderr)

    def send_setpoint(self, roll, pitch, yawrate, thrust):
        """Same wire format/convention as cflib's Commander.send_setpoint:
        pitch is negated on the wire."""
        data = struct.pack('<fffH', roll, -pitch, yawrate, int(thrust))
        self._send_frame(CRTP_PORT_COMMANDER, data)

    def close(self):
        try:
            if self._connected and self.send_disconnect:
                self.sock.send(b'\xFF\x01\x02\x02')  # legacy best-effort disconnect packet
        except OSError:
            pass
        finally:
            self._connected = False
            self.sock.close()


class YawCalibrator:
    """Measures the gyro Z (yaw-rate) offset while the rig is held still.

    Feed it samples; it returns None while collecting, True once an offset is
    stored, or False when a window was rejected (rig moved / too few samples),
    in which case it starts a fresh window.
    """

    def __init__(self, seconds=CAL_SECONDS, max_range=CAL_MAX_RANGE_DPS,
                 min_samples=CAL_MIN_SAMPLES):
        self.seconds = seconds
        self.max_range = max_range
        self.min_samples = min_samples
        self.offset = None
        self.reset()

    def reset(self):
        self._vals = []
        self._start = None
        self._last_t = None

    def update(self, yawrate, t_recv, now):
        if self.offset is not None:
            return True
        if self._start is None:
            self._start = now
        if t_recv != self._last_t:           # count each sample once
            self._vals.append(yawrate)
            self._last_t = t_recv
        if now - self._start < self.seconds:
            return None
        vals = self._vals
        rng = (max(vals) - min(vals)) if vals else float("inf")
        if len(vals) >= self.min_samples and rng <= self.max_range:
            self.offset = sum(vals) / len(vals)
            return True
        self.reset()
        return False


class MPUReader:
    """Reads fused attitude samples from the ESP32+MPU6050 rig over BT SPP."""

    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self._ser = None
        self._lock = threading.Lock()
        self._latest = None          # (pitch, roll, yaw, thrust, mode, t_recv)
        self.bytes_rx = 0            # raw bytes received, for diagnosing a silent link
        self.samples_ok = 0          # valid samples parsed
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        # macOS Tahoe: the BT SPP device node can appear in /dev well
        # before the underlying RFCOMM connection is actually ready to
        # hand off data - opening it too early "succeeds" with no error
        # but every read comes back empty forever. Poll for the port to
        # actually appear, then retry the open a few times. This mirrors
        # the working approach in aethon_visualizer.py's _run_serial().
        if platform.system() == "Darwin":
            print(f"[MPU] waiting for {self.port} to be ready...")
            for _ in range(40):  # up to 20s
                names = [p.device for p in serial.tools.list_ports.comports()]
                if self.port in names:
                    print(f"[MPU] {self.port} ready.")
                    break
                time.sleep(0.5)
            else:
                print(f"[MPU] {self.port} not seen in port list - attempting open anyway...")

        last_err = None
        for attempt in range(10):
            try:
                self._ser = serial.Serial(
                    self.port, self.baud, timeout=1.0,
                    rtscts=False, dsrdtr=False, xonxoff=False,
                )
                break
            except serial.SerialException as e:
                last_err = e
                time.sleep(1.0)
        else:
            raise last_err

        self._thread = threading.Thread(target=self._run, daemon=True, name="MPUReader")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._ser:
            self._ser.close()

    def _run(self):
        while not self._stop.is_set():
            try:
                raw_bytes = self._ser.readline()
                self.bytes_rx += len(raw_bytes)
                raw = raw_bytes.decode("utf-8", errors="ignore")
            except (serial.SerialException, OSError) as e:
                print(f"[MPU] serial error: {e}", file=sys.stderr)
                time.sleep(0.5)
                continue
            self._ingest_line(raw)

    def _ingest_line(self, raw):
        """Parse one CSV line from the rig and store it as the latest sample."""
        m = LINE_RE.match(raw)
        if not m:
            if raw.strip():
                print(f"[MPU] ignored non-matching line: {raw.strip()[:120]!r}", file=sys.stderr)
            return
        try:
            angleX, angleY, angleZ, gxR, gyR, gzR = map(float, m.groups())
            # angleX = pitch (deg), angleY = roll (deg), gzR = yaw rate (deg/s)
            if not all(map(lambda v: abs(v) < 1e6, (angleX, angleY, angleZ, gxR, gyR, gzR))):
                raise ValueError("sensor value outside plausible numeric bounds")
            sample = (angleX, angleY, gzR, time.monotonic())
        except ValueError as e:
            print(f"[MPU] invalid sensor line: {e}", file=sys.stderr)
            return
        with self._lock:
            first_sample = self._latest is None
            self._latest = sample
            self.samples_ok += 1
        if first_sample:
            print("[MPU] first valid sensor sample received")

    def has_ever_received(self):
        """True once at least one valid sample has arrived (even if stale now)."""
        with self._lock:
            return self._latest is not None

    def latest(self):
        """Returns the latest (pitch_deg, roll_deg, yawrate_dps, t_recv), or
        None if nothing received yet / the link has gone stale."""
        with self._lock:
            sample = self._latest
        if sample is None:
            return None
        pitch, roll, yawrate, t_recv = sample
        if time.monotonic() - t_recv > LINK_TIMEOUT_S:
            return None
        return (pitch, roll, yawrate, t_recv)


class BLEReader(MPUReader):
    """Reads the same 13-field CSV lines over BLE (Nordic UART notifications).

    Use this instead of the serial reader when Classic Bluetooth SPP is
    broken (macOS Tahoe opens the port but delivers no bytes). Needs the
    BLE build of the ESP32 firmware and `pip install bleak`. BLE
    notifications can split a line across packets, so bytes are buffered
    until a newline before parsing.
    """

    NUS_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
    NUS_TX_NOTIFY = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"   # device -> host

    def __init__(self, name_hint="AETHON", scan_s=10.0):
        super().__init__(port="BLE", baud=0)
        self.name_hint = (name_hint or "").lower()
        self.scan_s = scan_s
        self._buf = b""
        self.drops = 0                # BLE link drops after the first connection
        self._ready = threading.Event()
        self._start_error = None

    def start(self):
        try:
            import bleak  # noqa: F401
        except ImportError:
            raise RuntimeError("BLE mode needs bleak: pip install bleak")
        print(f"[MPU] BLE: scanning up to {self.scan_s:.0f}s for a Nordic UART device "
              f"(name contains {self.name_hint!r} or advertises the NUS service)...")
        self._thread = threading.Thread(target=self._thread_main, daemon=True, name="BLEReader")
        self._thread.start()
        # Fail fast on scan/connect errors, but do not block forever
        self._ready.wait(timeout=self.scan_s + 15.0)
        if self._start_error:
            raise RuntimeError(self._start_error)
        if not self._ready.is_set():
            raise RuntimeError("BLE: timed out connecting to the ESP32")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _thread_main(self):
        import asyncio
        try:
            asyncio.run(self._ble_main())
        except Exception as e:
            self._start_error = self._start_error or f"BLE error: {e}"
            self._ready.set()

    def _on_notify(self, _char, data):
        self.bytes_rx += len(data)
        self._buf += bytes(data)
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._ingest_line(line.decode("utf-8", errors="ignore"))
        if len(self._buf) > 4096:     # no newline in a long time: drop junk
            self._buf = b""

    async def _find_device(self):
        import asyncio
        from bleak import BleakScanner
        found = await BleakScanner.discover(timeout=self.scan_s, return_adv=True)
        for dev, adv in found.values():
            name = (adv.local_name or dev.name or "").lower()
            uuids = [u.lower() for u in (adv.service_uuids or [])]
            if self.NUS_SERVICE in uuids or (self.name_hint and self.name_hint in name):
                print(f"[MPU] BLE: found {adv.local_name or dev.name!r} ({dev.address}) rssi={adv.rssi}")
                return dev
        return None

    async def _ble_main(self):
        import asyncio
        from bleak import BleakClient
        while not self._stop.is_set():
            dev = await self._find_device()
            if dev is None:
                if not self._ready.is_set():
                    self._start_error = ("BLE: no matching device found. Is the BLE firmware "
                                         "flashed and the ESP32 powered? (not paired as Classic SPP; "
                                         "BLE devices are NOT paired in System Settings)")
                    self._ready.set()
                    return
                print("[MPU] BLE: device not found, retrying...", file=sys.stderr)
                continue
            try:
                async with BleakClient(dev) as client:
                    await client.start_notify(self.NUS_TX_NOTIFY, self._on_notify)
                    print("[MPU] BLE: connected, notifications started")
                    self._ready.set()
                    while client.is_connected and not self._stop.is_set():
                        await asyncio.sleep(0.2)
                    try:
                        await client.stop_notify(self.NUS_TX_NOTIFY)
                    except Exception:
                        pass
            except Exception as e:
                if not self._ready.is_set():
                    self._start_error = f"BLE connect failed: {e}"
                    self._ready.set()
                    return
                print(f"[MPU] BLE connection lost ({e}), reconnecting...", file=sys.stderr)
            if not self._stop.is_set():
                self.drops += 1
                print(f"[MPU] BLE: link dropped (#{self.drops}), reconnecting...", file=sys.stderr)


class KeyboardThrottle:
    """Reads single keypresses (no Enter needed) to drive a thrust value.

    W / Up-Arrow   : increase thrust by THRUST_STEP
    S / Down-Arrow : decrease thrust by THRUST_STEP
    Space          : zero thrust instantly (soft stop, stays connected)
    k              : set kill flag (caller disconnects and exits)
    """

    def __init__(self):
        self._thrust = 0
        self._lock = threading.Lock()
        self._kill = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._old_term_settings = None

    def start(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Keyboard controls require an interactive terminal (TTY).")
        self._old_term_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._run, daemon=True, name="KeyboardThrottle")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._old_term_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term_settings)

    def thrust(self):
        with self._lock:
            return self._thrust

    def killed(self):
        return self._kill.is_set()

    def zero(self):
        """Drop keyboard thrust to 0 (used by the failsafe so thrust does not
        silently resume when the sensor link comes back)."""
        with self._lock:
            self._thrust = 0

    def _bump(self, delta):
        with self._lock:
            self._thrust = int(clamp(self._thrust + delta, 0, MAX_THRUST))

    def _run(self):
        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch == "\x1b":  # possible arrow-key escape sequence
                seq = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0.05)[0] else ""
                if seq == "[A":
                    self._bump(+THRUST_STEP)
                elif seq == "[B":
                    self._bump(-THRUST_STEP)
                continue
            if ch in ("w", "W"):
                self._bump(+THRUST_STEP)
            elif ch in ("s", "S"):
                self._bump(-THRUST_STEP)
            elif ch == " ":
                with self._lock:
                    self._thrust = 0
            elif ch in ("k", "K"):
                self._kill.set()
                self._stop.set()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def map_to_setpoint(pitch, roll, yawrate, thrust):
    """
    Direct passthrough control law: MPU tilt/yaw-rate + keyboard thrust ->
    CRTP setpoint.

    This is the block to replace with the Simulink control loop later -
    swap this function body for a call that hands (pitch, roll, yawrate,
    thrust) to Simulink and returns its computed (roll_sp, pitch_sp,
    yawrate_sp, thrust_sp) instead.
    """
    roll_sp = clamp(roll, -MAX_TILT_DEG, MAX_TILT_DEG)
    pitch_sp = clamp(pitch, -MAX_TILT_DEG, MAX_TILT_DEG)
    yawrate_sp = clamp(yawrate, -MAX_YAWRATE_DPS, MAX_YAWRATE_DPS)
    thrust_sp = int(clamp(thrust, 0, MAX_THRUST))
    return roll_sp, pitch_sp, yawrate_sp, thrust_sp


def main():
    parser = argparse.ArgumentParser(description="MPU6050 -> LiteWing raw CRTP bridge")
    parser.add_argument("--bt-port", default=BT_PORT, help="Bluetooth SPP serial device")
    parser.add_argument("--ble", action="store_true",
                        help="Read the MPU rig over BLE (Nordic UART) instead of a serial port; "
                             "needs the BLE firmware and `pip install bleak`")
    parser.add_argument("--ble-name", default="AETHON",
                        help="Substring of the BLE device name to look for (default: AETHON)")
    parser.add_argument("--ble-scan", type=float, default=10.0, help="BLE scan time in seconds")
    parser.add_argument("--ip", default=DRONE_IP, help="LiteWing IP address")
    parser.add_argument("--port", type=int, default=DRONE_PORT, help="LiteWing CRTP UDP port")
    parser.add_argument("--local-port", type=int, default=LOCAL_UDP_PORT,
                        help="Fixed local UDP source port (use the same one in probe scripts)")
    parser.add_argument("--no-disconnect", action="store_true",
                        help="Do NOT send the disconnect packet on exit (test whether it stops "
                             "the drone accepting a new registration without a power cycle)")
    parser.add_argument("--no-yaw-cal", action="store_true",
                        help="Skip the startup yaw-rate offset calibration (hold the rig still "
                             "for ~2 s after launch otherwise)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and print setpoints but do NOT connect to the drone")
    parser.add_argument("--handshake-timeout", type=float, default=1.0,
                        help="Seconds to wait for each UDP handshake response")
    parser.add_argument("--handshake-retries", type=int, default=3,
                        help="Number of handshake attempts")
    args = parser.parse_args()

    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.handshake_timeout <= 0 or args.handshake_retries < 1:
        parser.error("handshake timeout must be positive and retries must be >= 1")

    if args.ble:
        reader = BLEReader(name_hint=args.ble_name, scan_s=args.ble_scan)
    else:
        print(f"[MPU] opening {args.bt_port} @ {BT_BAUD} baud")
        reader = MPUReader(args.bt_port, BT_BAUD)
    throttle = KeyboardThrottle()
    link = None
    session_start = time.monotonic()
    yaw_cal = YawCalibrator()
    if args.no_yaw_cal:
        yaw_cal.offset = 0.0
    started_reader = False
    started_throttle = False

    try:
        reader.start()
        started_reader = True
        throttle.start()
        started_throttle = True
        print("Controls: W/Up = thrust+, S/Down = thrust-, Space = zero thrust, "
              "k = kill & disconnect, Ctrl+C = kill & disconnect.")

        if not args.dry_run:
            link = RawCRTPLink(args.ip, args.port,
                               timeout_s=args.handshake_timeout,
                               retries=args.handshake_retries,
                               local_port=args.local_port,
                               send_disconnect=not args.no_disconnect)
            print(f"[CRTP] connecting to {args.ip}:{args.port} (raw UDP) ...")
            if not link.connect():
                print("[CRTP] could not establish link - no setpoints will be sent")
                return 1
            print("[CRTP] UDP response received. Note: response contents are not validated.")

        print("[MPU] waiting for a valid sensor sample; no fresh sample means zero thrust")
        period = 1.0 / SEND_RATE_HZ
        last_print = 0.0
        loop_start = time.monotonic()
        warned_no_data = False
        while not throttle.killed():
            t0 = time.monotonic()
            sample = reader.latest()
            keyboard_thrust = throttle.thrust()

            if sample is None:
                roll_sp, pitch_sp, yawrate_sp, thrust_sp = 0.0, 0.0, 0.0, 0
                if yaw_cal.offset is None:
                    yaw_cal.reset()
                if keyboard_thrust > 0:
                    # Latch: after a sensor dropout the pilot must raise thrust
                    # again deliberately; it must not jump back to the old value
                    # when the link returns.
                    throttle.zero()
                    print("\n[SAFETY] sensor link lost - thrust latched to 0; "
                          "press W to raise it again once data returns", file=sys.stderr)
            elif yaw_cal.offset is None:
                # Calibrating: hold thrust and all setpoints at zero while the
                # rig is still, so a constant gyro Z offset is not sent as yaw.
                throttle.zero()
                roll_sp, pitch_sp, yawrate_sp, thrust_sp = 0.0, 0.0, 0.0, 0
                result = yaw_cal.update(sample[2], sample[3], t0)
                if result is True:
                    print(f"\n[MPU] yaw-rate offset calibrated: {yaw_cal.offset:+.1f} deg/s "
                          "(subtracted from yaw). You can raise thrust now.")
                elif result is False:
                    print("\n[MPU] yaw calibration rejected (rig moved?) - hold it still...",
                          file=sys.stderr)
            else:
                pitch, roll, yawrate, _ = sample
                roll_sp, pitch_sp, yawrate_sp, thrust_sp = map_to_setpoint(
                    pitch, roll, yawrate - yaw_cal.offset, keyboard_thrust
                )

            if link is not None:
                link.send_setpoint(roll_sp, pitch_sp, yawrate_sp, thrust_sp)

            if (sample is None and not warned_no_data and t0 - loop_start > 5.0
                    and not reader.has_ever_received()):
                warned_no_data = True
                if reader.bytes_rx == 0:
                    msg = ("0 bytes received from the ESP32 - it is not streaming. Check it is "
                           "powered, connected, and not held by another client.")
                else:
                    msg = (f"{reader.bytes_rx} bytes received but no line matched the 13-field "
                           "CSV format - check baud rate / firmware output.")
                print(f"\n[MPU] WARNING: no valid sample after 5s: {msg}", file=sys.stderr)

            if t0 - last_print > 0.5:
                if not sample:
                    status = "NO SIGNAL (failsafe)"
                elif yaw_cal.offset is None:
                    status = "CALIBRATING - hold rig still"
                else:
                    status = "OK"
                print(f"\r[{status}] roll={roll_sp:6.1f} pitch={pitch_sp:6.1f} "
                      f"yawrate={yawrate_sp:6.1f} thrust={thrust_sp:5d}   ", end="", flush=True)
                last_print = t0

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))
        return 0
    except KeyboardInterrupt:
        print("\n[SAFETY] Ctrl+C received")
        return 130
    except (OSError, serial.SerialException, RuntimeError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    finally:
        print("\n[SAFETY] sending zero-thrust and disconnecting")
        if link is not None:
            try:
                if link._connected:
                    for _ in range(10):
                        link.send_setpoint(0, 0, 0, 0)
                        time.sleep(0.02)
            finally:
                link.close()
        if started_reader:
            reader.stop()
        if started_throttle:
            throttle.stop()
        print("[SAFETY] done")
        run_s = time.monotonic() - session_start
        drops = getattr(reader, "drops", None)
        extra = f", BLE drops: {drops}" if drops is not None else ""
        print(f"[MPU] session summary: {run_s:.0f}s, {reader.samples_ok} valid samples{extra}")


if __name__ == "__main__":
    main()
