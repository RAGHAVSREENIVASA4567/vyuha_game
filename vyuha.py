#!/usr/bin/env python3
"""
vyuha.py — ESP32/MPU6050 telemetry bridge for the AETHON-X
Terrain Run browser game
=====================================================================
Browsers cannot speak Classic Bluetooth SPP (Web Bluetooth only
supports BLE/GATT), so the three.js game can't open the same
Bluetooth port `aethon_visualizer_3d.py` does directly. This script is
the fix: it runs the *exact same* line-parsing / dead-reckoning code
as `aethon_visualizer_3d.py`'s IMUReader, and re-broadcasts each
parsed telemetry frame as JSON over a local WebSocket server. The game
(index.html + game.js) connects to that WebSocket instead of a serial
port, and everything downstream — the drone's attitude, its
dead-reckoned position, thrust magnitude, landed state — behaves
identically to the Panda3D visualizer, because it's computed with the
same formulas from the same protocol.

TRANSPORTS
----------
--ble               BLE (GATT notify) — the AETHON-X firmware's current
                     transport. Sidesteps macOS's Classic Bluetooth SPP
                     stack entirely (see below for why that matters).
--port <device>     Classic Bluetooth SPP or a wired USB-serial port
                     (whatever pyserial can open).
--sim               No hardware — fake telemetry for UI testing.

If you switch the ESP32 to plug in over USB, you can also skip this
bridge entirely and have the browser talk to it directly via the Web
Serial API (Chrome/Edge only) — the game's "USB (Web Serial)" connect
option in the menu does that.

WHY BLE INSTEAD OF CLASSIC SPP
-------------------------------
On macOS Tahoe (26.5), the Classic Bluetooth SPP port opens
successfully (no error, `screen`/pyserial both attach fine) but
delivers zero bytes — the bug lives in macOS's RFCOMM stack, not in
this script or the firmware (confirmed: firmware telemetry over USB
Serial is clean). `bleak`'s BLE central mode goes through
CoreBluetooth instead, a completely different stack, and needs no
manual "pair in System Settings" step at all — it connects
programmatically by device name.

Usage:
    pip install pyserial websockets bleak
    python3 vyuha.py --ble                    # BLE (recommended)
    python3 vyuha.py --ble --ble-name AETHON-X
    python3 vyuha.py                           # Classic/USB, auto-detect port
    python3 vyuha.py --port COM5                # explicit port (Windows)
    python3 vyuha.py --port /dev/cu.usbserial-0001
    python3 vyuha.py --sim                      # no hardware — fake telemetry
    python3 vyuha.py --ws-port 8765             # change the WebSocket port

Then in the game's terrain-select menu, choose "Bluetooth (bridge)" and
leave the address as ws://localhost:8765 (default).

Wire protocol from the ESP32 (identical whether sent via BLE notify or
Classic SPP — only the transport changed), one CSV frame per sample:
    angleX,angleY,angleZ,gyroX,gyroY,gyroZ[,accelX,accelY,waz]
        [,thrX,thrY,thrZ][,landed]
(6 fields minimum; the rest are optional and derived if omitted, same
as aethon_visualizer_3d.py.)
"""

import argparse
import asyncio
import json
import math
import platform
import threading
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

try:
    import websockets
except ImportError:
    websockets = None

try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    BleakClient = None
    BleakScanner = None


# Nordic UART Service (NUS) UUIDs — a widely-used de facto standard for
# streaming text over BLE notify/write characteristics. Reusing it
# (rather than inventing new UUIDs) means generic tools like nRF
# Connect / LightBlue can see and read the TX characteristic directly,
# which is handy for bench-testing the firmware before the bridge is
# even involved.
BLE_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
BLE_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # notify: ESP32 -> bridge


def _list_ports():
    if serial is None:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


# macOS always exposes these regardless of whether any real hardware is
# paired/plugged in. They match the old loose "cu." filter below, so
# without this exclusion list _default_port() can silently pick one of
# them instead of the ESP32 and the bridge will "work" while reading
# from a port that never emits a single byte.
_BOGUS_PORT_MARKERS = ("debug-console", "Bluetooth-Incoming-Port", "wlan-debug")


def _default_port():
    ports = _list_ports()
    real = [p for p in ports if not any(b in p for b in _BOGUS_PORT_MARKERS)]
    for p in real:
        if any(k in p for k in ("AETHON", "usbserial", "ttyUSB", "ttyACM", "cu.")):
            return p
    # Fall back to a bogus-looking port only if nothing else exists at
    # all, so --port errors are loud instead of a silent wrong pick.
    return real[0] if real else (ports[0] if ports else None)


# ──────────────────────────────────────────────────────────────────
#  Telemetry source — ported essentially verbatim from
#  aethon_visualizer_3d.py's IMUReader, minus the Panda3D-specific
#  bits. Same sign flips, same dead-reckoning, same --sim fallback
#  motion. Three transports (serial/classic, BLE, sim) all feed the
#  same _process_line() so downstream code never needs to know which
#  one is active.
# ──────────────────────────────────────────────────────────────────
class TelemetrySource(threading.Thread):
    def __init__(self, port, baud=115200, simulated=False,
                 use_ble=False, ble_name="AETHON-X"):
        super().__init__(daemon=True)
        self.port, self.baud, self.simulated = port, baud, simulated
        self.use_ble, self.ble_name = use_ble, ble_name
        self.angleX = self.angleY = self.angleZ = 0.0
        self.gyroX = self.gyroY = self.gyroZ = 0.0
        self.accelX = self.accelY = self.accelZ = 0.0
        self.vertZ = 1.0
        self.thrX = 0.0
        self.thrY = 0.0
        self.thrZ = 1.0
        self.landed = False
        self.posX = self.posY = self.posZ = 0.0
        self.velX = self.velY = self.velZ = 0.0
        self._lock = threading.Lock()
        self._quit_evt = threading.Event()
        self.connected = False
        self.error_msg = ""
        # state used by _process_line(), shared across transports
        self._last_t = time.time()
        self._raw_line_count = 0

    def stop(self):
        self._quit_evt.set()

    def snapshot(self):
        with self._lock:
            return dict(
                angleX=self.angleX, angleY=self.angleY, angleZ=self.angleZ,
                gyroX=self.gyroX, gyroY=self.gyroY, gyroZ=self.gyroZ,
                accelX=self.accelX, accelY=self.accelY, accelZ=self.accelZ,
                vertZ=self.vertZ,
                thrX=self.thrX, thrY=self.thrY, thrZ=self.thrZ,
                landed=self.landed,
                posX=self.posX, posY=self.posY, posZ=self.posZ,
                velX=self.velX, velY=self.velY, velZ=self.velZ,
                connected=self.connected,
            )

    # ── simulated motion (no hardware needed) ───────────────────────
    def _run_sim(self):
        self.connected = True
        t0 = time.time()
        G, DT = 9.81, 0.02
        vx = vy = vz = 0.0
        px = py = pz = 0.0
        while not self._quit_evt.is_set():
            t = time.time() - t0
            ax = math.degrees(math.sin(t * 0.7) * 0.35)
            ay = math.degrees(math.cos(t * 0.5) * 0.30)
            az = math.degrees(math.sin(t * 0.2) * 0.60)
            gx = math.degrees(0.7 * 0.35 * math.cos(t * 0.7))
            gy = math.degrees(-0.5 * 0.30 * math.sin(t * 0.5))
            gz = math.degrees(0.2 * 0.60 * math.cos(t * 0.2))
            pitch, roll = math.radians(ax), math.radians(ay)
            agx = -math.sin(pitch) + 0.05 * math.sin(t * 3.1)
            agy = -math.cos(pitch) * math.sin(roll)
            agz_wf = math.cos(pitch) * math.cos(roll)

            waz = agz_wf
            ax_w = math.sin(pitch) * G
            ay_w = -math.sin(roll) * G
            az_w = (waz - 1.0) * G
            drag = 0.18
            vx = vx * (1 - drag) + ax_w * DT
            vy = vy * (1 - drag) + ay_w * DT
            vz = vz * (1 - drag) + az_w * DT
            px += vx * DT
            py += vy * DT
            pz += vz * DT

            thr_x = math.sin(pitch) * waz
            thr_y = -math.cos(pitch) * math.sin(roll) * waz
            thr_z = math.cos(pitch) * math.cos(roll) * waz

            is_still = (abs(gx) < 2.0 and abs(gy) < 2.0 and abs(gz) < 2.0
                        and abs(ax) < 3.0 and abs(ay) < 3.0)

            with self._lock:
                self.angleX, self.angleY, self.angleZ = ax, ay, az
                self.gyroX, self.gyroY, self.gyroZ = gx, gy, gz
                self.accelX, self.accelY, self.accelZ = agx, agy, agz_wf
                self.vertZ = agz_wf
                self.thrX, self.thrY, self.thrZ = thr_x, thr_y, thr_z
                self.landed = is_still
                self.posX, self.posY, self.posZ = px, py, pz
                self.velX, self.velY, self.velZ = vx, vy, vz
            time.sleep(DT)

    # ── shared parser: one CSV frame -> updated state ───────────────
    # Used by every live transport (BLE notify, Classic SPP, USB
    # serial) so the dead-reckoning / sign-flip logic exists in exactly
    # one place. `raw` is one already-delimited frame, no trailing
    # newline required.
    def _process_line(self, raw: str):
        self._raw_line_count += 1
        n = self._raw_line_count
        if n <= 20 or n % 100 == 0:
            print(f"[bridge] raw line #{n}: {raw!r}", flush=True)
        raw = raw.strip()
        if not raw or len(raw) < 5:
            if n <= 20:
                print("[bridge]   -> skipped: empty/too short", flush=True)
            return
        for i, ch in enumerate(raw):
            if ch in '0123456789-+':
                raw = raw[i:]
                break
        parts = raw.split(",")
        if len(parts) < 6:
            if n <= 20:
                print(f"[bridge]   -> skipped: only {len(parts)} field(s), need >= 6", flush=True)
            return
        try:
            nx, ny, nz = float(parts[0]), float(parts[1]), float(parts[2])
            # Firmware sends pitch/roll with the opposite sign convention
            # this game expects (nose-up should read positive,
            # tilt-right should read positive) — flip both once here,
            # same as aethon_visualizer_3d.py.
            nx = -nx
            ny = -ny
            nz = -nz
            ngx, ngy, ngz = float(parts[3]), float(parts[4]), float(parts[5])
            if len(parts) >= 9:
                nax, nay, nwaz = float(parts[6]), float(parts[7]), float(parts[8])
            else:
                p, r = math.radians(nx), math.radians(ny)
                nax = -math.sin(p)
                nay = -math.cos(p) * math.sin(r)
                nwaz = math.cos(p) * math.cos(r)
            if len(parts) >= 12:
                ntx, nty, ntz = float(parts[9]), float(parts[10]), float(parts[11])
            else:
                p, r = math.radians(nx), math.radians(ny)
                ntx = math.sin(p) * nwaz
                nty = -math.cos(p) * math.sin(r) * nwaz
                ntz = math.cos(p) * math.cos(r) * nwaz
            nlanded = bool(int(float(parts[12]))) if len(parts) >= 13 else False
        except ValueError as e:
            if n <= 20:
                print(f"[bridge]   -> skipped: parse error ({e})", flush=True)
            return

        now = time.time()
        dt = max(0.0, min(0.1, now - self._last_t))
        self._last_t = now
        pitch, roll = math.radians(nx), math.radians(ny)
        G, DRAG = 9.81, 0.18
        ax_w = math.sin(pitch) * G
        ay_w = -math.sin(roll) * G
        az_w = (nwaz - 1.0) * G

        with self._lock:
            self.angleX, self.angleY, self.angleZ = nx, ny, nz
            self.gyroX, self.gyroY, self.gyroZ = ngx, ngy, ngz
            self.accelX, self.accelY, self.accelZ = nax, nay, nwaz
            self.vertZ = nwaz
            self.thrX, self.thrY, self.thrZ = ntx, nty, ntz
            self.landed = nlanded

            if nlanded:
                self.velX = self.velY = self.velZ = 0.0
            else:
                self.velX = self.velX * (1 - DRAG) + ax_w * dt
                self.velY = self.velY * (1 - DRAG) + ay_w * dt
                self.velZ = self.velZ * (1 - DRAG) + az_w * dt
                self.posX += self.velX * dt
                self.posY += self.velY * dt
                self.posZ += self.velZ * dt

    # ── live BLE (GATT notify) ───────────────────────────────────────
    # Runs its own asyncio event loop inside this thread (bleak is
    # async-only). Scans for a device advertising `self.ble_name`,
    # connects, subscribes to the TX notify characteristic, and feeds
    # every decoded frame to the same _process_line() the Classic/USB
    # path uses.
    def _run_ble(self):
        if BleakClient is None:
            self.error_msg = "bleak not installed — run: pip install bleak"
            print(f"[bridge] {self.error_msg}", flush=True)
            return
        try:
            asyncio.run(self._run_ble_async())
        except Exception as e:
            print(f"[bridge] BLE fatal error: {e}", flush=True)

    async def _run_ble_async(self):
        line_buf = bytearray()

        def _on_notify(_handle, data: bytearray):
            nonlocal line_buf
            for b in data:
                if b in (10, 13):  # \n or \r
                    if line_buf:
                        self._process_line(line_buf.decode("ascii", "ignore"))
                        line_buf = bytearray()
                else:
                    line_buf.append(b)
                    if len(line_buf) > 512:
                        line_buf = bytearray()

        while not self._quit_evt.is_set():
            print(f"[bridge] scanning for BLE device '{self.ble_name}' ...", flush=True)
            device = await BleakScanner.find_device_by_filter(
                lambda d, adv: (d.name or "") == self.ble_name,
                timeout=10.0,
            )
            if device is None:
                print(f"[bridge] no BLE device named '{self.ble_name}' found, retrying...", flush=True)
                await asyncio.sleep(2.0)
                continue

            print(f"[bridge] found {device.address}, connecting...", flush=True)
            try:
                async with BleakClient(device) as client:
                    self.connected = True
                    self.error_msg = ""
                    print(f"[bridge] BLE connected to {self.ble_name}", flush=True)
                    await client.start_notify(BLE_TX_CHAR_UUID, _on_notify)
                    # Sit here until disconnected or asked to quit — bleak
                    # delivers notifications on its own via the callback
                    # above, this loop just watches the connection.
                    while client.is_connected and not self._quit_evt.is_set():
                        await asyncio.sleep(0.5)
            except Exception as e:
                print(f"[bridge] BLE error: {e}", flush=True)
            self.connected = False
            print("[bridge] BLE disconnected, will retry...", flush=True)
            await asyncio.sleep(1.0)

    # ── live serial / Classic Bluetooth SPP / USB ────────────────────
    def _run_serial(self):
        if serial is None:
            self.error_msg = "pyserial not installed — run: pip install pyserial"
            return
        available = _list_ports()

        if platform.system() == "Darwin" and any(
            k in self.port for k in ('AETHON', 'tty.', 'cu.', 'rfcomm')
        ):
            print("[bridge] waiting for BT port...", flush=True)
            for _ in range(40):
                current = _list_ports()
                if self.port in current:
                    print("[bridge] port ready.", flush=True)
                    break
                time.sleep(0.5)

        try:
            ser = serial.Serial(self.port, self.baud, timeout=1,
                                 rtscts=False, dsrdtr=False, xonxoff=False)
        except serial.SerialException as e:
            self.error_msg = str(e)
            print(f"[bridge] failed to open {self.port}: {e}", flush=True)
            return
        self.connected = True

        line_buf = bytearray()
        last_byte_time = time.time()
        last_silence_warning = 0.0

        while not self._quit_evt.is_set():
            try:
                byte = ser.read(1)
            except (OSError, serial.SerialException) as e:
                # This is the case a Bluetooth SPP peer disconnecting
                # mid-session actually hits — previously this was caught
                # by a bare `except Exception: continue`, which silently
                # busy-spun forever with no sign anything had gone wrong.
                if self.connected:
                    print(f"[bridge] port dropped: {e}", flush=True)
                self.connected = False
                try:
                    ser.close()
                except Exception:
                    pass
                time.sleep(1.0)
                try:
                    ser = serial.Serial(self.port, self.baud, timeout=1,
                                         rtscts=False, dsrdtr=False, xonxoff=False)
                    self.connected = True
                    last_byte_time = time.time()
                    print(f"[bridge] reconnected to {self.port}", flush=True)
                except serial.SerialException:
                    pass  # keep retrying on the next loop iteration
                continue
            if not byte:
                # read() timed out (1s) with nothing available — not an
                # error by itself, but prolonged silence on a port that
                # claims to be connected means bytes aren't arriving
                # (e.g. wrong port, or firmware/BT stopped transmitting).
                now = time.time()
                if self.connected and now - last_byte_time > 5.0 and now - last_silence_warning > 5.0:
                    print(f"[bridge] warning: no bytes received in "
                          f"{now - last_byte_time:.0f}s (port open, link idle)", flush=True)
                    last_silence_warning = now
                continue
            last_byte_time = time.time()
            ch = byte[0]
            if ch in (ord('\n'), ord('\r')):
                if line_buf:
                    self._process_line(line_buf.decode("ascii", "ignore"))
                    line_buf.clear()
            else:
                line_buf.append(ch)
                if len(line_buf) > 512:
                    line_buf.clear()
        ser.close()

    def run(self):
        if self.simulated:
            self._run_sim()
        elif self.use_ble:
            self._run_ble()
        else:
            self._run_serial()


# ──────────────────────────────────────────────────────────────────
#  WebSocket server — broadcasts the latest snapshot to every
#  connected browser tab at a fixed rate.
# ──────────────────────────────────────────────────────────────────
async def _broadcast_loop(source: TelemetrySource, clients: set, hz: float = 30.0):
    period = 1.0 / hz
    while True:
        if clients:
            payload = json.dumps(source.snapshot())
            dead = set()
            for ws in clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            clients -= dead
        await asyncio.sleep(period)


async def _handler(ws, clients: set):
    clients.add(ws)
    print(f"[bridge] game connected ({len(clients)} client(s))")
    try:
        async for _msg in ws:
            pass  # game doesn't send anything back currently
    finally:
        clients.discard(ws)
        print(f"[bridge] game disconnected ({len(clients)} client(s))")


async def _main_async(args):
    if websockets is None:
        raise SystemExit("websockets not installed — run: pip install websockets")

    if args.ble:
        if BleakClient is None:
            raise SystemExit("bleak not installed — run: pip install bleak")
        source = TelemetrySource(None, use_ble=True, ble_name=args.ble_name)
        print(f"[bridge] BLE mode — looking for device '{args.ble_name}'")
    elif args.sim:
        source = TelemetrySource(None, simulated=True)
        print("[bridge] running in --sim mode (no hardware)")
    else:
        port = args.port or _default_port()
        if not port:
            available = _list_ports()
            raise SystemExit(
                "No serial/Bluetooth port found.\n"
                f"Available: {available or ['(none)']}\n"
                "Pass --port explicitly, use --ble for the BLE transport, "
                "or --sim to test without hardware."
            )
        source = TelemetrySource(port, baud=args.baud)
        print(f"[bridge] opening {port} @ {args.baud} baud ...")

    source.start()

    clients = set()

    async def handler(ws):
        await _handler(ws, clients)

    async with websockets.serve(handler, "localhost", args.ws_port):
        print(f"[bridge] WebSocket server listening on ws://localhost:{args.ws_port}")
        print("[bridge] open the game and choose 'Bluetooth (bridge)' to connect.")
        await _broadcast_loop(source, clients)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None,
                     help="Classic Bluetooth SPP or USB-serial port (e.g. COM5, "
                          "/dev/cu.usbserial-0001, /dev/cu.AETHON-X). "
                          "Auto-detected if omitted and --ble/--sim not given.")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--ble", action="store_true",
                     help="Use BLE (GATT notify) instead of Classic Bluetooth SPP/USB. "
                          "Recommended on macOS — sidesteps the Classic SPP bug.")
    ap.add_argument("--ble-name", default="AETHON-X",
                     help="BLE advertised device name to scan for (default: AETHON-X).")
    ap.add_argument("--sim", action="store_true",
                     help="Generate fake telemetry instead of reading hardware.")
    ap.add_argument("--ws-port", type=int, default=8765,
                     help="WebSocket port the game connects to (default 8765).")
    args = ap.parse_args()

    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
