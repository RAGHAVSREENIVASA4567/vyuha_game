#!/usr/bin/env python3
"""
AETHON-X · Panda3D IMU Motion Visualiser (Full 3D Engine edition)
-------------------------------------------------------------------------
Same telemetry source as the matplotlib version (ESP32 Bluetooth SPP /
USB serial / simulation), rendered as a real-time 3D flight-sim scene:
  • Real textured drone model (glTF/.glb) with its propellers spinning,
    or a procedural lit hull as an automatic fallback
  • Ground grid + soft drop shadow that tracks altitude
  • Glowing thrust-plume particles scaled by thrust magnitude
  • Three camera modes: chase / orbit / free-orbit (mouse drag)
  • Full telemetry HUD + aerial status badge (same state machine as before)

Bundled asset: CesiumDrone.glb — from CesiumGS/cesium's open-source
sample data (Apache License 2.0), placed next to this script. Swap in
your own model any time with --model path/to/your.glb.

Dependencies:  pip install panda3d panda3d-gltf pyserial numpy

USAGE
  macOS  : python3 aethon_visualizer_3d.py --port /dev/cu.AETHON-X
  Linux  : python3 aethon_visualizer_3d.py --port /dev/rfcomm0
  Windows: python3 aethon_visualizer_3d.py --port COM5
  Sim    : python3 aethon_visualizer_3d.py --sim

CONTROLS
  V        cycle camera mode (chase / orbit / free)
  Mouse-drag (free mode)   orbit the camera
  Scroll / +/-             zoom
  G        deploy both arms (fold out; in mouse mode both then track
            the mouse cursor together)
  H        retract both arms (fold back into the body)
  P        toggle palm-tracking mode (webcam controls the arms instead
            of the mouse; requires `pip install mediapipe opencv-python`)
            With both hands visible, your RIGHT palm independently aims
            and deploys/retracts the original (right-side) arm, and your
            LEFT palm independently aims and deploys/retracts the
            mirrored (left-side) arm — open hand to deploy/aim, close a
            fist to retract, per hand. In mouse mode (palm tracking off)
            both arms instead move together, driven by the single mouse
            cursor. Also opens a live camera panel, top-right, showing
            both tracked hand skeletons and their detection readouts.

            Left/right aim uses the palm's horizontal position directly,
            since that tracks reliably. Forward/back and up/down don't —
            a webcam can't reliably tell a hand moving toward it from
            depth alone, and small vertical tracking noise reads as arm
            jitter — so those two axes are driven by finger gestures
            instead, per hand, and hold their position when you let go
            (they don't snap back):
              point with only your INDEX finger  -> arm extends forward
              point with only your PINKY finger  -> arm pulls back
              THUMBS UP  (only thumb, tip above wrist)   -> arm swings up
              THUMBS DOWN (only thumb, tip below wrist)  -> arm swings down
            While a thumbs-up/down gesture is held, moving your palm
            farther from centre speeds up how fast the arm swings —
            proportional control in that plane, without depending on the
            camera judging your palm's absolute height.
  L        toggle the legs' back-and-forth throttle motion
  Esc      quit
"""

import argparse
import math
import platform
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    from hand_tracker import HandTracker, PREVIEW_W, PREVIEW_H
    HAND_TRACKING_AVAILABLE = True
except ImportError:
    HAND_TRACKING_AVAILABLE = False
    PREVIEW_W, PREVIEW_H = 320, 240  # only used to size the (hidden) preview texture

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
from panda3d.core import (
    GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomTriangles,
    GeomLines, Geom, GeomNode, NodePath, LineSegs,
    AmbientLight, DirectionalLight, Vec3, Vec4, Point2, Point3, Quat,
    TransparencyAttrib, CardMaker, TextNode, WindowProperties,
    ClockObject, CullFaceAttrib, Material,
    Fog, Texture, PNMImage, ColorBlendAttrib, loadPrcFileData,
)

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

# ── render-quality knobs (must be set before the window opens) ─────
loadPrcFileData("", "framebuffer-multisample 1")
loadPrcFileData("", "multisamples 4")
loadPrcFileData("", "framebuffer-srgb 1")

# ── constants ────────────────────────────────────────────────────
GYRO_LIM   = 300.0
TILT_MOVE  = 8.0
TILT_STATIC = 3.0
ROT_THRESH = 15.0
BUF        = 150

STATUS_COLORS = {
    "LANDED":    (0.53, 0.53, 0.53, 1),
    "HOVERING":  (0.0, 1.0, 0.67, 1),
    "ASCENDING": (0.0, 1.0, 0.27, 1),
    "DESCENDING": (1.0, 0.27, 0.0, 1),
    "FORWARD":   (0.0, 0.67, 1.0, 1),
    "BACKWARD":  (1.0, 0.0, 0.67, 1),
    "LEFT":      (1.0, 0.67, 0.0, 1),
    "RIGHT":     (0.67, 0.27, 1.0, 1),
    "TILT_R":    (1.0, 1.0, 0.0, 1),
    "TILT_L":    (1.0, 0.87, 0.0, 1),
    "ROT_CW":    (1.0, 0.53, 0.0, 1),
    "ROT_CCW":   (0.53, 1.0, 0.0, 1),
}


def _default_port():
    s = platform.system()
    if s == "Darwin": return "/dev/cu.AETHON-X"
    if s == "Linux":  return "/dev/rfcomm0"
    return "COM6"


def _list_ports():
    if serial is None:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


def _shortest_arc_quat(v_from, v_to):
    """Quaternion that rotates unit vector v_from onto unit vector v_to,
    taking the shortest path. Used to point the gun's rest-pose direction
    (straight down) at an arbitrary aim direction with a single rotation."""
    a = Vec3(v_from); b = Vec3(v_to)
    a.normalize(); b.normalize()
    d = max(-1.0, min(1.0, a.dot(b)))
    if d > 0.999999:
        return Quat()  # already aligned
    if d < -0.999999:
        # 180-degree case: any axis perpendicular to 'a' works.
        axis = a.cross(Vec3(1, 0, 0))
        if axis.length() < 1e-6:
            axis = a.cross(Vec3(0, 1, 0))
        axis.normalize()
        q = Quat()
        q.setFromAxisAngleRad(math.pi, axis)
        return q
    axis = a.cross(b)
    axis.normalize()
    angle = math.acos(d)
    q = Quat()
    q.setFromAxisAngleRad(angle, axis)
    return q


def _aerial_status(pitch, roll, waz, gz, landed):
    if landed:
        return "LANDED", STATUS_COLORS["LANDED"]
    if waz > 1.20:
        return "ASCENDING", STATUS_COLORS["ASCENDING"]
    if waz < 0.80:
        return "DESCENDING", STATUS_COLORS["DESCENDING"]
    if gz > ROT_THRESH:
        return "ROTATING CW", STATUS_COLORS["ROT_CW"]
    if gz < -ROT_THRESH:
        return "ROTATING CCW", STATUS_COLORS["ROT_CCW"]
    if pitch > TILT_MOVE:
        return "MOVING FORWARD", STATUS_COLORS["FORWARD"]
    if pitch < -TILT_MOVE:
        return "MOVING BACKWARD", STATUS_COLORS["BACKWARD"]
    if roll > TILT_MOVE:
        return "MOVING RIGHT", STATUS_COLORS["RIGHT"]
    if roll < -TILT_MOVE:
        return "MOVING LEFT", STATUS_COLORS["LEFT"]
    if roll > TILT_STATIC:
        return "TILTED RIGHT", STATUS_COLORS["TILT_R"]
    if roll < -TILT_STATIC:
        return "TILTED LEFT", STATUS_COLORS["TILT_L"]
    return "HOVERING", STATUS_COLORS["HOVERING"]


# ──────────────────────────────────────────────────────────────────
#  TELEMETRY THREAD  (identical contract to the matplotlib version)
# ──────────────────────────────────────────────────────────────────
class IMUReader(threading.Thread):
    def __init__(self, port, baud=115200, simulated=False):
        super().__init__(daemon=True)
        self.port, self.baud, self.simulated = port, baud, simulated
        self.angleX = self.angleY = self.angleZ = 0.0
        self.gyroX = self.gyroY = self.gyroZ = 0.0
        self.accelX = self.accelY = self.accelZ = 0.0
        self.vertZ = 1.0
        self.thrX = 0.0; self.thrY = 0.0; self.thrZ = 1.0
        self.landed = False
        self.posX = self.posY = self.posZ = 0.0
        self.velX = self.velY = self.velZ = 0.0
        self._lock = threading.Lock()
        self._quit_evt = threading.Event()
        self.connected = False
        self.error_msg = ""

    def stop(self):
        self._quit_evt.set()

    def snapshot(self):
        with self._lock:
            return (self.angleX, self.angleY, self.angleZ,
                     self.gyroX, self.gyroY, self.gyroZ,
                     self.accelX, self.accelY, self.accelZ,
                     self.vertZ,
                     self.thrX, self.thrY, self.thrZ,
                     self.landed,
                     self.posX, self.posY, self.posZ,
                     self.velX, self.velY, self.velZ)

    # ── simulated motion ─────────────────────────────────────────
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
            px += vx * DT; py += vy * DT; pz += vz * DT

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

    # ── live serial ───────────────────────────────────────────────
    def _run_serial(self):
        if serial is None:
            self.error_msg = "pyserial not installed — run: pip install pyserial"
            return
        available = _list_ports()

        if platform.system() == "Darwin" and any(
            k in self.port for k in ('AETHON', 'tty.', 'cu.', 'rfcomm')
        ):
            print("AETHON-X: waiting for BT port...", flush=True)
            for _ in range(40):
                current = _list_ports()
                if self.port in current:
                    print("AETHON-X: port ready.", flush=True)
                    break
                # macOS often only exposes the cu.* (call-up) device for a
                # Bluetooth SPP peer, not the matching tty.* one — tty.*
                # blocks waiting for a carrier signal BT never raises. If
                # our configured port never shows up but a same-name
                # cu./tty. sibling does, switch to it automatically.
                if '.' in self.port:
                    suffix = self.port.split('.', 1)[1]
                    alt = next((p for p in current
                                if p != self.port and p.endswith(suffix)), None)
                    if alt:
                        print(f"AETHON-X: '{self.port}' not offered, "
                              f"using '{alt}' instead.", flush=True)
                        self.port = alt
                        break
                time.sleep(0.5)

        ser = None
        for _ in range(10):
            try:
                ser = serial.Serial(self.port, self.baud, timeout=1,
                                     rtscts=False, dsrdtr=False, xonxoff=False)
                break
            except serial.SerialException:
                time.sleep(1.0)
        if ser is None:
            self.error_msg = (
                f"Cannot open '{self.port}' after 10 attempts.\n"
                f"Available: {available or ['(none)']}\n"
                "Use --sim to test without hardware."
            )
            return

        is_bt = ('AETHON' in self.port or 'tty.' in self.port
                 or 'cu.' in self.port or 'rfcomm' in self.port)
        if not is_bt:
            ser.reset_input_buffer()
        else:
            time.sleep(2.0)
            ser.flushInput() if hasattr(ser, 'flushInput') else None
        self.connected = True

        line_buf = bytearray()
        last_t = time.time()

        def _process_line(raw: str):
            nonlocal last_t
            raw = raw.strip()
            if not raw or len(raw) < 5:
                return
            for i, ch in enumerate(raw):
                if ch in '0123456789-+':
                    raw = raw[i:]
                    break
            parts = raw.split(",")
            if len(parts) < 6:
                return
            try:
                nx, ny, nz = float(parts[0]), float(parts[1]), float(parts[2])
                # Firmware sends pitch/roll with the opposite sign convention
                # this visualizer expects (nose-up should read positive,
                # tilt-right should read positive) — flip both once here so
                # the HUD numbers, status labels, and physics stay consistent.
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
            except ValueError:
                return

            # Dead-reckoning position/velocity from raw accelerometer data —
            # same integration the simulator uses. Accel-only integration
            # drifts (no GPS/optical-flow correction), so treat this as a
            # rough sense of motion, not an accurate absolute position.
            now = time.time()
            dt = max(0.0, min(0.1, now - last_t))
            last_t = now
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
                    # Zero the velocity/position drift whenever the drone is
                    # detected as stationary, instead of letting bias creep
                    # accumulate indefinitely while sitting on the ground.
                    self.velX = self.velY = self.velZ = 0.0
                else:
                    self.velX = self.velX * (1 - DRAG) + ax_w * dt
                    self.velY = self.velY * (1 - DRAG) + ay_w * dt
                    self.velZ = self.velZ * (1 - DRAG) + az_w * dt
                    self.posX += self.velX * dt
                    self.posY += self.velY * dt
                    self.posZ += self.velZ * dt

        while not self._quit_evt.is_set():
            try:
                byte = ser.read(1)
            except Exception:
                continue
            if not byte:
                continue
            ch = byte[0]
            if ch in (ord('\n'), ord('\r')):
                if line_buf:
                    _process_line(line_buf.decode("ascii", "ignore"))
                    line_buf.clear()
            else:
                line_buf.append(ch)
                if len(line_buf) > 512:
                    line_buf.clear()
        ser.close()

    def run(self):
        self._run_sim() if self.simulated else self._run_serial()


# ──────────────────────────────────────────────────────────────────
#  GEOMETRY BUILDERS
# ──────────────────────────────────────────────────────────────────
def make_box(sx, sy, sz, color):
    """Axis-aligned lit box, centered at origin, size sx*sy*sz."""
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("box", fmt, Geom.UHStatic)
    vdata.setNumRows(24)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")

    x, y, z = sx / 2, sy / 2, sz / 2
    faces = [
        ((0, 0, 1),  [(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]),
        ((0, 0, -1), [(-x, y, -z), (x, y, -z), (x, -y, -z), (-x, -y, -z)]),
        ((0, 1, 0),  [(-x, y, -z), (-x, y, z), (x, y, z), (x, y, -z)]),
        ((0, -1, 0), [(-x, -y, z), (-x, -y, -z), (x, -y, -z), (x, -y, z)]),
        ((1, 0, 0),  [(x, -y, -z), (x, y, -z), (x, y, z), (x, -y, z)]),
        ((-1, 0, 0), [(-x, -y, z), (-x, y, z), (-x, y, -z), (-x, -y, -z)]),
    ]
    tris = GeomTriangles(Geom.UHStatic)
    idx = 0
    for normal, pts in faces:
        for p in pts:
            vw.addData3(*p)
            nw.addData3(*normal)
            cw.addData4(*color)
        tris.addVertices(idx, idx + 1, idx + 2)
        tris.addVertices(idx, idx + 2, idx + 3)
        idx += 4
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("box")
    node.addGeom(geom)
    return NodePath(node)


def make_disc(radius, color, segments=20):
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("disc", fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")
    vw.addData3(0, 0, 0); nw.addData3(0, 0, 1); cw.addData4(*color)
    for i in range(segments + 1):
        th = 2 * math.pi * i / segments
        vw.addData3(radius * math.cos(th), radius * math.sin(th), 0)
        nw.addData3(0, 0, 1)
        cw.addData4(*color)
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(1, segments + 1):
        tris.addVertices(0, i, i + 1)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("disc")
    node.addGeom(geom)
    return NodePath(node)


def make_sky_dome(radius=250, rings=10, segments=28,
                   zenith_color=Vec4(0.01, 0.015, 0.03, 1),
                   horizon_color=Vec4(0.05, 0.09, 0.14, 1)):
    """A vertex-colored hemisphere, dark at the zenith and lighter at the
    horizon, meant to be recentred on the camera each frame so it always
    reads as an infinitely distant sky rather than a fixed-size object."""
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData("sky", fmt, Geom.UHStatic)
    vdata.setNumRows((rings + 1) * (segments + 1))
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")

    for i in range(rings + 1):
        t = i / rings
        elev = t * (math.pi / 2)
        # slight ease so the gradient stays richer near the horizon
        blend = t ** 0.7
        color = horizon_color + (zenith_color - horizon_color) * blend
        for j in range(segments + 1):
            az = 2 * math.pi * j / segments
            x = radius * math.cos(elev) * math.sin(az)
            y = radius * math.cos(elev) * math.cos(az)
            z = radius * math.sin(elev)
            vw.addData3(x, y, z)
            cw.addData4(color)

    tris = GeomTriangles(Geom.UHStatic)
    stride = segments + 1
    for i in range(rings):
        for j in range(segments):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1
            tris.addVertices(a, b, d)
            tris.addVertices(a, d, c)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("sky-dome")
    node.addGeom(geom)
    return NodePath(node)


def make_radial_gradient_texture(size=64, rgb=(1, 1, 1), power=1.8):
    """A soft round sprite (white core -> transparent edge) used to fake
    soft shadows and glowing sparks without needing external art assets."""
    img = PNMImage(size, size)
    img.addAlpha()
    cx = cy = size / 2.0
    maxr = size / 2.0
    for y in range(size):
        for x in range(size):
            dx, dy = (x + 0.5) - cx, (y + 0.5) - cy
            d = min(1.0, math.sqrt(dx * dx + dy * dy) / maxr)
            a = max(0.0, 1.0 - d) ** power
            img.setXelA(x, y, rgb[0], rgb[1], rgb[2], a)
    tex = Texture("radial-gradient")
    tex.load(img)
    tex.setMagfilter(Texture.FTLinear)
    tex.setMinfilter(Texture.FTLinear)
    return tex


def make_ground_grid(size=40, step=2, color=Vec4(0.05, 0.35, 0.28, 1)):
    ls = LineSegs("grid")
    ls.setThickness(1.0)
    ls.setColor(color)
    n = int(size / step)
    for i in range(-n, n + 1):
        ls.moveTo(i * step, -size, 0)
        ls.drawTo(i * step, size, 0)
        ls.moveTo(-size, i * step, 0)
        ls.drawTo(size, i * step, 0)
    return NodePath(ls.create())


# ──────────────────────────────────────────────────────────────────
#  PARTICLE POOL (manual, no ParticleEffect dependency)
# ──────────────────────────────────────────────────────────────────
class ThrustParticles:
    def __init__(self, parent, count=40):
        self.parent = parent
        self.pool = []
        cm = CardMaker("spark")
        cm.setFrame(-0.09, 0.09, -0.09, 0.09)
        spark_tex = make_radial_gradient_texture(32, rgb=(1, 1, 1), power=2.2)
        for _ in range(count):
            np_ = parent.attachNewNode(cm.generate())
            np_.setBillboardPointEye()
            np_.setTexture(spark_tex)
            np_.setTransparency(TransparencyAttrib.MAlpha)
            np_.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd))
            np_.setDepthWrite(False)
            np_.setBin("transparent", 0)
            np_.setLightOff()
            np_.setColor(1, 0.7, 0.2, 0)
            np_.hide()
            self.pool.append({"node": np_, "life": 0.0, "vel": Vec3(0, 0, 0)})

    def emit(self, origin, count, thrust_mag):
        spawned = 0
        for p in self.pool:
            if p["life"] <= 0 and spawned < count:
                p["node"].setPos(origin + Vec3(
                    (np.random.rand() - 0.5) * 0.15, 0,
                    (np.random.rand() - 0.5) * 0.15))
                speed = 1.2 + thrust_mag * 1.5
                p["vel"] = Vec3((np.random.rand() - 0.5) * 0.3,
                                -speed,
                                (np.random.rand() - 0.5) * 0.3)
                p["life"] = 0.6 + np.random.rand() * 0.3
                p["node"].show()
                spawned += 1

    def update(self, dt):
        for p in self.pool:
            if p["life"] > 0:
                p["life"] -= dt
                if p["life"] <= 0:
                    p["node"].hide()
                    continue
                p["node"].setPos(p["node"].getPos() + p["vel"] * dt)
                alpha = max(0.0, min(1.0, p["life"] / 0.9))
                p["node"].setColor(1, 0.65, 0.15, alpha * 0.85)


# ──────────────────────────────────────────────────────────────────
#  ROLLING GRAPH  (screen-space telemetry strip-chart panel)
# ──────────────────────────────────────────────────────────────────
GRAPH_PALETTE = [
    Vec4(0.23, 0.84, 0.65, 1),   # teal
    Vec4(1.0, 0.53, 0.0, 1),     # orange
    Vec4(0.30, 0.62, 1.0, 1),    # blue
]


class RollingGraph:
    """A small live strip-chart drawn in pixel2d (fixed pixel coordinates,
    origin top-left, +x right, +y down). Call push() once per frame with
    the latest values, then redraw() after resizing/repositioning."""

    def __init__(self, parent, title, series_names, buf=150, font=None):
        self.parent = parent
        self.series_names = series_names
        self.buf = buf
        self.data = {name: deque([0.0] * buf, maxlen=buf) for name in series_names}
        self.root = parent.attachNewNode("graph-root")
        self.line_np = None
        self.bg_np = None
        self.x = self.y = self.w = self.h = 0
        self.title_txt = OnscreenText(
            text=title, parent=self.root, pos=(0, 0), scale=13,
            fg=(0.55, 0.62, 0.66, 1), align=TextNode.ALeft, mayChange=True,
            font=font)
        legend_line = "  ".join(series_names)
        self.legend_txt = OnscreenText(
            text=legend_line, parent=self.root, pos=(0, 0), scale=11,
            fg=(0.4, 0.47, 0.5, 1), align=TextNode.ARight, mayChange=True,
            font=font)

    def set_rect(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.root.setPos(x, 0, -y)
        self.title_txt.setPos(4, -14)
        self.legend_txt.setPos(w - 4, -14)
        if self.bg_np is not None:
            self.bg_np.removeNode()
        cm = CardMaker("graph-bg")
        cm.setFrame(0, w, -h, 0)
        self.bg_np = self.root.attachNewNode(cm.generate())
        self.bg_np.setColor(0.05, 0.07, 0.11, 0.88)
        self.bg_np.setTransparency(TransparencyAttrib.MAlpha)
        self.bg_np.setBin("fixed", 0)

    def push(self, values):
        for name, v in zip(self.series_names, values):
            self.data[name].append(v)

    def redraw(self, colors=None):
        if self.line_np is not None:
            self.line_np.removeNode()
        colors = colors or GRAPH_PALETTE
        ls = LineSegs()
        top = 18
        ls.setThickness(1.0)
        ls.setColor(0.10, 0.14, 0.21, 1)
        ls.moveTo(0, 0, -top); ls.drawTo(self.w, 0, -top)
        ls.drawTo(self.w, 0, -self.h); ls.drawTo(0, 0, -self.h); ls.drawTo(0, 0, -top)
        for i in range(1, 4):
            yy = -top - (self.h - top) * i / 4
            ls.moveTo(0, 0, yy); ls.drawTo(self.w, 0, yy)

        all_vals = [v for name in self.series_names for v in self.data[name]]
        vmin, vmax = (min(all_vals), max(all_vals)) if all_vals else (0, 1)
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0
        pad = (vmax - vmin) * 0.1
        vmin, vmax = vmin - pad, vmax + pad
        plot_h = self.h - top

        for si, name in enumerate(self.series_names):
            color = colors[si % len(colors)]
            ls.setColor(color)
            pts = list(self.data[name])
            n = len(pts)
            if n < 2:
                continue
            for i, v in enumerate(pts):
                px = self.w * i / (n - 1)
                frac = (v - vmin) / (vmax - vmin)
                py = -top - plot_h * frac
                if i == 0:
                    ls.moveTo(px, 0, py)
                else:
                    ls.drawTo(px, 0, py)

        self.line_np = self.root.attachNewNode(ls.create())
        self.line_np.setBin("fixed", 1)


# ──────────────────────────────────────────────────────────────────
#  MAIN APP
# ──────────────────────────────────────────────────────────────────
class AethonApp(ShowBase):
    def __init__(self, reader, src_label, model_path=None):
        ShowBase.__init__(self)
        self.reader = reader
        self.src_label = src_label
        self.model_path = model_path
        self.disableMouse()

        props = WindowProperties()
        props.setTitle("AETHON-X · Panda3D Live Visualiser")
        try:
            self.win.requestProperties(props)
        except Exception:
            pass

        self._build_sky()
        self._build_lighting()
        self._build_post_fx()
        self._build_ground()
        self._build_drone()
        self._build_hud()
        self._build_graphs()
        self._build_cam_preview()

        self.cam_modes = ["chase", "orbit", "free"]
        self.cam_mode_idx = 0
        self.cam_yaw = 0.0
        self.cam_pitch = 20.0
        self.cam_dist = 6.0
        self.accept("v", self._cycle_cam)
        self.accept("wheel_up", lambda: self._zoom(-0.4))
        self.accept("wheel_down", lambda: self._zoom(0.4))
        self._drag_active = False
        self.accept("mouse1", self._drag_start)
        self.accept("mouse1-up", self._drag_stop)
        self._last_mouse = None

        # Retractable arm controls
        self.accept("g", self._gun_deploy)
        self.accept("h", self._gun_retract)

        # Palm-tracking control (webcam instead of mouse)
        self.hand_tracker = HandTracker() if HAND_TRACKING_AVAILABLE else None
        self.hand_tracking_on = False
        self._hand_state = None
        self.accept("p", self._toggle_hand_tracking)
        self.accept("escape", self._on_quit)

        # Leg throttle animation (single key toggles start/stop)
        self.accept("l", self._toggle_legs)

        self.fps_c = 0
        self.fps_v = 0
        self.last_t = time.time()
        self.pos_history = deque(maxlen=BUF)

        self.taskMgr.add(self._update, "aethon-update")

    # ── scene construction ─────────────────────────────────────────
    def _build_sky(self):
        self.sky_horizon = Vec4(0.05, 0.09, 0.14, 1)
        self.sky_zenith = Vec4(0.01, 0.015, 0.03, 1)
        dome = make_sky_dome(radius=250, rings=10, segments=28,
                              zenith_color=self.sky_zenith,
                              horizon_color=self.sky_horizon)
        dome.reparentTo(self.render)
        self.sky = dome
        self.sky.setTwoSided(True)   # camera sits inside the dome
        self.sky.setLightOff(1)      # flat vertex-color, not lit
        self.sky.setShaderOff(1)     # bypass the auto per-pixel shader (no normals)
        self.sky.setFogOff(1)        # the sky itself must not get fogged out
        self.sky.setBin("background", 0)
        self.sky.setDepthWrite(False)
        self.sky.setDepthTest(False)
        self.setBackgroundColor(self.sky_horizon)

    def _build_lighting(self):
        # Per-pixel lighting (instead of Panda's default per-vertex) — this
        # alone makes the glTF model's curved surfaces shade far more smoothly.
        self.render.setShaderAuto()

        # Key light (warm "sun")
        dlight = DirectionalLight("sun")
        dlight.setColor(Vec4(1.05, 1.0, 0.92, 1))
        dnp = self.render.attachNewNode(dlight)
        dnp.setHpr(35, -55, 0)
        self.render.setLight(dnp)

        # Fill/rim light from the opposite side, cool-toned and dim — kills
        # the flat, single-sided look a lone directional light produces.
        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.18, 0.24, 0.35, 1))
        fnp = self.render.attachNewNode(fill)
        fnp.setHpr(-150, -20, 0)
        self.render.setLight(fnp)

        alight = AmbientLight("amb")
        alight.setColor(Vec4(0.16, 0.18, 0.22, 1))
        anp = self.render.attachNewNode(alight)
        self.render.setLight(anp)

        # Depth fog matched to the sky's horizon color: distant grid lines
        # and the floor fade into the sky instead of ending in a hard edge.
        horizon = getattr(self, "sky_horizon", Vec4(0.02, 0.03, 0.06, 1))
        fog = Fog("scene-fog")
        fog.setColor(horizon.x, horizon.y, horizon.z)
        fog.setExpDensity(0.045)
        self.render.setFog(fog)

    def _build_post_fx(self):
        """Best-effort bloom + SSAO post-processing. Each effect is wrapped
        independently so a GPU/driver that only supports one of them (or
        neither) still leaves the app running normally."""
        self.filters = None
        try:
            from direct.filter.CommonFilters import CommonFilters
            self.filters = CommonFilters(self.win, self.cam)
        except Exception as e:
            print(f"[AETHON-X] Post-processing filters unavailable ({e}) — "
                  f"continuing without them.")
            return

        try:
            self.filters.setBloom(desat=0.4, intensity=1.6, size="medium")
        except Exception as e:
            print(f"[AETHON-X] Bloom unavailable ({e}).")

        try:
            self.filters.setAmbientOcclusion()
        except Exception as e:
            print(f"[AETHON-X] Ambient occlusion unavailable ({e}).")

    def _build_ground(self):
        grid = make_ground_grid()
        grid.reparentTo(self.render)
        grid.setTransparency(TransparencyAttrib.MAlpha)
        grid.setColorScale(1, 1, 1, 0.55)

        cm = CardMaker("floor")
        cm.setFrame(-40, 40, -40, 40)
        floor = self.render.attachNewNode(cm.generate())
        floor.setP(-90)
        floor.setColor(0.03, 0.05, 0.05, 1)
        floor.setZ(-0.01)
        floor.setLightOff()
        floor.setTwoSided(True)

        cm2 = CardMaker("shadow")
        cm2.setFrame(-0.7, 0.7, -0.7, 0.7)
        self.shadow = self.render.attachNewNode(cm2.generate())
        self.shadow.setP(-90)
        self.shadow.setTransparency(TransparencyAttrib.MAlpha)
        self.shadow.setTexture(make_radial_gradient_texture(64, rgb=(0, 0, 0), power=1.4))
        self.shadow.setColor(1, 1, 1, 0.6)
        self.shadow.setZ(0.005)
        self.shadow.setLightOff()
        self.shadow.setTwoSided(True)
        self.shadow.setBin("transparent", 0)
        self.shadow.setDepthWrite(False)

    def _build_drone(self):
        self.drone = self.render.attachNewNode("drone")
        self.rotors = []          # procedural fallback discs (if used)
        self.rotor_nodes = []     # (nodepath, spin_dir) pairs from a loaded model

        model_np = self._try_load_model()
        if model_np is not None:
            model_np.reparentTo(self.drone)
        else:
            self._build_procedural_hull()

        # ── Retractable / aimable gun rig ─────────────────────────────
        # Looks for the named joint chain baked into the combat-drone GLB
        # (J1_ShoulderAbduction ... J6_WristFlexion). Silently does nothing
        # if the loaded model doesn't have this rig (e.g. the stock Cesium
        # drone, or the procedural fallback hull).
        self.gun_j1 = self.gun_j2 = self.gun_j4 = self.gun_j6 = None
        if model_np is not None:
            j1 = model_np.find("**/J1_ShoulderAbduction")
            j2 = model_np.find("**/J2_ShoulderFlexion")
            j4 = model_np.find("**/J4_ElbowFlexion")
            j6 = model_np.find("**/J6_WristFlexion")
            if not j1.isEmpty():
                self.gun_j1, self.gun_j2 = j1, j2
                self.gun_j4, self.gun_j6 = j4, j6

        # Mirrored second arm rig, on the opposite side of the drone
        # (the "_L" suffixed joint chain baked into the model alongside
        # the original). Same joint names, driven with identical fold/aim
        # logic in _update_gun — both arms always mirror each other's
        # motion since they share one fold state and one aim target.
        self.gun2_j1 = self.gun2_j2 = self.gun2_j4 = self.gun2_j6 = None
        if model_np is not None:
            j1b = model_np.find("**/J1_ShoulderAbduction_L")
            j2b = model_np.find("**/J2_ShoulderFlexion_L")
            j4b = model_np.find("**/J4_ElbowFlexion_L")
            j6b = model_np.find("**/J6_WristFlexion_L")
            if not j1b.isEmpty():
                self.gun2_j1, self.gun2_j2 = j1b, j2b
                self.gun2_j4, self.gun2_j6 = j4b, j6b

        # 0.0 = fully deployed/extended (gun tracks the mouse), 1.0 = fully
        # retracted/folded against the body. Starts retracted; press G to
        # deploy, H to retract. gun_fold drives the original (right) arm;
        # gun2_fold drives the mirrored (left) arm. In mouse mode the two
        # stay locked together (both arms move as one); in palm-tracking
        # mode each is driven independently by its own hand's curl.
        self.gun_fold = 1.0
        self.gun_fold_target = 1.0
        self.gun2_fold = 1.0
        self.gun2_fold_target = 1.0
        self.GUN_FOLD_SPEED = 2.2          # fold progress per second
        self.GUN_FOLD_J2_DEG = 150.0       # shoulder pitch when fully folded
        self.GUN_FOLD_J4_DEG = -70.0       # elbow bend when fully folded
        self.GUN_FOLD_J6_DEG = -20.0       # wrist bend when fully folded
        self.GUN_AIM_DISTANCE = 8.0        # how far out to project the mouse aim point

        # While deployed, J2/J4/J6 no longer sit frozen — they flex along
        # with the base joint so the whole arm articulates toward the aim
        # point, not just the shoulder. "reach" below is 0 when the target
        # sits at the rest pose (straight down) and 1 when it's as far from
        # that rest pose as possible (straight up).
        self.GUN_AIM_J2_DEG = 45.0         # extra shoulder-flexion lift at full reach
        self.GUN_AIM_J4_BASE_DEG = -18.0   # elbow bend near the rest pose
        self.GUN_AIM_J4_REACH_DEG = -55.0  # extra elbow bend added at full reach
        self.GUN_AIM_J6_DEG = -18.0        # wrist counter-bend at full reach

        # Hysteresis thresholds for the palm-curl deploy/retract gesture
        # (see _update_gun). Deploy once curl drops below this...
        self.GUN_CURL_DEPLOY = 0.35
        # ...retract once curl climbs above this. The gap between the two
        # is the "ambiguous zone" that's ignored, so half-curled fingers
        # don't cause flicker between deployed and folded.
        self.GUN_CURL_RETRACT = 0.75

        # Gesture-driven forward/back "reach" bias, per arm. Camera depth
        # (apparent hand size) is too noisy to trust for this axis, so
        # instead a dedicated finger gesture nudges it directly (see
        # _update_arm_bias): point with just the index finger to push the
        # arm forward, just the pinky to pull it back. -1..1, and it holds
        # its value when neither gesture is being made — a throttle, not
        # something that snaps back to centre.
        self.reach_bias_r = 0.0
        self.reach_bias_l = 0.0
        self.ARM_REACH_SPEED = 0.7   # bias units/sec while the gesture is held

        # Gesture-driven up/down "vertical" bias, per arm. Same idea as
        # reach_bias above but for aiming the arm up or down, since the
        # palm's raw vertical screen position is also too noisy to drive
        # directly. A thumbs-up/thumbs-down gesture moves it; while that
        # gesture is held, the palm's own vertical position scales how
        # fast it moves, so you still get proportional, "free" control in
        # that plane rather than a fixed-speed nudge. -1..1, holds value
        # otherwise.
        self.vert_bias_r = 0.0
        self.vert_bias_l = 0.0
        self.ARM_VERT_SPEED = 0.7

        # ── Leg throttle/pump animation ───────────────────────────────
        # Swings all 4 leg hip pivots (Leg_0 .. Leg_3) back and forth
        # together. One key toggles it on and off.
        self.leg_nodes = []
        if model_np is not None:
            for i in range(4):
                leg = model_np.find(f"**/Leg_{i}")
                if not leg.isEmpty():
                    self.leg_nodes.append(leg)
        self.legs_moving = False
        self._leg_swing_t = 0.0
        self._leg_angle = 0.0
        self.LEG_SWING_AMPLITUDE_DEG = 22.0
        self.LEG_SWING_SPEED = 3.0          # radians/sec of internal phase
        self.LEG_EASE_RATE = 6.0            # how fast it eases to neutral when stopped

    def _try_load_model(self):
        """Attempt to load self.model_path as a real drone mesh. Returns a
        NodePath on success, or None (caller falls back to a procedural hull)."""
        path = self.model_path
        if not path:
            return None
        import os
        if not os.path.isfile(path):
            print(f"[AETHON-X] Model '{path}' not found — using procedural hull.")
            return None
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".glb", ".gltf"):
                import gltf
                root = gltf.load_model(path)
                np = NodePath(root)
            else:
                np = self.loader.loadModel(path)
        except Exception as e:
            print(f"[AETHON-X] Could not load model '{path}': {e}\n"
                  f"           Falling back to procedural hull.")
            return None

        # Normalise size: fit the model's footprint to roughly a 1.3-unit
        # radius so it matches the rest of the scene (grid step, camera dist).
        #
        # Measure this on the hull only, excluding the aimable arm rig(s)
        # (ShoulderSocket / ShoulderSocket_L). Their rest-pose reach can
        # extend well beyond the hull's own footprint — especially now
        # that the arm is scaled up 8x in the asset itself — and since
        # this normalization fits the WHOLE model to a fixed footprint,
        # letting the arms dominate that measurement would shrink the
        # entire drone down to compensate (arms included), which is
        # exactly backwards from "make the arm bigger". Stash them just
        # for the bounds check, then unstash before returning.
        arm_roots = [n for n in (np.find("**/ShoulderSocket"),
                                  np.find("**/ShoulderSocket_L")) if not n.isEmpty()]
        for n in arm_roots:
            n.stash()
        min_pt, max_pt = np.getTightBounds()
        for n in arm_roots:
            n.unstash()
        span = max(max_pt.x - min_pt.x, max_pt.y - min_pt.y)
        if span > 0:
            np.setScale(1.35 / span)
        # This particular bundled asset (Cesium's sample drone) faces -Y;
        # our nose/forward convention is +Y, so flip it 180 degrees.
        np.setH(180)
        np.setZ(-min_pt.z * (1.35 / span if span > 0 else 1))

        # Hook up any propeller sub-meshes so they actually spin.
        for child in np.findAllMatches("**/*"):
            name = child.getName().lower()
            if "prop" in name or "rotor" in name:
                spin_dir = -1.0 if "counter" in name else 1.0
                self.rotor_nodes.append((child, spin_dir))

        # Texture/material polish: mipmapped + anisotropic filtering so the
        # hull doesn't look muddy up close or shimmer at a distance, plus a
        # touch of specular so surfaces read as more than flat matte paint.
        for tex in np.findAllTextures():
            tex.setMinfilter(Texture.FTLinearMipmapLinear)
            tex.setMagfilter(Texture.FTLinear)
            tex.setAnisotropicDegree(8)
        try:
            mat = Material("drone-material")
            mat.setShininess(24.0)
            mat.setSpecular(Vec4(0.4, 0.4, 0.42, 1))
            np.setMaterial(mat, 1)
        except Exception:
            pass  # glTF's own PBR shader setup takes priority if present

        return np

    def _build_procedural_hull(self):
        hull = make_box(1.1, 0.28, 1.1, Vec4(0.06, 0.43, 0.34, 1))
        hull.reparentTo(self.drone)

        arm_len = 1.35
        colors = [Vec4(0.23, 0.84, 0.65, 1), Vec4(0.23, 0.84, 0.65, 1),
                  Vec4(1.0, 0.13, 0.27, 1), Vec4(0.23, 0.84, 0.65, 1)]
        for i, (sx, sz) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            tip = Vec3(sx * arm_len, 0, sz * arm_len)
            ls = LineSegs("arm")
            ls.setThickness(3.0)
            ls.setColor(0.6, 0.9, 0.85, 1)
            ls.moveTo(0, 0, 0)
            ls.drawTo(tip)
            self.drone.attachNewNode(ls.create())

            rotor = make_disc(0.42, colors[i])
            rotor.reparentTo(self.drone)
            rotor.setPos(tip)
            rotor.setTwoSided(True)
            rotor.setTransparency(TransparencyAttrib.MAlpha)
            rotor.setColorScale(1, 1, 1, 0.55)
            rotor.setBin("transparent", 1)
            rotor.setDepthWrite(False)
            self.rotors.append(rotor)

        nose = LineSegs("nose")
        nose.setThickness(3.5)
        nose.setColor(1, 0.13, 0.27, 1)
        nose.moveTo(0, 0, 0)
        nose.drawTo(0, 1.5, 0)
        self.drone.attachNewNode(nose.create())



    def _build_hud(self):
        # Translucent backing panels so text stays legible over bright parts
        # of the 3D scene, instead of floating bare on top of it.
        cm = CardMaker("status-panel")
        cm.setFrame(-0.32, 0.32, 0.80, 0.94)
        status_bg = self.aspect2d.attachNewNode(cm.generate())
        status_bg.setColor(0.03, 0.05, 0.08, 0.55)
        status_bg.setTransparency(TransparencyAttrib.MAlpha)
        status_bg.setBin("fixed", -2)

        cm2 = CardMaker("hud-panel")
        cm2.setFrame(-1.33, -0.68, 0.55, 0.94)
        hud_bg = self.aspect2d.attachNewNode(cm2.generate())
        hud_bg.setColor(0.03, 0.05, 0.08, 0.55)
        hud_bg.setTransparency(TransparencyAttrib.MAlpha)
        hud_bg.setBin("fixed", -2)

        self.status_txt = OnscreenText(
            text="HOVERING", pos=(0, 0.88), scale=0.07,
            fg=(0, 1, 0.67, 1), align=TextNode.ACenter,
            mayChange=True, font=self.loader.loadFont("cmtt12"))

        self.hud_txt = OnscreenText(
            text="", pos=(-1.28, 0.82), scale=0.045,
            fg=(0.6, 1, 0.85, 1), align=TextNode.ALeft,
            mayChange=True)

        self.help_txt = OnscreenText(
            text="V  cycle camera   |   drag  orbit (free mode)   |   "
                 "G  deploy arms  |   H  retract arms  |   P  palm tracking   |   "
                 "L  toggle legs   |   esc  quit",
            pos=(0, -0.95), scale=0.04, fg=(0.4, 0.5, 0.55, 1),
            align=TextNode.ACenter, mayChange=False)

    def _build_cam_preview(self):
        """Small webcam panel in the top-right corner showing the live
        palm-tracking feed (skeleton overlay + detection readouts).
        Hidden until palm tracking is switched on with P."""
        aspect = self.getAspectRatio()
        margin = 0.03
        panel_w, panel_h = 0.62, 0.62 * (PREVIEW_H / PREVIEW_W)
        right = aspect - margin
        top = 1.0 - margin
        left = right - panel_w
        bottom = top - panel_h

        cm = CardMaker("camcard-bg")
        cm.setFrame(left - 0.02, right + 0.02, bottom - 0.06, top + 0.02)
        bg = self.aspect2d.attachNewNode(cm.generate())
        bg.setColor(0.03, 0.05, 0.08, 0.6)
        bg.setTransparency(TransparencyAttrib.MAlpha)
        bg.setBin("fixed", -2)

        self.cam_tex = Texture("hand-cam")
        self.cam_tex.setup2dTexture(PREVIEW_W, PREVIEW_H,
                                     Texture.T_unsigned_byte, Texture.F_rgb)
        self.cam_tex.setMinfilter(Texture.FTLinear)
        self.cam_tex.setMagfilter(Texture.FTLinear)
        blank = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
        self.cam_tex.setRamImage(blank.tobytes())

        cm2 = CardMaker("camcard-img")
        cm2.setFrame(left, right, bottom, top)
        img = self.aspect2d.attachNewNode(cm2.generate())
        img.setTexture(self.cam_tex)
        img.setBin("fixed", -1)

        label = OnscreenText(
            text="PALM CAM - press P", pos=((left + right) / 2, bottom - 0.035),
            scale=0.035, fg=(0.6, 1, 0.85, 1), align=TextNode.ACenter,
            mayChange=False)

        self.cam_preview_nodes = [bg, img, label]
        for n in self.cam_preview_nodes:
            n.hide()

    def _update_cam_preview(self):
        if not (HAND_TRACKING_AVAILABLE and self.hand_tracking_on
                and self.hand_tracker is not None):
            return
        frame = self.hand_tracker.snapshot().get("frame")
        if frame is None:
            return
        # Panda textures are stored bottom-up; flip the top-down camera
        # frame vertically so it renders right-side up on the card.
        self.cam_tex.setRamImage(np.flipud(frame).tobytes())

    def _build_graphs(self):
        defs = [
            ("ATTITUDE (deg)", ["pitch", "roll", "yaw"]),
            ("GYRO (deg/s)", ["gx", "gy", "gz"]),
            ("POSITION (m)", ["x", "y", "z"]),
            ("VELOCITY (m/s)", ["vx", "vy", "vz"]),
            ("THRUST (g)", ["vert-g", "thrust"]),
        ]
        self.graph_panels = [RollingGraph(self.pixel2d, title, names) for title, names in defs]

    def _layout_graphs(self):
        win_w = max(self.win.getXSize(), 400)
        win_h = max(self.win.getYSize(), 300)
        margin, gap, panel_h = 10, 10, 118
        n = len(self.graph_panels)
        panel_w = (win_w - 2 * margin - (n - 1) * gap) / n
        y = win_h - margin - panel_h
        for i, g in enumerate(self.graph_panels):
            x = margin + i * (panel_w + gap)
            g.set_rect(x, y, panel_w, panel_h)

    # ── camera control ──────────────────────────────────────────────
    def _cycle_cam(self):
        self.cam_mode_idx = (self.cam_mode_idx + 1) % len(self.cam_modes)

    def _zoom(self, delta):
        self.cam_dist = max(2.0, min(20.0, self.cam_dist + delta))

    def _drag_start(self):
        self._drag_active = True
        md = self.win.getPointer(0)
        self._last_mouse = (md.getX(), md.getY())

    def _drag_stop(self):
        self._drag_active = False

    def _update_camera(self, dt, drone_pos, yaw_deg):
        mode = self.cam_modes[self.cam_mode_idx]
        if mode == "chase":
            yaw_r = math.radians(yaw_deg)
            back = Vec3(-math.sin(yaw_r) * self.cam_dist, -math.cos(yaw_r) * self.cam_dist, self.cam_dist * 0.5)
            target = drone_pos + back
            self.camera.setPos(self.camera.getPos() + (target - self.camera.getPos()) * min(1, dt * 3))
            self.camera.lookAt(drone_pos + Vec3(0, 0, 0.3))
        elif mode == "orbit":
            self.cam_yaw += dt * 12
            yaw_r = math.radians(self.cam_yaw)
            pitch_r = math.radians(self.cam_pitch)
            x = drone_pos.x + self.cam_dist * math.cos(pitch_r) * math.sin(yaw_r)
            y = drone_pos.y - self.cam_dist * math.cos(pitch_r) * math.cos(yaw_r)
            z = drone_pos.z + self.cam_dist * math.sin(pitch_r)
            self.camera.setPos(x, y, z)
            self.camera.lookAt(drone_pos)
        else:  # free
            if self._drag_active and self.mouseWatcherNode.hasMouse():
                md = self.win.getPointer(0)
                mx, my = md.getX(), md.getY()
                if self._last_mouse:
                    dx = mx - self._last_mouse[0]
                    dy = my - self._last_mouse[1]
                    self.cam_yaw -= dx * 0.3
                    self.cam_pitch = max(-80, min(80, self.cam_pitch - dy * 0.3))
                self._last_mouse = (mx, my)
            yaw_r = math.radians(self.cam_yaw)
            pitch_r = math.radians(self.cam_pitch)
            x = drone_pos.x + self.cam_dist * math.cos(pitch_r) * math.sin(yaw_r)
            y = drone_pos.y - self.cam_dist * math.cos(pitch_r) * math.cos(yaw_r)
            z = drone_pos.z + self.cam_dist * math.sin(pitch_r)
            self.camera.setPos(x, y, z)
            self.camera.lookAt(drone_pos)

    # ── retractable / aimable gun ─────────────────────────────────────
    def _gun_deploy(self):
        self.gun_fold_target = 0.0
        self.gun2_fold_target = 0.0

    def _gun_retract(self):
        self.gun_fold_target = 1.0
        self.gun2_fold_target = 1.0

    # ── palm-tracking control ────────────────────────────────────────
    def _toggle_hand_tracking(self):
        if self.hand_tracker is None:
            print("[AETHON-X] Palm tracking unavailable — install with:\n"
                  "           pip install mediapipe opencv-python")
            return
        if self.hand_tracking_on:
            self.hand_tracker.stop()
            self.hand_tracking_on = False
            self._hand_state = None
            for n in self.cam_preview_nodes:
                n.hide()
            print("[AETHON-X] Palm tracking OFF — mouse controls both arms again.")
        else:
            ok = self.hand_tracker.start()
            if not ok:
                print("[AETHON-X] Could not open the webcam.")
                return
            self.hand_tracking_on = True
            for n in self.cam_preview_nodes:
                n.show()
            print("[AETHON-X] Palm tracking ON — your right hand aims and "
                  "deploys the right arm, your left hand aims and deploys "
                  "the left arm (open hand = deploy/aim, fist = retract).\n"
                  "           Left/right follows your palm directly. For "
                  "forward/back and up/down, use gestures: point with just "
                  "your index finger = forward, just your pinky = back, "
                  "thumbs up = swing up, thumbs down = swing down (move "
                  "your palm while held for finer control).")

    def _on_quit(self):
        if self.hand_tracker is not None:
            self.hand_tracker.stop()
        sys.exit()

    def _get_mouse_screen_pos(self):
        """Returns a (x, y) pair in Panda3D's -1..1 mouse-coordinate
        convention from the real mouse, or None if the mouse isn't
        currently over the window."""
        if self.mouseWatcherNode is None or not self.mouseWatcherNode.hasMouse():
            return None
        mpos = self.mouseWatcherNode.getMouse()
        return mpos.getX(), mpos.getY()

    def _aim_point_from_screen(self, x, y, depth=None, reach_bias=0.0):
        """Casts a ray from the camera through a point given in Panda3D's
        -1..1 mouse-coordinate convention and returns a world-space point
        a fixed distance out along it. `depth` (0..1, bigger = closer),
        when given, pulls that distance in or out — used only in mouse
        mode; palm tracking no longer feeds a live depth value here since
        it's too unreliable (see the module docstring). `reach_bias`
        (-1..1) is the gesture-driven replacement for that in palm-
        tracking mode — point-finger/pinky gestures push it further out
        or pull it in (see _update_arm_bias). Returns None if the ray
        can't be built."""
        near_pt, far_pt = Point3(), Point3()
        if not self.camLens.extrude(Point2(x, y), near_pt, far_pt):
            return None
        near_w = self.render.getRelativePoint(self.camera, near_pt)
        far_w = self.render.getRelativePoint(self.camera, far_pt)
        ray_dir = far_w - near_w
        if ray_dir.length() < 1e-6:
            return None
        ray_dir.normalize()

        aim_dist = self.GUN_AIM_DISTANCE
        if depth is not None:
            # Reaching the hand toward the camera (bigger apparent size,
            # depth -> 1) brings the arm in closer.
            aim_dist = self.GUN_AIM_DISTANCE * (0.4 + 1.2 * (1.0 - depth))
        if reach_bias:
            aim_dist *= (1.0 + reach_bias * 0.6)
        aim_dist = max(1.0, aim_dist)

        return self.camera.getPos(self.render) + ray_dir * aim_dist

    def _update_arm_bias(self, hand, reach_bias, vert_bias, dt):
        """Advances one arm's gesture-driven forward/back (reach_bias) and
        up/down (vert_bias) state from this frame's hand gesture. Point
        with just the index finger to push the arm forward, just the
        pinky to pull it back; thumbs-up/thumbs-down to swing it up or
        down, with the palm's own vertical position (while that gesture
        is held) scaling how fast it moves — proportional control in that
        plane instead of a fixed-speed nudge. Any other gesture (open
        palm, fist, "none", ...) leaves both values exactly where they
        were: they hold position rather than snapping back to centre."""
        gesture = hand.get("gesture", "none")
        if gesture == "point_index":
            reach_bias = min(1.0, reach_bias + self.ARM_REACH_SPEED * dt)
        elif gesture == "point_pinky":
            reach_bias = max(-1.0, reach_bias - self.ARM_REACH_SPEED * dt)
        elif gesture == "thumb_up":
            rate = self.ARM_VERT_SPEED * (1.0 + abs(hand["y"]))
            vert_bias = min(1.0, vert_bias + rate * dt)
        elif gesture == "thumb_down":
            rate = self.ARM_VERT_SPEED * (1.0 + abs(hand["y"]))
            vert_bias = max(-1.0, vert_bias - rate * dt)
        return reach_bias, vert_bias

    def _update_gun(self, dt):
        if self.gun_j1 is None and self.gun2_j1 is None:
            return

        if self.hand_tracking_on and self.hand_tracker is not None:
            # Palm-tracking mode: each arm is driven independently by its
            # own hand — right arm by the right palm, left arm (the
            # mirrored rig) by the left palm. A clearly open hand deploys
            # its arm and a clearly closed fist retracts it. This uses
            # hysteresis (two thresholds, not one) rather than mapping
            # curl straight onto the fold amount — with a 1:1 mapping,
            # any partial curl (natural tremor, fingers not perfectly
            # flat) pushed the fold target above zero, which made
            # "deployed" nearly impossible to hold. With hysteresis, once
            # deployed an arm stays deployed through normal finger
            # motion and only retracts on a real fist — independently
            # per hand, so one hand can be a fist while the other is open.
            state = self.hand_tracker.snapshot()
            self._hand_state = state
            right_hand = state["right"]
            left_hand = state["left"]

            if right_hand["active"]:
                curl = right_hand["curl"]
                if curl > self.GUN_CURL_RETRACT:
                    self.gun_fold_target = 1.0
                elif curl < self.GUN_CURL_DEPLOY:
                    self.gun_fold_target = 0.0
                self.reach_bias_r, self.vert_bias_r = self._update_arm_bias(
                    right_hand, self.reach_bias_r, self.vert_bias_r, dt)
            if left_hand["active"]:
                curl = left_hand["curl"]
                if curl > self.GUN_CURL_RETRACT:
                    self.gun2_fold_target = 1.0
                elif curl < self.GUN_CURL_DEPLOY:
                    self.gun2_fold_target = 0.0
                self.reach_bias_l, self.vert_bias_l = self._update_arm_bias(
                    left_hand, self.reach_bias_l, self.vert_bias_l, dt)
        else:
            # Mouse mode: a single shared control drives both arms
            # together, same as before palm tracking existed.
            self.gun2_fold_target = self.gun_fold_target

        # Animate each arm's fold progress toward its own target.
        def _advance(current, target_):
            diff = target_ - current
            step = self.GUN_FOLD_SPEED * dt
            if abs(diff) <= step:
                return target_
            return current + (step if diff > 0 else -step)

        self.gun_fold = _advance(self.gun_fold, self.gun_fold_target)
        self.gun2_fold = _advance(self.gun2_fold, self.gun2_fold_target)

        f1, f2 = self.gun_fold, self.gun2_fold

        # Only track the aim source once (nearly) fully deployed; otherwise
        # play the fold pose so retracting/deploying looks clean.
        if self.hand_tracking_on and self.hand_tracker is not None:
            right_hand = self._hand_state["right"]
            left_hand = self._hand_state["left"]
            target_r = None
            wrist_r = None
            if f1 <= 0.02 and right_hand["active"]:
                target_r = self._aim_point_from_screen(
                    right_hand["x"], self.vert_bias_r,
                    reach_bias=self.reach_bias_r)
                wrist_r = right_hand["wrist_deg"]
            target_l = None
            wrist_l = None
            if f2 <= 0.02 and left_hand["active"]:
                target_l = self._aim_point_from_screen(
                    left_hand["x"], self.vert_bias_l,
                    reach_bias=self.reach_bias_l)
                wrist_l = left_hand["wrist_deg"]
        else:
            # Mouse mode: one shared aim point drives both arms identically.
            mouse_pos = self._get_mouse_screen_pos()
            shared_target = None
            if mouse_pos is not None and (f1 <= 0.02 or f2 <= 0.02):
                shared_target = self._aim_point_from_screen(*mouse_pos)
            target_r = None if f1 > 0.02 else shared_target
            target_l = None if f2 > 0.02 else shared_target
            wrist_r = wrist_l = None

        self._pose_arm(self.gun_j1, self.gun_j2, self.gun_j4, self.gun_j6,
                        f1, target_r, wrist_r)
        self._pose_arm(self.gun2_j1, self.gun2_j2, self.gun2_j4, self.gun2_j6,
                        f2, target_l, wrist_l)

    def _pose_arm(self, j1, j2, j4, j6, f, target, wrist_deg=None):
        """Applies fold/aim state to a single arm rig's four joints.
        Called once per arm with that arm's own fold amount, aim target,
        and (in palm-tracking mode) its own hand's wrist rotation — so
        the two arms can move identically (mouse mode) or independently
        (palm-tracking mode) depending on what the caller passes in."""
        if j1 is None:
            return

        if f > 0.02:
            j1.setQuat(Quat())
            if j2 is not None:
                j2.setP(f * self.GUN_FOLD_J2_DEG)
            if j4 is not None:
                j4.setP(f * self.GUN_FOLD_J4_DEG)
            if j6 is not None:
                j6.setP(f * self.GUN_FOLD_J6_DEG)
            return

        if target is None:
            return
        shoulder_pos = j1.getPos(self.render)
        aim_dir_world = target - shoulder_pos
        if aim_dir_world.length() < 1e-4:
            return
        aim_dir_world.normalize()
        parent = j1.getParent()
        aim_dir_local = parent.getRelativeVector(self.render, aim_dir_world)
        aim_dir_local.normalize()

        # Base joint: a full ball-and-socket rotation from the rest pose
        # (local -Z, hanging straight down) to the aim direction. Because
        # this is a genuine 3D quaternion — not a single-axis hinge — it
        # can point anywhere on the sphere around the shoulder: left,
        # right, forward, back, up, or down, i.e. the full 360 degrees
        # around the base, not just up/down.
        quat = _shortest_arc_quat(Vec3(0, 0, -1), aim_dir_local)
        j1.setQuat(quat)

        # "reach" = how far the aim direction has swung away from the
        # rest-down pose (0 = hanging straight down, 1 = swung all the
        # way up). The other joints flex along with it so the whole arm
        # visibly articulates instead of staying locked once deployed.
        reach = 0.5 * (1.0 - aim_dir_local.z)  # z=-1 (down) -> 0, z=+1 (up) -> 1
        reach = max(0.0, min(1.0, reach))

        if j2 is not None:
            j2.setP(reach * self.GUN_AIM_J2_DEG)
        if j4 is not None:
            j4.setP(self.GUN_AIM_J4_BASE_DEG + reach * self.GUN_AIM_J4_REACH_DEG)
        if j6 is not None:
            j6.setP(reach * self.GUN_AIM_J6_DEG)
            # Palm-tracking bonus: roll the wrist joint to match this
            # arm's own hand's rotation about the camera axis.
            if wrist_deg is not None:
                j6.setR(wrist_deg)

    # ── leg throttle/pump animation ─────────────────────────────────
    def _toggle_legs(self):
        self.legs_moving = not self.legs_moving

    def _update_legs(self, dt):
        if not self.leg_nodes:
            return
        if self.legs_moving:
            self._leg_swing_t += dt * self.LEG_SWING_SPEED
            self._leg_angle = math.sin(self._leg_swing_t) * self.LEG_SWING_AMPLITUDE_DEG
        else:
            # ease back to neutral instead of snapping when stopped
            self._leg_angle += (0.0 - self._leg_angle) * min(1.0, dt * self.LEG_EASE_RATE)
        for leg in self.leg_nodes:
            leg.setP(self._leg_angle)

    # ── per-frame update ─────────────────────────────────────────────
    def _update(self, task):
        dt = globalClock.getDt()
        (ax_, ay_, az_, gx_, gy_, gz_, acx, acy, acz, vert_g,
         thr_x, thr_y, thr_z, landed,
         px_, py_, pz_, vx_, vy_, vz_) = self.reader.snapshot()

        drone_pos = Vec3(px_, py_, max(pz_, 0) + 1.0)
        self.drone.setPos(drone_pos)
        self.drone.setHpr(az_, ax_, ay_)

        thrust_mag = abs(vert_g)
        spin_rate = dt * (200 + thrust_mag * 500)
        for rotor in self.rotors:
            rotor.setR(rotor.getR() + spin_rate)
        for node, spin_dir in self.rotor_nodes:
            node.setH(node.getH() + spin_dir * spin_rate)

        self.shadow.setPos(drone_pos.x, drone_pos.y, 0.005)
        alt = max(0.05, drone_pos.z)
        shadow_scale = max(0.3, 1.0 - alt * 0.12)
        self.shadow.setScale(shadow_scale, shadow_scale, 1)
        self.shadow.setColorScale(1, 1, 1, max(0.1, 0.5 - alt * 0.05))

        self._update_camera(dt, drone_pos, az_)
        self._update_gun(dt)
        self._update_legs(dt)
        self._update_cam_preview()
        self.sky.setPos(self.camera.getPos(self.render))

        status_str, color = _aerial_status(ax_, ay_, vert_g, gz_, landed)
        self.status_txt.setText(status_str)
        self.status_txt.setFg(color)

        self.graph_panels[0].push([ax_, ay_, az_])
        self.graph_panels[1].push([gx_, gy_, gz_])
        self.graph_panels[2].push([px_, py_, pz_])
        self.graph_panels[3].push([vx_, vy_, vz_])
        self.graph_panels[4].push([vert_g, thrust_mag])
        self._layout_graphs()
        for g in self.graph_panels:
            g.redraw()

        self.fps_c += 1
        now = time.time()
        if now - self.last_t >= 1.0:
            self.fps_v = self.fps_c
            self.fps_c = 0
            self.last_t = now

        self.hud_txt.setText(
            f"AETHON-X TELEMETRY\n"
            f"status  {status_str}\n\n"
            f"pitch   {ax_:+6.2f} deg\n"
            f"roll    {ay_:+6.2f} deg\n"
            f"yaw     {az_:+6.2f} deg\n\n"
            f"gyroX   {gx_:+6.1f} deg/s\n"
            f"gyroY   {gy_:+6.1f} deg/s\n"
            f"gyroZ   {gz_:+6.1f} deg/s\n\n"
            f"vert-g  {vert_g:+6.3f} g\n"
            f"thrust  {thrust_mag:+6.3f} g\n\n"
            f"pos     {px_:+5.2f} {py_:+5.2f} {pz_:+5.2f} m\n"
            f"vel     {vx_:+5.2f} {vy_:+5.2f} {vz_:+5.2f} m/s\n\n"
            f"fps     {self.fps_v}\n"
            f"source  {self.src_label}"
        )
        return Task.cont


# ──────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────
def _default_model_path():
    """The bundled CesiumDrone.glb next to this script, if present."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "CesiumDrone.glb")
    return candidate if os.path.isfile(candidate) else None


def main():
    ap = argparse.ArgumentParser(description="AETHON-X Panda3D visualiser")
    ap.add_argument("--port", default=_default_port())
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--model", default=None,
                     help="Path to a .glb/.gltf/.egg/.bam drone model. "
                          "Defaults to the bundled CesiumDrone.glb next to "
                          "this script; falls back to a procedural hull if "
                          "no model is found or loading fails.")
    ap.add_argument("--no-model", action="store_true",
                     help="Force the procedural hull even if a model is available.")
    args = ap.parse_args()
    model_path = None if args.no_model else (args.model or _default_model_path())

    reader = IMUReader(args.port, args.baud, simulated=args.sim)
    reader.start()
    for _ in range(12):
        if reader.connected:
            break
        time.sleep(1.0)
    if not args.sim and not reader.connected:
        print("\n[AETHON-X ERROR]", reader.error_msg or "Connection failed.", file=sys.stderr)
        print("Available ports:", _list_ports() or "(none)", file=sys.stderr)
        reader.stop()
        sys.exit(1)

    src = "SIM" if args.sim else args.port
    app = AethonApp(reader, src, model_path=model_path)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        reader.join(timeout=1)


if __name__ == "__main__":
    main()
