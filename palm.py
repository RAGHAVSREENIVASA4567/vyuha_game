#!/usr/bin/env python3
"""
palm_bridge.py — OpenCV/MediaPipe webcam palm-tracking bridge for the
VYUHA-X Terrain Run browser game
=====================================================================
Runs entirely separately from vyuha_bridge.py (the flight-telemetry
bridge) and on its own WebSocket port. That's deliberate: this bridge
only ever supplies two things — where you're aiming, and when you
fired — never flight steering. In the game, palm tracking layers on
top of whichever "Control source" is flying the drone (normally
Mouse/keyboard, which still handles WASD/tilt steering); it just
replaces mouse-look with your hand and click/X with a pinch.

What it does, frame by frame:
  1. Grabs a webcam frame with OpenCV.
  2. Runs MediaPipe's HandLandmarker (the current "Tasks" API — the
     older mp.solutions.hands API used by earlier versions of this
     file has been removed from recent mediapipe releases, which
     forced pinning to an old mediapipe version; this Tasks-API
     version runs on any current mediapipe) to find BOTH hands' 21
     landmarks each, same dual-hand tracking as AETHON-X's
     hand_tracker.py. The two hands now have fixed, separate jobs
     instead of one hand doing everything: your LEFT hand steers aim
     and your RIGHT hand pinches to fire. Each is only read for its
     own job — the left hand's pinch is ignored, the right hand's
     tilt/height is ignored — so you can rest either hand without it
     accidentally doing the other's thing.
  3. Takes the average of the wrist + four knuckle landmarks as a
     stable "palm center" point (steadier than tracking a fingertip),
     and separately the wrist->middle-knuckle vector's angle for sideways hand roll — both from the LEFT hand only.
  4. Maps the LEFT palm's sideways ROLL (wrist-to-middle-finger axis
     tilted left/right like a waving gesture) to YAW, while palm HEIGHT in
     frame controls PITCH. This avoids requiring the wrist to twist until
     the palm faces left/right. About 55° of comfortable sideways tilt
     reaches the configured horizontal aim limit. Both are ABSOLUTE mappings
     on purpose: return the hand upright / recenter palm height and aim
     recenters immediately.
  5. Watches the distance between thumb tip and index fingertip of the
     RIGHT hand only (normalized by that hand's own size, so it works
     at any distance from the camera). Pinch the two together and
     release — a single edge-triggered "fire" event goes out per
     pinch, the same way a real trigger squeeze fires once per pull
     rather than once per frame it's held. Firing only ever does
     anything in-game once the arms are deployed (G) and pistols drawn
     (F) — same gate as mouse/keyboard — so pinching with the right
     hand before then is harmless.
  6. Broadcasts {handFound, lookYaw, lookPitch, fire} as JSON over a
     local WebSocket, same broadcast-to-all-clients pattern as
     vyuha_bridge.py. handFound reflects the LEFT (aim) hand
     specifically — see below.

Palm motions, once it's running:
  - LEFT hand aims, and only aims: TILT your open left palm sideways
    left/right (like a gentle waving gesture, while keeping the palm facing
    the camera) to look/aim left/right — this is YAW. Move your left palm
    UP/DOWN in frame to look up/down — this is PITCH. The left hand's pinch is also read as a fire gesture.
  - GREY/LEFT hand can fire: PINCH your left thumb and index
    fingertip together, then apart, to fire one shot (needs the arms
    deployed with G and pistols drawn with F in-game first, same as
    mouse/keyboard). The right hand's tilt/height never move the aim; its pinch remains a fallback fire gesture.
  - Keep whichever hand(s) you're using roughly open and facing the
    camera; a closed fist or an edge-on hand won't have 21 landmarks
    to find and will read as "no hand" for that role.
  - If the LEFT hand's tracking drops (hand out of frame, bad
    lighting, etc.) aim freezes at its last value rather than snapping
    to center, and the game's mouse-look automatically takes over as a
    fallback in the meantime. If the RIGHT hand drops, firing simply
    can't trigger until it's back — aim is unaffected either way,
    since the two hands are read completely independently.

A front-facing webcam physically can't see behind you, so this is a
generous look CONE (~yaw-range/pitch-range degrees each way from
center), not literal 360° like the mouse's pointer-lock look — call
out --yaw-range/--pitch-range explicitly if you want it wider.

The hand-landmark model (~7.5 MB) ships alongside this file as
hand_landmarker.task and is used directly — nothing is downloaded at
runtime. If that file is ever missing, it's fetched automatically to
this file's folder and cached there.

Usage:
    pip install opencv-python mediapipe websockets
    python3 palm_bridge.py                   # webcam 0, defaults
    python3 palm_bridge.py --camera 1         # different webcam
    python3 palm_bridge.py --sim              # no camera/mediapipe — fake sweep, for testing the pipeline
    python3 palm_bridge.py --headless         # skip sending the camera preview to the browser (tracking-only, saves bandwidth/CPU)
    python3 palm_bridge.py --ws-port 8766     # change the WebSocket port (default 8766)
    python3 palm_bridge.py --yaw-range 90 --pitch-range 55
    python3 palm_bridge.py --no-mirror        # if your webcam feed isn't already mirrored

Then in the game's terrain-select menu, set "Control source" to
"Webcam (OpenCV palm aim)" and leave the address as
ws://localhost:8766 (default). No separate window opens on this
machine — the camera view (with the tracked hand skeleton drawn on
it) streams over the same WebSocket and shows up in-game, in a small
preview panel in the corner of the browser.
"""

import argparse
import asyncio
import base64
import json
import math
import os
import threading
import time
import urllib.request

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, RunningMode,
    )
except ImportError:
    mp = None

try:
    import websockets
except ImportError:
    websockets = None


# Landmark indices from MediaPipe's 21-point hand model (same indexing
# under the Tasks API as the old solutions.hands API).
_WRIST = 0
_INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP = 5, 9, 13, 17
_THUMB_TIP, _INDEX_TIP = 4, 8

# 21-point hand skeleton edges, used to draw the tracked skeleton on
# the debug preview frame ourselves — mp.solutions.drawing_utils
# expects the old solutions.hands landmark type, not the Tasks API's.
_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)

# Central fraction of the frame that maps to the FULL pitch range —
# you don't need to jam your palm into the physical top/bottom edge
# of frame to hit max angle. 0.8 means the outer 10% on each side is
# already "fully deflected", similar to a joystick's soft-edge deadzone.
_ACTIVE_FRACTION = 0.8
# Palm orientation is converted directly to a spherical aim vector.
# These limits are only for elevation; azimuth is full 360 degrees.
_MAX_PALM_PITCH_DEG = 89.0
# Closed-fist detector thresholds.
_DEFAULT_FIST_CURL_RATIO = 0.92
_DEFAULT_FIST_OPEN_RATIO = 1.02

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "hand_landmarker.task")


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print("[palm-bridge] Downloading hand_landmarker.task model (~7.5 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[palm-bridge] Download complete.")
    return _MODEL_PATH


class PalmTracker:
    """Wraps a webcam + MediaPipe's HandLandmarker and reduces each frame
    down to the handful of numbers the game actually needs: aim angles
    and a one-shot fire edge. Both hands are tracked (left/right,
    dual-hand), but each has a fixed, separate job now: the LEFT hand
    drives aim (yaw/pitch) and the RIGHT hand drives fire (pinch) —
    never the other way around, and never both from one hand. That
    keeps the WebSocket payload and game.js side unchanged (still a
    single {handFound, lookYaw, lookPitch, fire} stream) while making
    the two actions fully independent: aim keeps tracking (or stays
    frozen at its last value) regardless of whether the fire hand is
    visible, and vice versa.
    """

    def __init__(self, camera_index, yaw_range, pitch_range, fist_curl_ratio, fist_open_ratio, mirror):
        if cv2 is None:
            raise SystemExit("opencv-python not installed — run: pip install opencv-python")
        if mp is None:
            raise SystemExit("mediapipe not installed — run: pip install mediapipe")

        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.fist_curl_ratio = fist_curl_ratio
        self.fist_open_ratio = fist_open_ratio
        self.mirror = mirror

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise SystemExit(
                f"Could not open webcam index {camera_index} — try a different --camera index, "
                "and make sure no other app is holding the camera."
            )

        model_path = _ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                # Force CPU: MediaPipe's Metal/GPU delegate crashes
                # (DrishtiMetalHelper / TensorsToDetectionsCalculator
                # abort) on some macOS + Apple Silicon combinations.
                # CPU is plenty fast for two-hand tracking at 30fps.
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._start_time = time.monotonic()

        # Which hand key drives which job. Fixed by role, not by
        # confidence/visibility — unlike the old single-active-hand
        # scheme, there's nothing to pick each frame here.
        self._AIM_KEY = "left"
        self._FIRE_KEY = "right"

        self._was_fist = False
        # Held rather than reset on a dropped frame, so losing tracking
        # for a moment freezes aim instead of snapping it to center.
        self.last_yaw = 0.0
        self.last_pitch = 0.0

    def close(self):
        self._landmarker.close()
        self.cap.release()

    def read_frame(self):
        """Grabs and processes exactly one frame. Returns
        (frame_bgr_for_display, result_dict). frame_bgr_for_display is
        None if the frame grab itself failed (camera hiccup) — result
        still comes back with handFound False in that case so the
        caller can keep the broadcast loop running.
        """
        ok, frame = self.cap.read()
        if not ok:
            return None, {"handFound": False, "lookYaw": self.last_yaw,
                           "lookPitch": self.last_pitch, "fire": False}

        if self.mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        # Sort whatever hands were found this frame into "left"/"right"
        # buckets by MediaPipe's own handedness label, same convention
        # as AETHON-X's hand_tracker.py: with a mirrored (selfie-style)
        # frame — the default here — the label already matches the
        # user's own hand directly; unmirrored, it's swapped.
        hands_seen = {}
        if result.hand_landmarks:
            for lm, handedness in zip(result.hand_landmarks, result.handedness):
                top = handedness[0]
                mp_label = top.category_name  # "Left" or "Right"
                user_hand = mp_label.lower() if self.mirror else (
                    "right" if mp_label == "Left" else "left")
                if user_hand not in hands_seen or top.score > hands_seen[user_hand][1]:
                    hands_seen[user_hand] = (lm, top.score)

        hand_found = False
        fire = False
        # LEFT hand: aim only. Reading this independently of the fire
        # hand means aim keeps updating (or stays frozen at its last
        # value if this hand isn't visible) no matter what the right
        # hand is doing.
        aim_lm = hands_seen.get(self._AIM_KEY)
        if aim_lm is not None:
            lm = aim_lm[0]  # list of 21 NormalizedLandmark, same indexing as before
            hand_found = True

            # AIM CONTROL: horizontal aiming uses the palm's SIDEWAYS ROLL —
            # a comfortable waving gesture — rather than turning the palm to
            # face left/right. Vertical aiming keeps the already-correct 3-D
            # palm-elevation calculation from the previous version.
            wrist = lm[_WRIST]
            middle_mcp = lm[_MIDDLE_MCP]

            # --- YAW: wrist -> middle finger axis in the image plane ---
            # Fingers straight up = neutral.  Tilting them toward screen-right
            # gives positive yaw (game look-right); screen-left gives negative
            # yaw. This is a roll/wave motion and does NOT require wrist twisting.
            dx = middle_mcp.x - wrist.x
            dy = middle_mcp.y - wrist.y
            palm_len = math.hypot(dx, dy)
            if palm_len > 1e-5:
                roll_deg = math.degrees(math.atan2(dx, -dy))
                # Invert the horizontal mapping so a palm tilt to the
                # user's right moves the arm/look to the user's right.
                yaw = -(roll_deg / 55.0) * self.yaw_range
                yaw = max(-self.yaw_range, min(self.yaw_range, yaw))
            else:
                yaw = self.last_yaw

            # --- PITCH: preserve the working 3-D palm elevation mapping ---
            index_mcp = lm[_INDEX_MCP]
            pinky_mcp = lm[_PINKY_MCP]
            ax = index_mcp.x - wrist.x
            ay = index_mcp.y - wrist.y
            az = index_mcp.z - wrist.z
            bx = pinky_mcp.x - wrist.x
            by = pinky_mcp.y - wrist.y
            bz = pinky_mcp.z - wrist.z
            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx
            nlen = math.sqrt(nx * nx + ny * ny + nz * nz)

            if nlen > 1e-6:
                nx /= nlen; ny /= nlen; nz /= nlen
                # Fixed winding retained from the last working vertical fix.
                nx, ny, nz = -nx, -ny, -nz
                pitch = -math.degrees(math.asin(max(-1.0, min(1.0, ny))))
                # 2x vertical sensitivity while retaining the configured
                # maximum pitch range as the safety/output limit.
                pitch = pitch * 2.0
                pitch = max(-self.pitch_range, min(self.pitch_range, pitch))
            else:
                pitch = self.last_pitch

            self.last_yaw = yaw
            self.last_pitch = pitch

        # GREEN/RIGHT hand: fire only. The grey/left hand is never
        # read for firing. A closed fist creates one edge-triggered shot.
        fire_lm = hands_seen.get(self._FIRE_KEY)

        if fire_lm is not None:
            lm = fire_lm[0]
            wrist = lm[_WRIST]
            middle_mcp = lm[_MIDDLE_MCP]
            hand_span = math.hypot(middle_mcp.x - wrist.x,
                                   middle_mcp.y - wrist.y) or 1e-6

            # Index, middle, ring and pinky: compare fingertip distance
            # from the wrist with the corresponding PIP distance.
            finger_pairs = (
                (8, 6), (12, 10), (16, 14), (20, 18)
            )

            def count_extended():
                count = 0
                for tip_idx, pip_idx in finger_pairs:
                    tip_dist = math.hypot(lm[tip_idx].x - wrist.x,
                                          lm[tip_idx].y - wrist.y) / hand_span
                    pip_dist = math.hypot(lm[pip_idx].x - wrist.x,
                                          lm[pip_idx].y - wrist.y) / hand_span
                    if tip_dist > pip_dist * self.fist_open_ratio:
                        count += 1
                return count

            def count_curled():
                count = 0
                for tip_idx, pip_idx in finger_pairs:
                    tip_dist = math.hypot(lm[tip_idx].x - wrist.x,
                                          lm[tip_idx].y - wrist.y) / hand_span
                    pip_dist = math.hypot(lm[pip_idx].x - wrist.x,
                                          lm[pip_idx].y - wrist.y) / hand_span
                    if tip_dist < pip_dist * self.fist_curl_ratio:
                        count += 1
                return count

            # 3+ curled fingers = closed fist. Hysteresis requires 3+
            # clearly extended fingers before the fist can reset.
            fist = count_curled() >= 3
            if self._was_fist:
                fist = count_extended() < 3

            if fist and not self._was_fist:
                fire = True
            self._was_fist = fist
        else:
            # Tracking loss resets the gesture so the next detected fist
            # is a fresh trigger.
            self._was_fist = False

        if frame is not None:
            h, w = frame.shape[:2]
            for hand_key, (lm, _score) in hands_seen.items():
                # Grey = aim hand (left), green = fire hand (right) —
                # a fixed role color now, not "which hand is active".
                color = (140, 140, 140) if hand_key == self._AIM_KEY else (60, 220, 60)
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                for a, b in _HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
                for i, (x, y) in enumerate(pts):
                    cv2.circle(frame, (x, y), 6 if i == 0 else 4, color, -1, cv2.LINE_AA)

        return frame, {
            "handFound": hand_found,
            "lookYaw": round(self.last_yaw, 2),
            "lookPitch": round(self.last_pitch, 2),
            "fire": fire,
            "fist": self._was_fist,
        }

    # Aim is now derived directly from the 3-D palm normal above, so
    # there is intentionally no separate wrist-tilt/palm-height mapper.


def _sim_frame(t, yaw_range, pitch_range):
    """No-camera test mode: sweeps a fake palm in a slow figure-eight
    and pinches briefly once per cycle, so you can sanity-check the
    game side (menu wiring, HUD, firing) without a working webcam.
    """
    yaw = math.sin(t * 0.6) * yaw_range * 0.8
    pitch = math.sin(t * 1.2) * pitch_range * 0.5
    fire = (t % 3.0) < 0.08  # a short pulse once every ~3 seconds
    return {"handFound": True, "lookYaw": round(yaw, 2), "lookPitch": round(pitch, 2), "fire": fire}


# ──────────────────────────────────────────────────────────────────
#  WebSocket server — same broadcast-to-all-clients shape as
#  vyuha_bridge.py, just fed by the palm tracker above instead of a
#  serial IMU stream.
# ──────────────────────────────────────────────────────────────────
async def _broadcast_loop(get_latest, clients: set, hz: float):
    period = 1.0 / hz
    while True:
        if clients:
            payload = json.dumps(get_latest())
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
    print(f"[palm-bridge] game connected ({len(clients)} client(s))")
    try:
        async for _msg in ws:
            pass  # game doesn't send anything back currently
    finally:
        clients.discard(ws)
        print(f"[palm-bridge] game disconnected ({len(clients)} client(s))")


# Width (px) the debug preview is downscaled to before JPEG-encoding.
# It goes out over the WebSocket ~30x/sec alongside the tracking data,
# so this trades preview sharpness for bandwidth and browser decode
# cost — plenty legible as a small corner picture-in-picture.
_PREVIEW_WIDTH = 320
_PREVIEW_JPEG_QUALITY = 55


def _run_capture_loop(tracker, send_preview, latest_holder, stop_evt):
    """Runs on the main thread — grabs frames, draws the debug overlay
    (tracked skeleton + status text) on them, and updates
    latest_holder[0] with the tracking result until stop_evt is set or
    Ctrl+C. No native window is opened: when send_preview is True the
    overlaid frame is downscaled and JPEG-encoded into the result
    dict's "frame" field, which _broadcast_loop then ships to the
    browser alongside the yaw/pitch/fire data — that's what renders as
    the in-game camera preview. Pass send_preview=False (--headless)
    to skip that encode/broadcast step entirely and save CPU/bandwidth
    when you don't need to see the feed (tracking still runs).
    """
    while not stop_evt.is_set():
        frame, result = tracker.read_frame()

        if send_preview and frame is not None:
            status = "GREY AIM" if result["handFound"] else "no grey aim hand"
            fist = "GREEN FIST / FIRE" if result.get("fist") else ""
            cv2.putText(frame, f"{status}  yaw={result['lookYaw']:+.1f}  pitch={result['lookPitch']:+.1f}  {fist}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (int(w * 0.1), int(h * 0.1)), (int(w * 0.9), int(h * 0.9)), (60, 160, 220), 1)

            scale = _PREVIEW_WIDTH / w
            small = cv2.resize(frame, (_PREVIEW_WIDTH, max(1, int(h * scale))))
            ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, _PREVIEW_JPEG_QUALITY])
            if ok:
                result["frame"] = base64.b64encode(buf).decode('ascii')

        latest_holder[0] = result


async def _server_async(args, latest_holder, stop_evt_threading):
    if websockets is None:
        raise SystemExit("websockets not installed — run: pip install websockets")

    clients = set()

    async def handler(ws):
        await _handler(ws, clients)

    server = await websockets.serve(handler, "localhost", args.ws_port)
    print(f"[palm-bridge] WebSocket server listening on ws://localhost:{args.ws_port}")
    print("[palm-bridge] in the game menu, set Control source to 'Webcam (OpenCV palm aim)'.")

    broadcast_task = asyncio.ensure_future(_broadcast_loop(lambda: latest_holder[0], clients, hz=30.0))

    try:
        while not stop_evt_threading.is_set():
            await asyncio.sleep(0.05)
    finally:
        broadcast_task.cancel()
        server.close()
        await server.wait_closed()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    ap.add_argument("--ws-port", type=int, default=8766,
                     help="WebSocket port the game connects to (default 8766, distinct from "
                          "vyuha_bridge.py's 8765 so both can run at once).")
    ap.add_argument("--yaw-range", type=float, default=70.0,
                     help="Degrees of look-left/right at the edge of the active frame area (default 70).")
    ap.add_argument("--pitch-range", type=float, default=45.0,
                     help="Degrees of look-up/down at the edge of the active frame area (default 45).")
    ap.add_argument("--fist-curl-ratio", type=float, default=_DEFAULT_FIST_CURL_RATIO,
                     help=f"Closed-finger curl ratio for firing (default {_DEFAULT_FIST_CURL_RATIO}).")
    ap.add_argument("--fist-open-ratio", type=float, default=_DEFAULT_FIST_OPEN_RATIO,
                     help=f"Open-hand reset ratio for firing (default {_DEFAULT_FIST_OPEN_RATIO}).")
    ap.add_argument("--no-mirror", action="store_true",
                     help="Disable the horizontal flip (on by default, since most webcam feeds read like a mirror).")
    ap.add_argument("--headless", action="store_true",
                     help="Don't send the camera preview (landmarks/aim/pinch overlay) to the "
                          "browser — tracking still runs, this just saves bandwidth/CPU.")
    ap.add_argument("--sim", action="store_true",
                     help="Generate a fake sweeping palm instead of reading a webcam — no OpenCV/MediaPipe needed.")
    args = ap.parse_args()

    latest_holder = [{"handFound": False, "lookYaw": 0.0, "lookPitch": 0.0, "fire": False, "fist": False}]
    stop_evt_threading = threading.Event()

    # The websocket server runs on a background thread with its own
    # asyncio event loop. cv2.imshow (below) needs the real main
    # thread on macOS, so that's the side that stays there — this is
    # the reverse of an earlier version of this file, which ran the
    # capture/display loop in a worker thread instead and crashed
    # OpenCV's HighGUI on macOS.
    server_thread = threading.Thread(
        target=lambda: asyncio.run(_server_async(args, latest_holder, stop_evt_threading)),
        daemon=True,
    )
    server_thread.start()

    try:
        if args.sim:
            print("[palm-bridge] running in --sim mode (no camera/mediapipe) — sweeping a fake palm.")
            t0 = time.time()
            while not stop_evt_threading.is_set():
                latest_holder[0] = _sim_frame(time.time() - t0, args.yaw_range, args.pitch_range)
                time.sleep(1 / 30.0)
        else:
            tracker = PalmTracker(args.camera, args.yaw_range, args.pitch_range,
                                   args.fist_curl_ratio, args.fist_open_ratio,
                                   mirror=not args.no_mirror)
            try:
                _run_capture_loop(tracker, not args.headless, latest_holder, stop_evt_threading)
            finally:
                tracker.close()
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt_threading.set()
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
