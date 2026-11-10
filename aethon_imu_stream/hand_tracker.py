#!/usr/bin/env python3
"""
hand_tracker.py — Webcam dual-palm tracker for AETHON-X
====================================================
Runs MediaPipe's HandLandmarker (the current "Tasks" API — the older
mp.solutions.hands API has been removed from recent mediapipe releases)
on a background thread against the default webcam, tracking up to two
hands at once and classifying each as the user's left or right hand.
Exposes a thread-safe snapshot with one state dict per hand, in units
the Panda3D visualiser can consume directly:

    snapshot() -> {
        "left":  {active, x, y, depth, wrist_deg, curl, gesture},
        "right": {active, x, y, depth, wrist_deg, curl, gesture},
        "frame": ndarray or None,
    }

Per-hand fields (identical meaning to the original single-hand tracker):
    active     bool   True once this hand has been seen this frame
    x, y       float  palm centre in Panda3D's -1..1 mouse-coordinate
                       convention (same range MouseWatcherNode.getMouse()
                       returns)
    depth      float  0 (hand far / small in frame) .. 1 (hand close /
                       fills the frame) — a rough distance proxy from
                       apparent hand size. In practice this signal is
                       noisy (a webcam can't tell a hand moving toward
                       it from a hand that's just slightly bigger), so
                       treat it as a weak hint, not a reliable axis —
                       `gesture` below is the robust alternative.
    wrist_deg  float  rotation of the hand about the camera axis
                       (wrist -> middle-finger-knuckle vector angle)
    curl       float  0 (fingers open) .. 1 (fist)
    gesture    str    one of "open", "fist", "point_index", "point_pinky",
                       "thumb_up", "thumb_down", "none". A simple, robust
                       finger-shape classifier (see _classify_gesture)
                       meant as a stand-in for axes the camera can't read
                       reliably (forward/back, up/down): "point_index"/
                       "point_pinky" (only that finger extended) and
                       "thumb_up"/"thumb_down" (only the thumb extended,
                       pointing above/below the wrist) are unambiguous
                       hand shapes that don't depend on depth or precise
                       vertical tracking at all.

    frame      ndarray or None  latest annotated preview frame — an
                       RGB uint8 array of shape (PREVIEW_H, PREVIEW_W, 3)
                       with both tracked skeletons and readouts burned
                       in, ready to hand straight to a texture

Left/right classification: the frame is mirrored before detection (see
`mirror` below), which is MediaPipe's documented convention for a
front-facing/selfie camera — under that convention its "Left"/"Right"
handedness label already matches the user's own left/right hand
directly, no swap needed. If you run with mirror=False (an unflipped,
"security camera" style feed), the label is swapped automatically so
"left"/"right" still means the user's own hands.

Nothing here touches Panda3D — it only produces plain numbers (and one
annotated preview image) so it can be tested/tuned independently of the
3D app.

Install:
    pip install mediapipe opencv-python

The hand-landmark model (~12 MB) is downloaded automatically to this
file's folder the first time you run it, and cached there afterward.
"""

import math
import os
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)

# Size of the annotated preview frame handed back in state["frame"].
PREVIEW_W = 320
PREVIEW_H = 240

# 21-point MediaPipe hand skeleton edges (thumb, index, middle, ring,
# pinky, plus the palm base) — used to draw the tracked skeleton on the
# preview frame without depending on mp.solutions.drawing_utils, which
# expects the old solutions.hands landmark type rather than the Tasks
# API's landmark objects.
_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)

# Distinct skeleton colors per hand so the preview panel reads clearly
# at a glance which is which.
_HAND_COLOR = {
    "left":  (255, 210, 60),   # cyan-ish blue (BGR: blue-heavy)
    "right": (60, 255, 140),   # green (matches original single-hand color)
}
_NO_HAND_COLOR = (60, 60, 220)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "hand_landmarker.task")


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print("[hand_tracker] Downloading hand_landmarker.task model (~12 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[hand_tracker] Download complete.")
    return _MODEL_PATH


def _empty_hand_state():
    return {
        "active": False,
        "x": 0.0,
        "y": 0.0,
        "depth": 0.5,
        "wrist_deg": 0.0,
        "curl": 0.0,
        "gesture": "none",
    }


class HandTracker:
    def __init__(self, camera_index=0, mirror=True):
        self.camera_index = camera_index
        self.mirror = mirror

        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._cap = None
        self._landmarker = None

        self._state = {
            "left": _empty_hand_state(),
            "right": _empty_hand_state(),
            "frame": None,
        }

    # ── lifecycle ──────────────────────────────────────────────────
    def start(self):
        """Opens the webcam and starts the tracking thread. Returns False
        (and leaves everything untouched) if the camera can't be opened."""
        if self._running:
            return True

        model_path = _ensure_model()

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            self._cap = None
            return False

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                # Force CPU: MediaPipe's Metal/GPU delegate crashes
                # (DrishtiMetalHelper / TensorsToDetectionsCalculator abort)
                # on some macOS + Apple Silicon combinations. CPU is
                # plenty fast for two-hand tracking at 30fps.
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        with self._lock:
            self._state["left"]["active"] = False
            self._state["right"]["active"] = False

    @property
    def running(self):
        return self._running

    def snapshot(self):
        with self._lock:
            return {
                "left": dict(self._state["left"]),
                "right": dict(self._state["right"]),
                "frame": self._state["frame"],
            }

    # ── worker thread ─────────────────────────────────────────────
    def _run(self):
        start_time = time.monotonic()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            # Sort whatever hands were found this frame into "left"/"right"
            # buckets by their classified handedness. hands_seen maps
            # "left"/"right" -> landmark list, for at most one hand each
            # (if MediaPipe ever reports two hands with the same label,
            # the higher-confidence one wins).
            hands_seen = {}
            if result.hand_landmarks:
                for lm, handedness in zip(result.hand_landmarks, result.handedness):
                    top = handedness[0]
                    mp_label = top.category_name  # "Left" or "Right"
                    # MediaPipe assumes a mirrored (selfie-style) input;
                    # when we've already flipped the frame ourselves that
                    # assumption holds and the label already matches the
                    # user's own hand. If we're running unflipped, the
                    # label is reversed relative to the user's own body,
                    # so swap it.
                    user_hand = mp_label.lower() if self.mirror else (
                        "right" if mp_label == "Left" else "left")
                    if user_hand not in hands_seen or top.score > hands_seen[user_hand][1]:
                        hands_seen[user_hand] = (lm, top.score)

            for hand_key in ("left", "right"):
                if hand_key in hands_seen:
                    lm, _score = hands_seen[hand_key]
                    self._update_from_landmarks(hand_key, lm)
                else:
                    with self._lock:
                        self._state[hand_key]["active"] = False

            preview = self._render_preview(
                frame,
                {k: hands_seen[k][0] if k in hands_seen else None
                 for k in ("left", "right")},
            )

            with self._lock:
                self._state["frame"] = preview

            time.sleep(1.0 / 30.0)

    def _update_from_landmarks(self, hand_key, lm):
        # lm is a list of 21 NormalizedLandmark objects (.x, .y, .z), same
        # indexing as the old solutions.hands API:
        #   0 wrist, 5 index MCP, 9 middle MCP, 17 pinky MCP,
        #   8/12/16/20 index/middle/ring/pinky fingertips.
        wrist = lm[0]
        middle_mcp = lm[9]
        index_mcp = lm[5]
        pinky_mcp = lm[17]

        # Palm centre -> Panda3D mouse-coordinate convention (-1..1,
        # y flipped since image coordinates are top-down).
        px = (wrist.x + middle_mcp.x) / 2.0
        py = (wrist.y + middle_mcp.y) / 2.0
        nx = px * 2.0 - 1.0
        ny = 1.0 - py * 2.0

        # Depth proxy from palm width in normalised image coordinates.
        span = math.hypot(index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y)
        depth = max(0.0, min(1.0, (span - 0.05) / 0.25))

        wrist_deg = math.degrees(math.atan2(
            middle_mcp.x - wrist.x, -(middle_mcp.y - wrist.y)))

        curl = self._finger_curl(lm)
        gesture = self._classify_gesture(lm)

        with self._lock:
            self._state[hand_key].update(
                active=True, x=nx, y=ny, depth=depth,
                wrist_deg=wrist_deg, curl=curl, gesture=gesture,
            )

    def _render_preview(self, frame, hands_lm):
        """Builds the small annotated preview frame shown in the app's
        camera panel: both tracked skeletons (or a "no hand" state per
        side) plus the live detection signals, burned directly into the
        image so the panel is self-contained. Returns an RGB uint8
        ndarray sized PREVIEW_W x PREVIEW_H.
        hands_lm: {"left": lm or None, "right": lm or None}"""
        h, w = frame.shape[:2]
        out = frame.copy()

        any_tracking = False
        for hand_key, lm in hands_lm.items():
            if lm is None:
                continue
            any_tracking = True
            color = _HAND_COLOR[hand_key]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in _HAND_CONNECTIONS:
                cv2.line(out, pts[a], pts[b], color, 2, cv2.LINE_AA)
            for i, (x, y) in enumerate(pts):
                cv2.circle(out, (x, y), 6 if i == 0 else 4,
                           color, -1, cv2.LINE_AA)
            # Label near the wrist point.
            label = "L" if hand_key == "left" else "R"
            cv2.putText(out, label, (pts[0][0] - 8, pts[0][1] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        border_color = (140, 255, 0) if any_tracking else _NO_HAND_COLOR
        status = "TRACKING" if any_tracking else "NO HANDS"
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), border_color, 3)
        cv2.putText(out, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, border_color, 2, cv2.LINE_AA)

        with self._lock:
            left_s = dict(self._state["left"])
            right_s = dict(self._state["right"])

        lines = []
        for tag, s in (("L", left_s), ("R", right_s)):
            if s["active"]:
                lines.append(f"{tag} pos {s['x']:+.2f} {s['y']:+.2f}  "
                              f"depth {s['depth']:.2f}  curl {s['curl']:.2f}")
            else:
                lines.append(f"{tag} -- not tracked --")
        for i, line in enumerate(lines):
            cv2.putText(out, line, (10, h - 40 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

        small = cv2.resize(out, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _finger_curl(lm):
        """0 = fingers fully open, 1 = closed fist. Compares average
        fingertip distance from the wrist against the palm's own size."""
        wrist = lm[0]
        palm_ref = math.hypot(lm[9].x - wrist.x, lm[9].y - wrist.y)
        if palm_ref < 1e-6:
            return 0.0
        tips = (8, 12, 16, 20)  # index, middle, ring, pinky tips
        avg = sum(math.hypot(lm[t].x - wrist.x, lm[t].y - wrist.y)
                  for t in tips) / len(tips)
        ratio = avg / palm_ref            # ~2.0 open .. ~1.0 curled
        return max(0.0, min(1.0, (2.0 - ratio) / 1.0))

    @staticmethod
    def _fingers_extended(lm):
        """Returns [thumb, index, middle, ring, pinky] booleans. A finger
        counts as extended when its tip sits meaningfully farther from the
        wrist than its own base knuckle does — orientation-independent
        (works with the hand sideways, upside down, etc.), unlike a plain
        "is the tip above the knuckle" y-coordinate check."""
        wrist = lm[0]

        def dist(a, b):
            return math.hypot(a.x - b.x, a.y - b.y)

        palm_ref = dist(lm[9], wrist)
        if palm_ref < 1e-6:
            return [False] * 5
        tips = (4, 8, 12, 16, 20)
        bases = (2, 5, 9, 13, 17)
        return [dist(lm[t], wrist) > dist(lm[b], wrist) * 1.25
                for t, b in zip(tips, bases)]

    @classmethod
    def _classify_gesture(cls, lm):
        """Classifies one hand's landmarks into a small, deliberately
        unambiguous set of finger shapes — see the module docstring for
        what each one drives. Only ever returns one shape at a time (or
        "none"); shapes that don't cleanly match any of these (e.g. two
        fingers extended) fall through to "none" rather than guessing."""
        thumb, index, middle, ring, pinky = cls._fingers_extended(lm)
        if index and not (thumb or middle or ring or pinky):
            return "point_index"
        if pinky and not (thumb or index or middle or ring):
            return "point_pinky"
        if thumb and not (index or middle or ring or pinky):
            return "thumb_up" if lm[4].y < lm[0].y else "thumb_down"
        if thumb and index and middle and ring and pinky:
            return "open"
        if not (thumb or index or middle or ring or pinky):
            return "fist"
        return "none"


if __name__ == "__main__":
    # Quick standalone test: prints both hands' tracked values to the
    # terminal so you can verify the webcam + MediaPipe pipeline before
    # wiring it into the Panda3D app.
    tracker = HandTracker()
    if not tracker.start():
        print("Could not open the webcam.")
        raise SystemExit(1)
    print("Tracking... show one or both hands to the camera. Ctrl+C to stop.")
    try:
        while True:
            s = tracker.snapshot()
            parts = []
            for tag, key in (("L", "left"), ("R", "right")):
                hs = s[key]
                if hs["active"]:
                    parts.append(f"{tag} x={hs['x']:+.2f} y={hs['y']:+.2f} "
                                  f"depth={hs['depth']:.2f} "
                                  f"wrist={hs['wrist_deg']:+6.1f} curl={hs['curl']:.2f} "
                                  f"gesture={hs['gesture']}")
                else:
                    parts.append(f"{tag} --")
            print(" | ".join(parts) + " " * 10, end="\r")
            time.sleep(1.0 / 20.0)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        print("\nStopped.")
