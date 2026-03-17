#!/usr/bin/env python3
"""
Data Collection Script — Obstacle Avoidance Neural Network Training
====================================================================
Reads raw frames from the unified image shared memory, captures human
keyboard steering input, runs YOLO + LKAS lane detection per frame,
and saves a labelled driving dataset with per-frame mode annotations.

Run alongside:
  - LKAS detection server  (creates image + detection SHM)
  - LKAS decision server   (creates control SHM)
  - Stop yolo_depth_avoidance.py — this script runs its own YOLO to
    avoid loading two YOLO models on the same GPU simultaneously.
  - vehicle.py must be STOPPED — this script controls JetRacer directly

Controls:
  ← / →   steer left / right  (accumulates ±STEER_STEP per frame, resets on release)
  ↓        throttle = 0  (full stop)
  space    release locked mode back to NORMAL
  q        quit and save

Data layout (relative to this script):
  data/
  ├── images/
  │   ├── 000001.jpg   (raw colour, no overlays)
  │   ├── 000001.npy   (raw depth uint16)
  │   └── ...
  └── labels.csv       (frame_id, input_steering, input_throttle, mode)

Mode values: NORMAL | MICRO_ADJUST | LANE_CHANGE | STOP
"""

import sys
import time
import signal
import csv
import argparse
import threading
import tty
import termios
import select
import importlib.util
import numpy as np
import cv2
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
# lkas is installed as an editable package — do NOT add ads-skynet/ to sys.path,
# it would shadow the installed package with a namespace package at lkas/.
# vehicle/src is NOT an installed package, so add it explicitly for Camera import.
sys.path.insert(0, str(script_dir.parent.parent / "vehicle" / "src"))

# ── PyTorch legacy weights fix (must happen before ultralytics import) ────────
import torch
_orig_torch_load = torch.load

def _torch_load_legacy(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_legacy

from ultralytics import YOLO as _YOLO

# ── RealSense camera (vehicle/src/camera.py) ─────────────────────────────────
from camera import Camera

# ── JetRacer ──────────────────────────────────────────────────────────────────
try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    JETRACER_AVAILABLE = True
except ImportError:
    print("[WARN] JetRacer library not available — running in simulation mode (no motor output)")
    JETRACER_AVAILABLE = False

# ── LKAS shared memory + client ───────────────────────────────────────────────
try:
    from lkas.integration.shared_memory import (
        SharedMemoryImageChannel,
        SharedMemoryControlChannel,
    )
    from lkas import LKASClient as _LKASClient
    LKAS_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] LKAS not available: {e}")
    sys.exit(1)

# ── Web viewer ────────────────────────────────────────────────────────────────
from yolo_web_viewer import YOLOWebViewer

# ── Load YOLO config from Yolo_object_detection/config.py ─────────────────────
_config_path = script_dir.parent / "config.py"
_spec = importlib.util.spec_from_file_location("yolo_config", _config_path)
_cfg = importlib.util.module_from_spec(_spec)
_cfg.__file__ = str(_config_path)
_spec.loader.exec_module(_cfg)

MODEL_PATH           = _cfg.MODEL_PATH
CONFIDENCE_THRESHOLD = _cfg.CONFIDENCE_THRESHOLD
IOU_THRESHOLD        = _cfg.IOU_THRESHOLD
CLASS_NAMES          = _cfg.CLASS_NAMES
YOLO_DEVICE          = 0 if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_THROTTLE     = 0.20   # Normal driving throttle (auto-forward, human steers only)
OBSTACLE_THROTTLE = 0.10   # Reduced throttle when an obstacle is active
STEER_STEP        = 0.10   # Steering accumulated per frame while key held
SAVE_FPS          = 10     # Maximum frames written per second
YOLO_SKIP         = 2      # Run YOLO inference every N frames (memory pressure)
MODE_CONFIRM_FRAMES = 3    # Consecutive frames a candidate mode must hold before locking

# Mode decision thresholds
DECISION_ENTER    = 0.7    # metres — obstacle closer than this triggers mode
DECISION_EXIT     = 1.0    # metres — hysteresis, mode clears above this
MICRO_THRESHOLD   = 0.33   # lane coverage fraction below which → MICRO_ADJUST
ADJACENT_MAX_DIST = 2.0    # metres — scan this far for adjacent-lane obstacles

# Lane fallback (pixels) when LKAS has no lane detection
FIXED_LEFT_LANE_X  = 255
FIXED_RIGHT_LANE_X = 485

# Camera — must match common/config.yaml camera section
CAM_WIDTH  = 768
CAM_HEIGHT = 384

DATA_DIR   = script_dir / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_CSV = DATA_DIR / "labels.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Mode decision helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_y_at_distance(depth_array: np.ndarray, depth_scale: float,
                        target_m: float = 0.7) -> int:
    """
    Scan the centre column of depth_array and return the row whose depth
    value is closest to target_m.  Falls back to 2/3 of image height if
    depth data is unavailable.
    """
    fallback = int(depth_array.shape[0] * 2 / 3)
    if depth_scale <= 0:
        return fallback
    cx = depth_array.shape[1] // 2
    col = depth_array[:, cx].astype(np.float32) * depth_scale
    valid = col > 0
    if not valid.any():
        return fallback
    diffs = np.abs(col - target_m)
    diffs[~valid] = np.inf
    return int(np.argmin(diffs))


def _interpolate_lane_x(lane, y: float) -> float:
    """
    Linear interpolation of x at a given y along a lane line.
    Accepts LaneMessage (has .x1 .y1 .x2 .y2) or a plain (x1,y1,x2,y2) tuple.
    Extrapolates outside the segment when needed.
    """
    if hasattr(lane, 'x1'):
        x1, y1, x2, y2 = lane.x1, lane.y1, lane.x2, lane.y2
    else:
        x1, y1, x2, y2 = lane
    if y2 == y1:
        return float(x1)
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def _lane_coverage(box, left_x: float, right_x: float) -> float:
    """
    Fraction of lane width [0, 1] that a bounding box overlaps horizontally.
    """
    lane_width = right_x - left_x
    if lane_width <= 0:
        return 0.0
    x1, _y1, x2, _y2 = box
    overlap = max(0.0, min(x2, right_x) - max(x1, left_x))
    return overlap / lane_width


def _classify_zone(box, left_x: float, right_x: float) -> str:
    """Return 'left_third', 'center_third', or 'right_third' based on box centre."""
    third = (right_x - left_x) / 3.0
    cx = (box[0] + box[2]) / 2.0
    if cx < left_x + third:
        return 'left_third'
    if cx > right_x - third:
        return 'right_third'
    return 'center_third'


def _decide_mode(
    boxes: list,
    distances: list,
    left_x: float,
    right_x: float,
    current_mode: str,
) -> str:
    """
    Stateless mode decision with hysteresis via current_mode.

    Rules (applied to the closest obstacle that overlaps our lane):
      - No overlapping obstacle within EXIT distance → NORMAL
      - coverage < MICRO_THRESHOLD and box in left/right third → MICRO_ADJUST
      - coverage >= MICRO_THRESHOLD:
          - adjacent lane (x < left_x) also has obstacle within ADJACENT_MAX_DIST → STOP
          - otherwise → LANE_CHANGE
      - Hysteresis: only trigger new mode when distance < ENTER;
                    clear mode only when distance > EXIT
    """
    # Find the closest obstacle that overlaps our lane
    closest_box  = None
    closest_dist = float('inf')
    for box, dist in zip(boxes, distances):
        if dist <= 0:
            continue
        if _lane_coverage(box, left_x, right_x) > 0 and dist < closest_dist:
            closest_dist = dist
            closest_box  = box

    # No in-lane obstacle or obstacle cleared the hysteresis threshold
    if closest_box is None or closest_dist > DECISION_EXIT:
        return 'NORMAL'

    # Hysteresis: don't enter a new mode until obstacle is within ENTER distance
    if closest_dist > DECISION_ENTER and current_mode == 'NORMAL':
        return 'NORMAL'

    coverage = _lane_coverage(closest_box, left_x, right_x)
    zone     = _classify_zone(closest_box, left_x, right_x)

    if coverage < MICRO_THRESHOLD and zone in ('left_third', 'right_third'):
        return 'MICRO_ADJUST'

    if coverage >= MICRO_THRESHOLD:
        # Check whether the adjacent (left) lane is clear for a lane change
        for box, dist in zip(boxes, distances):
            if dist <= 0 or dist > ADJACENT_MAX_DIST:
                continue
            adj_cx = (box[0] + box[2]) / 2.0
            if adj_cx < left_x:  # obstacle in left (adjacent) lane
                return 'STOP'
        return 'LANE_CHANGE'

    return 'NORMAL'


# ─────────────────────────────────────────────────────────────────────────────
# Terminal keyboard reader  (works over SSH / headless, no X11 required)
# ─────────────────────────────────────────────────────────────────────────────
_ESC_MAP = {
    '\x1b[D': 'left',
    '\x1b[C': 'right',
    '\x1b[B': 'down',
    '\x1b[A': 'up',
}
_KEY_HOLD_TIMEOUT = 0.20  # seconds


class TerminalKeyboard:
    """
    Non-blocking keyboard reader using terminal raw mode.

    Runs a background thread that reads escape sequences from stdin and
    records the last-seen timestamp for each key.  The main thread calls
    `snapshot()` to get the current held-key state.

    Arrow keys generate auto-repeat escape sequences while held (at the
    terminal's key-repeat rate, typically ≥20 Hz).  A key is considered
    "held" if its last sequence arrived within _KEY_HOLD_TIMEOUT seconds.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._last_seen: dict[str, float] = {}
        self.quit = False
        self._space_pending = False   # consumed on first snapshot() read
        self._running = False
        self._thread: threading.Thread | None = None
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True, name="kbd-reader")
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=1.0)

    def _reader(self):
        try:
            tty.setraw(self._fd)
            while self._running:
                if not select.select([sys.stdin], [], [], 0.05)[0]:
                    continue
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    seq = ch
                    for _ in range(2):
                        if select.select([sys.stdin], [], [], 0.02)[0]:
                            seq += sys.stdin.read(1)
                        else:
                            break
                    key_name = _ESC_MAP.get(seq)
                    if key_name:
                        with self._lock:
                            self._last_seen[key_name] = time.monotonic()
                elif ch == ' ':
                    with self._lock:
                        self._space_pending = True
                elif ch in ('q', 'Q'):
                    with self._lock:
                        self.quit = True
                    break
        finally:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def snapshot(self) -> tuple[bool, bool, bool, bool, bool]:
        """Returns (left_held, right_held, down_held, space_pressed, quit_requested).
        space_pressed is True only once per press — consumed on read."""
        now = time.monotonic()
        with self._lock:
            def held(name: str) -> bool:
                return (now - self._last_seen.get(name, 0.0)) < _KEY_HOLD_TIMEOUT
            space = self._space_pending
            self._space_pending = False   # consume
            return held('left'), held('right'), held('down'), space, self.quit


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────
_CSV_HEADER = ["frame_id", "input_steering", "input_throttle", "mode"]


def _init_dataset(images_dir: Path, labels_csv: Path) -> tuple[int, object, object]:
    """
    Create output directories and open/append the CSV.

    Detects an existing 3-column CSV (old format without 'mode') and warns
    the user; new rows will still be written with 4 columns.

    Returns:
        (start_frame_id, csv_file_handle, csv_writer)
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(images_dir.glob("*.jpg"))
    if existing:
        start_id = int(existing[-1].stem) + 1
        print(f"[dataset] Resuming from frame {start_id:06d}  ({len(existing)} frames already saved)")
    else:
        start_id = 1
        print("[dataset] Starting fresh dataset")

    csv_exists = labels_csv.exists()
    if csv_exists:
        # Check whether the existing CSV already has the mode column
        with open(labels_csv, newline="") as f:
            header = next(csv.reader(f), [])
        if 'mode' not in header:
            print("[dataset] WARN: existing labels.csv has no 'mode' column (old format).")
            print("[dataset]       New rows will include mode; old rows will have N/A in that column.")

    csv_fh = open(labels_csv, "a", newline="")
    writer = csv.writer(csv_fh)
    if not csv_exists:
        writer.writerow(_CSV_HEADER)

    return start_id, csv_fh, writer


def _save_frame(
    frame_id: int,
    color_bgr: np.ndarray,
    depth: np.ndarray | None,
    steering: float,
    throttle: float,
    mode: str,
    images_dir: Path,
    csv_writer,
    csv_fh,
):
    stem = f"{frame_id:06d}"
    cv2.imwrite(str(images_dir / f"{stem}.jpg"), color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if depth is not None:
        np.save(str(images_dir / f"{stem}.npy"), depth)
    else:
        np.save(str(images_dir / f"{stem}.npy"), np.array([], dtype=np.uint16))
    csv_writer.writerow([stem, f"{steering:.4f}", f"{throttle:.4f}", mode])
    csv_fh.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helper  (broadcast only — never applied to saved images)
# ─────────────────────────────────────────────────────────────────────────────
_BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 165, 0),
    (128, 0, 128), (0, 255, 255), (255, 255, 0), (0, 128, 255),
    (128, 128, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255), (0, 128, 0),
]

_MODE_COLORS = {
    'NORMAL':       (0, 255, 0),
    'MICRO_ADJUST': (0, 255, 255),
    'LANE_CHANGE':  (0, 165, 255),
    'STOP':         (0, 0, 255),
}


def _draw_annotations(
    frame: np.ndarray,
    boxes: list,
    distances: list,
    class_ids: list,
    mode: str,
    fps: float,
) -> np.ndarray:
    """
    Return a copy of frame with YOLO boxes, distance labels, mode, and FPS
    drawn on it.  The original frame is never modified.
    """
    out = frame.copy()
    for box, dist, cid in zip(boxes, distances, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = _BOX_COLORS[cid % len(_BOX_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        dist_text = f"{dist:.2f}m" if dist > 0 else "N/A"
        cv2.putText(out, f"{label} {dist_text}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    mode_color = _MODE_COLORS.get(mode, (255, 255, 255))
    cv2.putText(out, f"MODE: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main collector
# ─────────────────────────────────────────────────────────────────────────────

def main(web_port: int = 8082):
    print("=" * 60)
    print("  Data Collection Script — Obstacle Avoidance Training")
    print("=" * 60)
    print(f"  Output        : {DATA_DIR}")
    print(f"  Save rate cap : {SAVE_FPS} fps")
    print(f"  YOLO skip     : every {YOLO_SKIP} frames")
    print(f"  Base throttle : {BASE_THROTTLE}")
    print(f"  Obstacle throttle : {OBSTACLE_THROTTLE}")
    print(f"  Web viewer    : {'port ' + str(web_port) if web_port > 0 else 'DISABLED'}")
    print()
    print("  Controls: ← steer left  → steer right  ↓ stop  q quit")
    print("=" * 60)

    # ── YOLO model ────────────────────────────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] YOLO model not found: {MODEL_PATH}")
        sys.exit(1)
    print(f"\n[YOLO] Loading model: {MODEL_PATH}")
    yolo = _YOLO(MODEL_PATH)
    if torch.cuda.is_available():
        free_mb, total_mb = [x / 1024**2 for x in torch.cuda.mem_get_info(0)]
        print(f"[YOLO] GPU device {YOLO_DEVICE}  ({free_mb:.0f} MB free / {total_mb:.0f} MB total)")
    else:
        print("[YOLO] CUDA not available — using CPU")

    # ── Web viewer ────────────────────────────────────────────────────────────
    web_viewer = None
    if web_port > 0:
        web_viewer = YOLOWebViewer(http_port=web_port, avoidance=None, detector=None)
        web_viewer.start()
        print(f"[WEB] Viewer started on http://0.0.0.0:{web_port}")

    # ── LKAS client (lane detection) ──────────────────────────────────────────
    lkas_client = None
    try:
        print("[LKAS] Connecting lane detection client...")
        lkas_client = _LKASClient(
            image_shm_name="image",
            detection_shm_name="detection",
            control_shm_name="control",
        )
        print("[LKAS] Lane detection client connected")
    except Exception as e:
        print(f"[WARN] LKAS client unavailable ({e}) — using fixed lane fallback")

    # ── RealSense camera ──────────────────────────────────────────────────────
    print("\n[CAM] Opening RealSense camera...")
    camera = Camera(width=CAM_WIDTH, height=CAM_HEIGHT, enable_depth=True)
    frame_w, frame_h = CAM_WIDTH, CAM_HEIGHT

    # ── Image shared memory (creator) ─────────────────────────────────────────
    # collect_data.py owns the camera and writes frames to image SHM so that
    # the LKAS detection server (lkas --broadcast) can read them.
    # vehicle.py must be stopped — it would conflict for both camera and image SHM.
    print("\n[SHM] Creating image shared memory (name='image')...")
    image_channel = SharedMemoryImageChannel(name="image", create=True)

    # ── JetRacer ──────────────────────────────────────────────────────────────
    car = None
    if JETRACER_AVAILABLE:
        print("\n[CAR] Initializing NvidiaRacecar...")
        car = NvidiaRacecar()
        car.steering_offset = 0.05
        car.steering = 0.0
        car.throttle = 0.0
        print("[CAR] NvidiaRacecar ready")
    else:
        print("\n[CAR] Simulation mode — steering/throttle logged but not applied")

    # ── Dataset ───────────────────────────────────────────────────────────────
    frame_id, csv_fh, csv_writer = _init_dataset(IMAGES_DIR, LABELS_CSV)

    # ── Keyboard ──────────────────────────────────────────────────────────────
    kbd = TerminalKeyboard()
    kbd.start()

    # ── Signal handling ───────────────────────────────────────────────────────
    running = True

    def _shutdown(signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Main loop state ───────────────────────────────────────────────────────
    print(f"\n[COLLECT] Running — press q or Ctrl+C to stop\n")

    current_steering  = 0.0
    current_mode      = 'NORMAL'   # locked mode written to CSV
    pending_mode      = 'NORMAL'   # candidate not yet confirmed
    pending_count     = 0          # consecutive frames pending_mode has held
    last_save_time    = 0.0
    save_interval     = 1.0 / SAVE_FPS
    saved_count       = 0
    yolo_frame_count  = 0
    cam_frame_id      = 0
    fps               = 0.0
    fps_frame_count   = 0
    fps_start_time    = time.time()

    # Cached YOLO results (reused on skipped frames)
    cached_boxes:     list = []
    cached_distances: list = []
    cached_class_ids: list = []

    depth_scale = camera.depth_scale if camera.depth_scale > 0 else 0.001

    try:
        while running:

            # ── Quit check ────────────────────────────────────────────────────
            left_held, right_held, down_held, space_pressed, quit_req = kbd.snapshot()
            if quit_req:
                print("\n[COLLECT] Quit key pressed")
                break

            # ── Capture frame from RealSense camera ───────────────────────────
            color_bgr, depth_raw = camera.read_frames()
            if color_bgr is None:
                continue
            cam_frame_id += 1
            depth_array = depth_raw if depth_raw is not None else \
                          np.zeros((frame_h, frame_w), dtype=np.uint16)

            # ── Write frame to image SHM (LKAS detection server reads from here) ──
            image_channel.write(color_bgr, time.time(), cam_frame_id,
                                depth_array, depth_scale)

            # ── YOLO inference (every YOLO_SKIP frames) ───────────────────────
            yolo_frame_count += 1
            if yolo_frame_count % YOLO_SKIP == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    results = yolo(
                        color_bgr,
                        conf=CONFIDENCE_THRESHOLD,
                        iou=IOU_THRESHOLD,
                        device=YOLO_DEVICE,
                        verbose=False,
                    )
                    boxes_raw = []
                    distances_raw = []
                    class_ids_raw = []
                    if len(results[0].boxes) > 0:
                        xyxy     = results[0].boxes.xyxy.cpu().numpy()
                        cls_ids  = results[0].boxes.cls.cpu().numpy().astype(int)
                        for box, cid in zip(xyxy, cls_ids):
                            x1, y1, x2, y2 = map(int, box)
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            raw = int(depth_array[
                                max(0, min(cy, frame_h - 1)),
                                max(0, min(cx, frame_w - 1))
                            ])
                            dist = raw * depth_scale if raw > 0 else -1.0
                            boxes_raw.append(box)
                            distances_raw.append(dist)
                            class_ids_raw.append(int(cid))
                    cached_boxes      = boxes_raw
                    cached_distances  = distances_raw
                    cached_class_ids  = class_ids_raw
                except Exception as e:
                    print(f"\n[YOLO] Inference error: {e}")

            boxes     = cached_boxes
            distances = cached_distances
            class_ids = cached_class_ids

            # ── Lane detection from LKAS ──────────────────────────────────────
            left_lane_x  = float(FIXED_LEFT_LANE_X)
            right_lane_x = float(FIXED_RIGHT_LANE_X)
            if lkas_client is not None:
                try:
                    det = lkas_client.get_detection(timeout=0.01)
                    if det is not None and det.left_lane and det.right_lane:
                        target_y = _find_y_at_distance(depth_array, depth_scale)
                        lx = _interpolate_lane_x(det.left_lane, target_y)
                        rx = _interpolate_lane_x(det.right_lane, target_y)
                        if 0 < lx < rx <= frame_w:
                            left_lane_x  = lx
                            right_lane_x = rx
                except Exception:
                    pass  # keep fixed fallback

            # ── Mode state machine ────────────────────────────────────────────
            # Spacebar manually releases a locked mode back to NORMAL.
            if space_pressed:
                current_mode  = 'NORMAL'
                pending_mode  = 'NORMAL'
                pending_count = 0
            elif current_mode == 'NORMAL':
                # Not locked — debounce candidates before committing.
                candidate = _decide_mode(
                    boxes, distances, left_lane_x, right_lane_x, current_mode
                )
                if candidate == pending_mode:
                    pending_count += 1
                else:
                    pending_mode  = candidate
                    pending_count = 1
                if pending_count >= MODE_CONFIRM_FRAMES and pending_mode != 'NORMAL':
                    current_mode  = pending_mode
                    pending_count = 0
            # else: mode is locked — ignore YOLO detections until space pressed

            # Throttle reduction only when an obstacle is actively within range,
            # not just because a mode is locked (obstacle may already be passed).
            nearest = min((d for d in distances if d > 0), default=-1.0)
            obstacle_active = nearest > 0 and nearest < DECISION_EXIT

            # ── Steering source: web viewer OR keyboard ───────────────────────
            # Throttle is always automatic (fixed forward + obstacle reduction).
            # Only steering is manually controlled.
            if web_viewer is not None and web_viewer.manual_mode:
                # Web viewer controls steering only; throttle stays automatic.
                input_steering = web_viewer.manual_steering
                current_steering = input_steering  # keep in sync for mode switch back
            else:
                # Keyboard: steering accumulates while key held, resets on release
                if left_held and not right_held:
                    current_steering = max(-1.0, current_steering - STEER_STEP)
                elif right_held and not left_held:
                    current_steering = min(1.0, current_steering + STEER_STEP)
                else:
                    current_steering = 0.0
                input_steering = current_steering

            # ── Throttle: always automatic ────────────────────────────────────
            if down_held:
                input_throttle = 0.0
            elif obstacle_active:
                input_throttle = OBSTACLE_THROTTLE
            else:
                input_throttle = BASE_THROTTLE

            # ── Apply to vehicle ──────────────────────────────────────────────
            if car is not None:
                car.steering = -float(input_steering)  # negated: inverted steering hardware
                car.throttle = -float(input_throttle)  # negated: negative = forward on this JetRacer

            # ── FPS tracking ──────────────────────────────────────────────────
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time  = time.time()

            # ── Broadcast annotated frame to web viewer ───────────────────────
            if web_viewer is not None:
                annotated = _draw_annotations(color_bgr, boxes, distances,
                                              class_ids, current_mode, fps)
                web_viewer.broadcast_frame(annotated)
                web_viewer.broadcast_status({
                    'fps':              fps,
                    'action':           current_mode,
                    'steering':         input_steering,
                    'throttle':         input_throttle,
                    'nearest_distance': nearest,
                    'overtaking_state': current_mode,
                    'lane_detected':    lkas_client is not None,
                    'lkas_steering':    0.0,
                    'left_lane_x':      left_lane_x,
                    'right_lane_x':     right_lane_x,
                })

            # ── Save frame (rate-limited) ─────────────────────────────────────
            now = time.monotonic()
            if now - last_save_time >= save_interval:
                last_save_time = now
                _save_frame(
                    frame_id   = frame_id,
                    color_bgr  = color_bgr,
                    depth      = depth_raw,
                    steering   = input_steering,
                    throttle   = input_throttle,
                    mode       = current_mode,
                    images_dir = IMAGES_DIR,
                    csv_writer = csv_writer,
                    csv_fh     = csv_fh,
                )
                frame_id  += 1
                saved_count += 1

                steer_tag = (
                    " ← LEFT" if input_steering < 0 else
                    " → RIGHT" if input_steering > 0 else
                    " STRAIGHT"
                )
                stop_tag = " ↓STOP" if down_held else ""
                sys.stdout.write(
                    f"\r[{saved_count:>6d}]  "
                    f"steer={input_steering:+.2f}{steer_tag}  "
                    f"throttle={input_throttle:.2f}  "
                    f"mode={current_mode:<12s}"
                    f"{stop_tag}   "
                )
                sys.stdout.flush()

    finally:
        running = False

        if car is not None:
            car.throttle = 0.0
            car.steering = 0.0

        kbd.stop()
        csv_fh.close()

        sys.stdout.write("\n")
        sys.stdout.flush()

        if web_viewer is not None:
            try:
                web_viewer.stop()
            except Exception:
                pass
        if lkas_client is not None:
            try:
                lkas_client.close()
            except Exception:
                pass
        try:
            image_channel.close()
            image_channel.unlink()  # creator must unlink on exit
        except Exception:
            pass

        print(f"\n[COLLECT] Done — {saved_count} frames saved to {DATA_DIR}")
        print(f"[COLLECT] labels.csv → {LABELS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data collection for obstacle avoidance training")
    parser.add_argument('--web-port', type=int, default=8082,
                        help='Web viewer HTTP port (default: 8082, 0 to disable)')
    args = parser.parse_args()
    main(web_port=args.web_port)
