#!/usr/bin/env python3
"""
ML-based Obstacle Avoidance  (replaces rule-based steering in yolo_depth_avoidance.py)
========================================================================================
Pipeline per frame:
  1. Read color + depth from unified image SHM
  2. YOLO detection  →  boxes, distances
  3. Rule-based mode decision  →  NORMAL / MICRO_ADJUST / LANE_CHANGE / STOP
     (same logic used during data collection in collect_data.py)
  4. NORMAL  →  LKAS handles steering (no change)
     MICRO_ADJUST / LANE_CHANGE  →  ML model (color + depth + mode) → steering
     STOP  →  throttle = 0, steering = 0
  5. Write ObstacleMessage to control SHM  (read by DecisionServer)
  6. Broadcast annotated frame to web viewer

Run alongside:
  lkas --broadcast   (creates image + detection SHM)
  vehicle.py         (reads control SHM and drives)

Usage:
  python yolo_depth_avoidance_ml.py [--web-port 8082] [--motor]
"""

import sys
import time
import argparse
import importlib.util
import numpy as np
import cv2
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent.parent / "vehicle" / "src"))

# ── PyTorch legacy weights fix (before ultralytics) ───────────────────────────
_orig_torch_load = torch.load

def _torch_load_legacy(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_legacy

from ultralytics import YOLO as _YOLO

# ── JetRacer ──────────────────────────────────────────────────────────────────
try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    JETRACER_AVAILABLE = True
except ImportError:
    print("[WARN] JetRacer not available — simulation mode")
    JETRACER_AVAILABLE = False

# ── LKAS ──────────────────────────────────────────────────────────────────────
try:
    from lkas.integration.shared_memory import (
        SharedMemoryImageChannel,
        SharedMemoryControlChannel,
    )
    from lkas.integration.shared_memory.messages import ObstacleMessage, ObstacleAction
    from lkas import LKASClient
    LKAS_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] LKAS not available: {e}")
    sys.exit(1)

# ── Web viewer ────────────────────────────────────────────────────────────────
from yolo_web_viewer import YOLOWebViewer

# ── YOLO config ───────────────────────────────────────────────────────────────
_cfg_path = script_dir.parent / "config.py"
_spec = importlib.util.spec_from_file_location("yolo_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_spec)
_cfg.__file__ = str(_cfg_path)
_spec.loader.exec_module(_cfg)

YOLO_MODEL_PATH      = _cfg.MODEL_PATH
CONFIDENCE_THRESHOLD = _cfg.CONFIDENCE_THRESHOLD
IOU_THRESHOLD        = _cfg.IOU_THRESHOLD
CLASS_NAMES          = _cfg.CLASS_NAMES

# ── ML model path ─────────────────────────────────────────────────────────────
ML_MODEL_PATH = script_dir / "obstacle_avoidance_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Mode decision  (mirrored from collect_data.py)
# ─────────────────────────────────────────────────────────────────────────────
DECISION_ENTER    = 0.7    # metres — trigger mode
DECISION_EXIT     = 1.0    # metres — hysteresis exit
MICRO_THRESHOLD   = 0.33   # lane coverage below which → MICRO_ADJUST
ADJACENT_MAX_DIST = 2.0    # metres — scan for adjacent-lane obstacles

FIXED_LEFT_LANE_X  = 255
FIXED_RIGHT_LANE_X = 485

BASE_THROTTLE     = 0.20
OBSTACLE_THROTTLE = 0.20
YOLO_SKIP         = 2       # run YOLO every N frames


def _lane_coverage(box, left_x: float, right_x: float) -> float:
    lane_width = right_x - left_x
    if lane_width <= 0:
        return 0.0
    x1, _y1, x2, _y2 = box
    overlap = max(0.0, min(x2, right_x) - max(x1, left_x))
    return overlap / lane_width


def _classify_zone(box, left_x: float, right_x: float) -> str:
    third = (right_x - left_x) / 3.0
    cx = (box[0] + box[2]) / 2.0
    if cx < left_x + third:
        return 'left_third'
    if cx > right_x - third:
        return 'right_third'
    return 'center_third'


def _decide_mode(boxes, distances, left_x, right_x, current_mode) -> str:
    closest_box  = None
    closest_dist = float('inf')
    for box, dist in zip(boxes, distances):
        if dist <= 0:
            continue
        if _lane_coverage(box, left_x, right_x) > 0 and dist < closest_dist:
            closest_dist = dist
            closest_box  = box

    if closest_box is None or closest_dist > DECISION_EXIT:
        return 'NORMAL'
    if closest_dist > DECISION_ENTER and current_mode == 'NORMAL':
        return 'NORMAL'

    coverage = _lane_coverage(closest_box, left_x, right_x)
    zone     = _classify_zone(closest_box, left_x, right_x)

    if coverage < MICRO_THRESHOLD and zone in ('left_third', 'right_third'):
        return 'MICRO_ADJUST'

    if coverage >= MICRO_THRESHOLD:
        for box, dist in zip(boxes, distances):
            if dist <= 0 or dist > ADJACENT_MAX_DIST:
                continue
            adj_cx = (box[0] + box[2]) / 2.0
            if adj_cx < left_x:
                return 'STOP'
        return 'LANE_CHANGE'

    return 'NORMAL'


def _interpolate_lane_x(lane, y: float) -> float:
    if hasattr(lane, 'x1'):
        x1, y1, x2, y2 = lane.x1, lane.y1, lane.x2, lane.y2
    else:
        x1, y1, x2, y2 = lane
    if y2 == y1:
        return float(x1)
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def _find_y_at_distance(depth_array, depth_scale, target_m=0.7) -> int:
    fallback = int(depth_array.shape[0] * 2 / 3)
    if depth_scale <= 0:
        return fallback
    cx  = depth_array.shape[1] // 2
    col = depth_array[:, cx].astype(np.float32) * depth_scale
    valid = col > 0
    if not valid.any():
        return fallback
    diffs = np.abs(col - target_m)
    diffs[~valid] = np.inf
    return int(np.argmin(diffs))


# ─────────────────────────────────────────────────────────────────────────────
# ML model  (must match train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
MODE_MAP = {'NORMAL': 0, 'MICRO_ADJUST': 1, 'LANE_CHANGE': 2, 'STOP': 3}
DEPTH_NORM = 10_000.0

_ACTION_MAP = {
    'NORMAL':       ObstacleAction.NORMAL,
    'MICRO_ADJUST': ObstacleAction.AVOID_LEFT,   # direction overridden by steering sign
    'LANE_CHANGE':  ObstacleAction.AVOID_LEFT,
    'STOP':         ObstacleAction.STOP,
}


class DepthCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class ObstacleAvoidanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.color_features = backbone.features
        self.color_pool     = nn.AdaptiveAvgPool2d(1)
        self.depth_cnn      = DepthCNN()
        self.mode_embed     = nn.Embedding(4, 8)
        fused_dim = 576 + 64 + 8  # 648
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, color, depth, mode):
        c = self.color_pool(self.color_features(color)).flatten(1)
        d = self.depth_cnn(depth)
        m = self.mode_embed(mode)
        return self.mlp(torch.cat([c, d, m], dim=1)).squeeze(1)


def _load_model(path: Path, device: torch.device) -> ObstacleAvoidanceModel:
    model = ObstacleAvoidanceModel().to(device)
    state = torch.load(str(path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[ML] Model loaded from {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing  (identical to ObstacleDataset in train.py)
# ─────────────────────────────────────────────────────────────────────────────
_color_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def _preprocess(color_bgr: np.ndarray, depth_raw: np.ndarray,
                mode_str: str, device: torch.device):
    """Return (color_t, depth_t, mode_t) tensors ready for the model."""
    # Color BGR → RGB → tensor
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_t   = _color_transform(color_rgb).unsqueeze(0).to(device)  # (1,3,224,224)

    # Depth
    if depth_raw is None or depth_raw.size == 0:
        depth_f32 = np.zeros((224, 224), dtype=np.float32)
    else:
        depth_f32 = depth_raw.astype(np.float32) / DEPTH_NORM
        depth_f32 = np.clip(depth_f32, 0.0, 1.0)
        depth_f32 = cv2.resize(depth_f32, (224, 224), interpolation=cv2.INTER_NEAREST)
    depth_t = torch.tensor(depth_f32, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,224,224)

    # Mode
    mode_idx = MODE_MAP.get(mode_str, 0)
    mode_t   = torch.tensor([mode_idx], dtype=torch.long).to(device)  # (1,)

    return color_t, depth_t, mode_t


# ─────────────────────────────────────────────────────────────────────────────
# Annotation
# ─────────────────────────────────────────────────────────────────────────────
_MODE_COLORS = {
    'NORMAL':       (0, 255, 0),
    'MICRO_ADJUST': (0, 255, 255),
    'LANE_CHANGE':  (0, 165, 255),
    'STOP':         (0, 0, 255),
}
_BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 165, 0),
    (128, 0, 128), (0, 255, 255), (255, 255, 0), (0, 128, 255),
    (128, 128, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255), (0, 128, 0),
]


def _draw(frame, boxes, distances, class_ids, mode, steering, fps):
    out = frame.copy()
    for box, dist, cid in zip(boxes, distances, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = _BOX_COLORS[cid % len(_BOX_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        dist_txt = f"{dist:.2f}m" if dist > 0 else "N/A"
        cv2.putText(out, f"{label} {dist_txt}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    mc = _MODE_COLORS.get(mode, (255, 255, 255))
    cv2.putText(out, f"MODE: {mode}",      (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mc, 2, cv2.LINE_AA)
    cv2.putText(out, f"Steer: {steering:+.3f}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(out, f"FPS: {fps:.1f}",    (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main(web_port: int = 8082, enable_motor: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[ML]  Device : {device}")
    if device.type == 'cuda':
        free, total = [x / 1024**2 for x in torch.cuda.mem_get_info(0)]
        print(f"[ML]  VRAM   : {free:.0f} MB free / {total:.0f} MB total")

    # ── ML model ──────────────────────────────────────────────────────────────
    if not ML_MODEL_PATH.exists():
        print(f"[ERROR] ML model not found: {ML_MODEL_PATH}")
        print("        Run train.py first to generate obstacle_avoidance_model.pth")
        sys.exit(1)
    model = _load_model(ML_MODEL_PATH, device)

    # ── YOLO model ────────────────────────────────────────────────────────────
    if not Path(YOLO_MODEL_PATH).exists():
        print(f"[ERROR] YOLO model not found: {YOLO_MODEL_PATH}")
        sys.exit(1)
    print(f"[YOLO] Loading: {YOLO_MODEL_PATH}")
    yolo = _YOLO(YOLO_MODEL_PATH)

    # ── Shared memory channels ────────────────────────────────────────────────
    print("[SHM] Connecting to image SHM (create=False)...")
    image_channel   = SharedMemoryImageChannel(name="image",   create=False,
                                               retry_count=60, retry_delay=1.0)
    control_channel = SharedMemoryControlChannel(name="control", create=False,
                                                 retry_count=30, retry_delay=1.0)

    # ── LKAS client (lane detection) ──────────────────────────────────────────
    lkas_client = None
    try:
        lkas_client = LKASClient(image_shm_name="image",
                                 detection_shm_name="detection",
                                 control_shm_name="control")
        print("[LKAS] Lane detection client connected")
    except Exception as e:
        print(f"[WARN] LKAS client unavailable ({e}) — fixed lane fallback active")

    # ── JetRacer ──────────────────────────────────────────────────────────────
    car = None
    if enable_motor and JETRACER_AVAILABLE:
        car = NvidiaRacecar()
        car.steering_offset = 0.05
        car.steering = 0.0
        car.throttle = 0.0
        print("[CAR] NvidiaRacecar ready — MOTORS ACTIVE")
    else:
        print("[CAR] Simulation mode (motor control disabled)")

    # ── Web viewer ────────────────────────────────────────────────────────────
    web_viewer = None
    if web_port > 0:
        web_viewer = YOLOWebViewer(http_port=web_port, avoidance=None, detector=None)
        web_viewer.start()
        print(f"[WEB] Viewer at http://0.0.0.0:{web_port}")

    # ── State ─────────────────────────────────────────────────────────────────
    current_mode     = 'NORMAL'
    pending_mode     = 'NORMAL'
    pending_count    = 0
    MODE_CONFIRM_FRAMES = 3

    yolo_frame_count = 0
    cached_boxes:     list = []
    cached_distances: list = []
    cached_class_ids: list = []

    fps = 0.0
    fps_count = 0
    fps_start = time.time()

    # Read first frame for frame dimensions and depth_scale
    first_msg = image_channel.read_blocking(timeout=10.0)
    if first_msg is None:
        print("[ERROR] No frame received from image SHM within 10 s")
        sys.exit(1)
    frame_h, frame_w = first_msg.image.shape[:2]
    depth_scale = first_msg.depth_scale if first_msg.depth_scale > 0 else 0.001
    print(f"[SHM] Frame size: {frame_w}x{frame_h}  depth_scale={depth_scale}")

    print("\n[RUN] Starting — Ctrl+C to stop\n")

    try:
        while True:
            msg = image_channel.read()
            if msg is None:
                time.sleep(0.001)
                continue

            color_bgr   = msg.image
            depth_array = msg.depth_image if msg.depth_image is not None else \
                          np.zeros((frame_h, frame_w), dtype=np.uint16)
            if msg.depth_scale > 0:
                depth_scale = msg.depth_scale

            # ── YOLO (every YOLO_SKIP frames) ─────────────────────────────────
            yolo_frame_count += 1
            if yolo_frame_count % YOLO_SKIP == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                try:
                    results = yolo(color_bgr, conf=CONFIDENCE_THRESHOLD,
                                   iou=IOU_THRESHOLD,
                                   device=0 if device.type == 'cuda' else 'cpu',
                                   verbose=False)
                    boxes_raw, distances_raw, class_ids_raw = [], [], []
                    if len(results[0].boxes) > 0:
                        xyxy    = results[0].boxes.xyxy.cpu().numpy()
                        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                        for box, cid in zip(xyxy, cls_ids):
                            cx = int((box[0] + box[2]) / 2)
                            cy = int((box[1] + box[3]) / 2)
                            cx = max(0, min(cx, frame_w - 1))
                            cy = max(0, min(cy, frame_h - 1))
                            raw  = int(depth_array[cy, cx])
                            dist = raw * depth_scale if raw > 0 else -1.0
                            boxes_raw.append(box)
                            distances_raw.append(dist)
                            class_ids_raw.append(int(cid))
                    cached_boxes      = boxes_raw
                    cached_distances  = distances_raw
                    cached_class_ids  = class_ids_raw
                except Exception as e:
                    print(f"[YOLO] Error: {e}")

            boxes     = cached_boxes
            distances = cached_distances
            class_ids = cached_class_ids

            # ── Lane boundaries from LKAS ──────────────────────────────────────
            left_x  = float(FIXED_LEFT_LANE_X)
            right_x = float(FIXED_RIGHT_LANE_X)
            lkas_steering = 0.0
            lane_detected = False
            if lkas_client is not None:
                try:
                    det = lkas_client.get_detection(timeout=0.01)
                    if det is not None and det.left_lane and det.right_lane:
                        y_ref = _find_y_at_distance(depth_array, depth_scale)
                        lx = _interpolate_lane_x(det.left_lane, y_ref)
                        rx = _interpolate_lane_x(det.right_lane, y_ref)
                        if 0 < lx < rx <= frame_w:
                            left_x  = lx
                            right_x = rx
                            lane_detected = True
                    ctrl = lkas_client.get_control(timeout=0.01)
                    if ctrl is not None:
                        lkas_steering = ctrl.steering
                except Exception:
                    pass

            # ── Mode state machine (same debounce logic as collect_data.py) ────
            if current_mode == 'STOP':
                if _decide_mode(boxes, distances, left_x, right_x, current_mode) == 'NORMAL':
                    current_mode  = 'NORMAL'
                    pending_mode  = 'NORMAL'
                    pending_count = 0
            elif current_mode == 'NORMAL':
                candidate = _decide_mode(boxes, distances, left_x, right_x, current_mode)
                if candidate == pending_mode:
                    pending_count += 1
                else:
                    pending_mode  = candidate
                    pending_count = 1
                if pending_count >= MODE_CONFIRM_FRAMES and pending_mode != 'NORMAL':
                    current_mode  = pending_mode
                    pending_count = 0
            # MICRO_ADJUST / LANE_CHANGE: stay locked until obstacle clears
            else:
                if _decide_mode(boxes, distances, left_x, right_x, current_mode) == 'NORMAL':
                    current_mode  = 'NORMAL'
                    pending_mode  = 'NORMAL'
                    pending_count = 0

            # ── Steering decision ──────────────────────────────────────────────
            if current_mode == 'NORMAL':
                final_steering = lkas_steering
                final_throttle = BASE_THROTTLE
                obstacle_action = ObstacleAction.NORMAL

            elif current_mode == 'STOP':
                final_steering  = 0.0
                final_throttle  = 0.0
                obstacle_action = ObstacleAction.STOP

            else:  # MICRO_ADJUST or LANE_CHANGE → ML model
                with torch.no_grad():
                    c_t, d_t, m_t = _preprocess(color_bgr, depth_array,
                                                 current_mode, device)
                    ml_steering = model(c_t, d_t, m_t).item()

                # Clamp to [-1, 1]
                ml_steering = float(np.clip(ml_steering, -1.0, 1.0))

                final_steering  = ml_steering
                final_throttle  = OBSTACLE_THROTTLE
                obstacle_action = (ObstacleAction.AVOID_LEFT  if ml_steering < 0
                                   else ObstacleAction.AVOID_RIGHT)

            # ── Apply to vehicle ───────────────────────────────────────────────
            if car is not None:
                car.steering = -float(final_steering)   # hardware inversion
                car.throttle = -float(final_throttle)   # negative = forward

            # ── Write to control SHM ───────────────────────────────────────────
            nearest = min((d for d in distances if d > 0), default=-1.0)
            obstacle_msg = ObstacleMessage(
                active   = (current_mode != 'NORMAL'),
                action   = obstacle_action,
                distance = nearest,
                steering = final_steering,
                throttle = final_throttle,
                brake    = 1.0 if current_mode == 'STOP' else 0.0,
                timestamp= time.time(),
                frame_id = msg.frame_id,
            )
            control_channel.write_obstacle(obstacle_msg)

            # ── FPS ───────────────────────────────────────────────────────────
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps       = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # ── Web viewer ────────────────────────────────────────────────────
            if web_viewer is not None:
                annotated = _draw(color_bgr, boxes, distances, class_ids,
                                  current_mode, final_steering, fps)
                web_viewer.broadcast_frame(annotated)
                web_viewer.broadcast_status({
                    'fps':              fps,
                    'action':           current_mode,
                    'steering':         final_steering,
                    'throttle':         final_throttle,
                    'nearest_distance': nearest,
                    'overtaking_state': current_mode,
                    'lane_detected':    lane_detected,
                    'lkas_steering':    lkas_steering,
                    'left_lane_x':      left_x,
                    'right_lane_x':     right_x,
                })

    except KeyboardInterrupt:
        print("\n[RUN] Stopped by user")

    finally:
        if car is not None:
            car.throttle = 0.0
            car.steering = 0.0
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
        print("[RUN] Cleanup done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML-based obstacle avoidance")
    parser.add_argument('--web-port', type=int, default=8082,
                        help='Web viewer port (default 8082, 0 to disable)')
    parser.add_argument('--motor', action='store_true',
                        help='Enable JetRacer motor output (default: simulation)')
    args = parser.parse_args()
    main(web_port=args.web_port, enable_motor=args.motor)