"""
JetRacer Autonomous Controller with YOLOv11 + RealSense Depth
- Detects pedestrians and crosswalks
- Controls throttle based on detections
- Includes safety failsafe
"""

import time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLO

try:
    from jetracer.nvidia_racecar import NvidiaRacecar
except (ModuleNotFoundError, ImportError) as e:
    print(f"Warning: Failed to import NvidiaRacecar: {e}")
    print("Jetson GPIO libraries may not be installed. Install with:")
    print("  sudo apt-get install python3-jetson-gpio")
    print("  sudo apt-get install python3-pip")
    print("  pip install adafruit-blinka adafruit-circuitpython-servokit")
    raise

# -------------------- Config --------------------
MODEL_PATH = "/home/siwoo/ADS-Skynet/Yolo_object_detection/yolov11/results/runs/detect/train/weights/best.pt"   # Path to your trained YOLO model
CLASS_NAMES = [
    'one_way_road_sign',
    'highway_entrance_sign',
    'stop_sign',
    'roundabout_sign',
    'parking_sign',
    'crosswalk_sign',
    'no_entry_road_sign',
    'highway_exit_sign',
    'priority_sign',
    'traffic_light',
    'highway_separator',
    'pedestrian',
    'car'
]

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
DEVICE = 0  # 0 for GPU, 'cpu' for CPU

# Vehicle control
CRUISE_THROTTLE = 0.5
CROSSWALK_THROTTLE = 0.3
STOP_THROTTLE = 0.0
PED_STOP_DISTANCE = 0.4  # 40 centimeters
STATE_TIMEOUT = 0.5      # seconds

# FPS display
DISPLAY_FPS = True
FONT_SCALE = 0.5
LINE_THICKNESS = 2

# ------------------------------------------------

class JetRacerController:
    def __init__(self):
        # Load YOLO model
        print(f"Loading YOLO model from {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH)
        if torch.cuda.is_available():
            self.model.to(f"cuda:{DEVICE}")
            print(f"Using GPU {DEVICE} for inference")
        else:
            self.model.to("cpu")
            print("Using CPU for inference")

        # Initialize RealSense
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        # JetRacer car
        self.car = NvidiaRacecar()
        self.car.throttle = 0.0

        # FPS tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

    # -------------------- Depth Utils --------------------
    def get_distance_at_point(self, depth_frame, x, y):
        distance = depth_frame.get_distance(x, y)
        return distance if distance > 0 else -1

    def get_box_center_distance(self, depth_frame, box):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        distance = self.get_distance_at_point(depth_frame, cx, cy)
        if distance < 0:
            # Sample middle 60% region
            w, h = x2 - x1, y2 - y1
            mx1, my1 = x1 + w//5, y1 + h//5
            mx2, my2 = x2 - w//5, y2 - h//5
            valid = [self.get_distance_at_point(depth_frame, px, py)
                     for py in range(my1, my2, 5)
                     for px in range(mx1, mx2, 5)]
            valid = [d for d in valid if d > 0]
            if valid:
                distance = np.median(valid)
        return cx, cy, distance

    # -------------------- Drawing --------------------
    def draw_detection(self, frame, box, class_id, distance):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if class_id == 0 else (255, 0, 0) if class_id == 1 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)
        label = f"{CLASS_NAMES[class_id]}"
        #label = f"{CLASS_NAMES}"
        print(f"\t{label}, and {class_id}")

        if distance > 0:
            label += f" {distance:.2f}m"
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 1)
        return frame

    # -------------------- Control --------------------
    def compute_throttle(self, crosswalk, ped_dist):
        # Timeout failsafe
        if time.time() - self.last_update > STATE_TIMEOUT:
            return STOP_THROTTLE
        if ped_dist is not None and ped_dist < PED_STOP_DISTANCE:
            return STOP_THROTTLE
        if crosswalk:
            return CROSSWALK_THROTTLE
        return CRUISE_THROTTLE

    # -------------------- Main Loop --------------------
    def run(self):
        print("Starting main loop. Press 'q' to exit.")
        self.last_update = time.time()
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data())

                # -------------------- YOLO Inference --------------------
                results = self.model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=DEVICE)
                crosswalk_detected = False
                closest_ped_distance = None

                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    for box, class_id in zip(boxes, class_ids):
                        cx, cy, dist = self.get_box_center_distance(depth_frame, box)
                        frame = self.draw_detection(frame, box, class_id, dist)

                        # Track crosswalk & pedestrian
                        if CLASS_NAMES[class_id] == "crosswalk":
                            crosswalk_detected = True
                        if CLASS_NAMES[class_id] == "pedestrian" and dist > 0:
                            if closest_ped_distance is None:
                                closest_ped_distance = dist
                            else:
                                closest_ped_distance = min(closest_ped_distance, dist)

                # -------------------- Throttle Control --------------------
                self.last_update = time.time()
                throttle = self.compute_throttle(crosswalk_detected, closest_ped_distance)
                self.car.throttle = throttle

                # -------------------- Display --------------------
                self.frame_count += 1
                if DISPLAY_FPS:
                    elapsed = time.time() - self.start_time
                    if elapsed > 1:
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.start_time = time.time()
                    cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("JetRacer YOLO Depth", frame)

                # -------------------- Exit --------------------
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")

        finally:
            print("Stopping car and cleaning up...")
            self.car.throttle = 0.0
            self.pipeline.stop()
            cv2.destroyAllWindows()

# -------------------- Main --------------------
if __name__ == "__main__":
    controller = JetRacerController()
    controller.run()
