"""
JetRacer Collision Avoidance with YOLOv11 + RealSense Depth
Implements algorithm-based collision avoidance and car following
- Detects cars and pedestrians
- Tracks distance changes
- Controls throttle based on distance threshold
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
MODEL_PATH = "/home/siwoo/ADS-Skynet/Yolo_object_detection/yolov11/results/runs/detect/train/weights/best.pt"
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

# Collision Avoidance Parameters
CRUISE_THROTTLE = 0.5
MIN_DISTANCE = 0.4 #0.3 #0.2  # meters (minimum safe distance)
THROTTLE_DECREMENT = 0.2
STOP_THROTTLE = 0.0
DISTANCE_THRESHOLD = 0.05  # meters (to detect if distance is changing)

# FPS display
DISPLAY_FPS = True
FONT_SCALE = 0.5
LINE_THICKNESS = 2

# ------------------------------------------------

class CollisionAvoidanceController:
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

        # Collision Avoidance State
        self.previous_distance = None
        self.current_throttle = CRUISE_THROTTLE
        self.target_detected = False
        self.detected_class_name = None

    # -------------------- Depth Utils --------------------
    def get_distance_at_point(self, depth_frame, x, y):
        """Get depth distance at a specific pixel"""
        distance = depth_frame.get_distance(x, y)
        return distance if distance > 0 else -1

    def get_box_center_distance(self, depth_frame, box):
        """Calculate distance to bounding box center"""
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        distance = self.get_distance_at_point(depth_frame, cx, cy)
        
        if distance < 0:
            # Sample middle region if center point invalid
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
        """Draw bounding box and label on frame"""
        x1, y1, x2, y2 = map(int, box)
        
        # Color code: green for car, red for pedestrian
        if CLASS_NAMES[class_id] == "car":
            color = (0, 255, 0)
        elif CLASS_NAMES[class_id] == "pedestrian":
            color = (0, 0, 255)
        else:
            color = (255, 165, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)
        label = f"{CLASS_NAMES[class_id]}"
        
        if distance > 0:
            label += f" {distance:.2f}m"
        
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                    FONT_SCALE, (255, 255, 255), 1)
        return frame

    # -------------------- Collision Avoidance Algorithm --------------------
    def compute_collision_avoidance_throttle(self, current_distance):
        """
        Algorithm for collision avoidance based on distance tracking
        
        Step 1: Detect Object as car or pedestrian (handled in main loop)
        Step 2: If car or pedestrian detected Goto Step 3
        Step 3: Is current distance < previous distance Goto Step 4 Else Goto Step 5
        Step 4: Is current distance < distance minimum Goto Step 6 Else Goto Step 5
        Step 5: Continue with Throttle = 0.5
        Step 6: Throttle = Throttle - 0.2
        Step 7: Is current distance != previous distance Goto Step 4 Else Goto Step 8
        Step 8: Set Throttle = 0
        Step 9: END
        """
        
        # If no previous distance recorded, initialize it
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return CRUISE_THROTTLE
        
        # Step 3: Is current distance < previous distance (object getting closer)?
        if current_distance < self.previous_distance:
            # Step 4: Is current distance < minimum safe distance?
            if current_distance < MIN_DISTANCE:
                # Step 6: Reduce throttle
                self.current_throttle = max(STOP_THROTTLE, self.current_throttle - THROTTLE_DECREMENT)
                print(f"⚠️  Object approaching! Distance: {current_distance:.2f}m, Throttle reduced to: {self.current_throttle:.2f}")
            else:
                # Step 5: Continue with normal throttle
                self.current_throttle = CRUISE_THROTTLE
        else:
            # Step 5: Distance not decreasing, continue with normal throttle
            self.current_throttle = CRUISE_THROTTLE
        
        # Step 7: Check if distance is still changing
        distance_changed = abs(current_distance - self.previous_distance) > DISTANCE_THRESHOLD
        if not distance_changed and self.current_throttle < CRUISE_THROTTLE:
            # Step 8: Distance stopped changing at reduced throttle, stop completely
            self.current_throttle = STOP_THROTTLE
            print("🛑 Object detected close by. Stopping.")
        
        # Update previous distance for next iteration
        self.previous_distance = current_distance
        
        return self.current_throttle

    def reset_state(self):
        """Reset collision avoidance state when no object detected"""
        self.previous_distance = None
        self.current_throttle = CRUISE_THROTTLE
        self.target_detected = False
        self.detected_class_name = None

    # -------------------- Main Loop --------------------
    def run(self):
        print("Starting Collision Avoidance system. Press 'q' to exit.")
        print(f"Min safe distance: {MIN_DISTANCE}m")
        print(f"Throttle decrement: {THROTTLE_DECREMENT}")
        
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
                
                closest_target = None
                closest_distance = float('inf')
                target_found = False

                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    # Find car or pedestrian with closest distance
                    for box, class_id in zip(boxes, class_ids):
                        class_name = CLASS_NAMES[class_id]
                        
                        # Step 1: Detect Object as car or pedestrian
                        if class_name in ["car", "pedestrian"]:
                            cx, cy, dist = self.get_box_center_distance(depth_frame, box)
                            
                            # Draw all detected cars and pedestrians
                            frame = self.draw_detection(frame, box, class_id, dist)
                            
                            # Track closest target
                            if dist > 0 and dist < closest_distance:
                                closest_distance = dist
                                closest_target = (box, class_id, dist, class_name)
                                target_found = True

                # -------------------- Step 2: If car or pedestrian detected --------------------
                if target_found and closest_target is not None:
                    _, class_id, distance, class_name = closest_target
                    self.target_detected = True
                    self.detected_class_name = class_name
                    
                    # Apply collision avoidance algorithm
                    throttle = self.compute_collision_avoidance_throttle(distance)
                    self.car.throttle = throttle
                    
                    # Display throttle info on frame
                    throttle_text = f"Throttle: {throttle:.2f} | Target: {class_name} | Dist: {distance:.2f}m"
                    cv2.putText(frame, throttle_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 255, 0), 2)
                else:
                    # No target detected, reset to cruise
                    if self.target_detected:
                        print("✅ No target detected. Resuming cruise.")
                        self.reset_state()
                    
                    self.car.throttle = CRUISE_THROTTLE
                    cv2.putText(frame, "Cruise: 0.5", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 255, 0), 2)

                # -------------------- Display FPS --------------------
                self.frame_count += 1
                if DISPLAY_FPS:
                    elapsed = time.time() - self.start_time
                    if elapsed > 1:
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.start_time = time.time()
                    cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Collision Avoidance System", frame)

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
    controller = CollisionAvoidanceController()
    controller.run()
