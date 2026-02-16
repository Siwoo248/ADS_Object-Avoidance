"""
Real-time YOLO object detection with depth and Obstacle Avoidance.
Reads color + depth frames from the unified image shared memory and
writes obstacle avoidance decisions to the control shared memory.

Acts as both a CV viewer and a pre-decision API for the LKAS pipeline.
Requires: lkas --broadcast, vehicle running first.
"""

import sys
import cv2
import numpy as np
import time
import torch
import importlib.util
from pathlib import Path

# JetRacer motor control
try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    JETRACER_AVAILABLE = True
except ImportError:
    print("JetRacer library not available - running in simulation mode")
    JETRACER_AVAILABLE = False

# Import LKAS shared memory channels and types
try:
    from lkas.integration.shared_memory.channels import (
        SharedMemoryImageChannel,
        SharedMemoryControlChannel,
    )
    from lkas.integration.shared_memory.messages import (
        ObstacleMessage,
        ObstacleAction,
    )
    from lkas import LKASClient as LKAS
    from common.visualization.visualizer import LKASVisualizer
    LKAS_AVAILABLE = True
except ImportError:
    print("LKAS not available - running without lane keeping integration")
    LKAS_AVAILABLE = False

# Fix PyTorch 2.6+ weights loading issue before importing ultralytics
_orig_torch_load = torch.load

def torch_load_with_legacy_support(*args, **kwargs):
    """Wrapper for torch.load that supports legacy weights."""
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = torch_load_with_legacy_support

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from ultralytics import YOLO

# Import the obstacle avoidance system
from obstacle_avoidance import ObstacleAvoidanceSystem

# Load configuration values directly from config file
config_path = parent_dir / "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)

# Set __file__ in the module before executing
config_module.__file__ = str(config_path)
spec.loader.exec_module(config_module)

# Extract config values
MODEL_PATH = config_module.MODEL_PATH
CLASS_NAMES = config_module.CLASS_NAMES
CONFIDENCE_THRESHOLD = config_module.CONFIDENCE_THRESHOLD
IOU_THRESHOLD = config_module.IOU_THRESHOLD
DEVICE = config_module.DEVICE
DISPLAY_FPS = config_module.DISPLAY_FPS
DISPLAY_CONFIDENCE = config_module.DISPLAY_CONFIDENCE
LINE_THICKNESS = config_module.LINE_THICKNESS
FONT_SCALE = config_module.FONT_SCALE

# Map avoidance action strings to ObstacleAction enum
_ACTION_MAP = {
    'NORMAL': ObstacleAction.NORMAL,
    'STOP': ObstacleAction.STOP,
    'AVOID_LEFT': ObstacleAction.AVOID_LEFT,
    'AVOID_RIGHT': ObstacleAction.AVOID_RIGHT,
    'SLOW': ObstacleAction.SLOW,
    'MONITOR': ObstacleAction.NORMAL,
    'IGNORE': ObstacleAction.NORMAL,
}


class YOLODepthDetectorWithAvoidance:
    """
    Real-time YOLO detection with depth and obstacle avoidance.
    Reads frames from unified image shared memory (color + depth).
    Writes obstacle avoidance decisions to control shared memory.
    """

    def __init__(self, model_path=MODEL_PATH, device=0,
                 conf_threshold=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD,
                 enable_motor_control=False, enable_lkas_integration=True):
        """
        Initialize the YOLO depth detector with avoidance and LKAS integration.

        Args:
            model_path (str): Path to the YOLO model
            device (int): GPU device index (will use CPU if no CUDA available)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for NMS
            enable_motor_control (bool): Enable actual motor control (default: False for safety)
            enable_lkas_integration (bool): Enable LKAS integration for lane keeping (default: True)
        """
        # Load YOLO model with validation
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)

        # Set device based on CUDA availability
        # Note: Don't call model.to(device) upfront — on Jetson Orin the GPU
        # memory is shared with lkas, so eagerly moving the whole model can OOM.
        # Instead, pass device= at inference time and let ultralytics handle it.
        if torch.cuda.is_available():
            self.device = device
            free_mb, total_mb = [x / 1024**2 for x in torch.cuda.mem_get_info(0)]
            print(f"Using GPU device {device} ({free_mb:.0f}MB free / {total_mb:.0f}MB total)")
        else:
            self.device = 'cpu'
            print("CUDA not available - using CPU for inference")

        print(f"YOLO model loaded successfully")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = CLASS_NAMES

        # Connect to shared memory via LKAS channel classes
        print("Connecting to shared memory...")
        print("  Make sure 'lkas --broadcast' and 'vehicle' are running.")

        # Image channel: read color + depth from unified image SHM
        self.image_channel = SharedMemoryImageChannel(name="image", create=False,
                                                      retry_count=60, retry_delay=1.0)

        # Control channel: read LKAS control, write obstacle data
        self.control_channel = SharedMemoryControlChannel(name="control", create=False,
                                                          retry_count=30, retry_delay=1.0)

        # Read actual frame dimensions from first image
        first_msg = self.image_channel.read_blocking(timeout=10.0)
        if first_msg is not None:
            self.frame_width = first_msg.width
            self.frame_height = first_msg.height
        else:
            self.frame_width, self.frame_height = 768, 384
        print(f"Frame size from shared memory: {self.frame_width}x{self.frame_height}")

        # Initialize LKAS client for lane detection visualization
        self.enable_lkas = enable_lkas_integration and LKAS_AVAILABLE
        self.lkas = None
        self.lkas_steering = 0.0
        self.lkas_lane_detected = False

        if self.enable_lkas:
            print("\nInitializing LKAS client for detection readout...")
            try:
                self.lkas = LKAS(
                    image_shm_name="image",
                    detection_shm_name="detection",
                    control_shm_name="control",
                )
                self.lkas_visualizer = LKASVisualizer(
                    image_width=self.frame_width, image_height=self.frame_height
                )
                print("LKAS integration ENABLED")
            except Exception as e:
                print(f"Failed to initialize LKAS: {e}")
                self.lkas = None
                self.enable_lkas = False
        else:
            print("\nLKAS integration DISABLED - using hardcoded lane boundaries")

        # Initialize Obstacle Avoidance System
        print("\nInitializing Obstacle Avoidance System...")
        self.avoidance = ObstacleAvoidanceSystem(
            frame_width=self.frame_width, frame_height=self.frame_height
        )

        # Initialize JetRacer motor control
        self.enable_motor_control = enable_motor_control
        self.car = None

        if JETRACER_AVAILABLE and self.enable_motor_control:
            print("\nInitializing JetRacer motor control...")
            try:
                self.car = NvidiaRacecar()
                self.car.throttle = -0.5
                self.car.steering = 0.0
                print("JetRacer control ENABLED - Motors are ACTIVE!")
                print("WARNING: Vehicle WILL move! Press 'q' to stop immediately!")
            except Exception as e:
                print(f"Failed to initialize JetRacer: {e}")
                self.car = None
                self.enable_motor_control = False
        else:
            if not JETRACER_AVAILABLE:
                print("\nMotor control DISABLED - JetRacer library not available (simulation mode)")
            else:
                print("\nMotor control DISABLED - Set enable_motor_control=True to enable")

        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Cache for storing detection results from previous frames
        self.last_results = None
        self.last_boxes = None
        self.last_confidences = None
        self.last_class_ids = None
        self.last_distances = None

        print("Shared memory integration initialized successfully")
        print(f"Inference settings: conf={self.conf_threshold}, iou={self.iou_threshold}")
        print("\n" + "=" * 60)
        print("System ready! Press 'q' to exit, 'r' to reset avoidance")
        print("=" * 60 + "\n")

    def get_detections(self, frame):
        """
        Run YOLO inference on a frame with memory management.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            Results object with detections
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            return results
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"Memory Error: {e}")
                if torch.cuda.is_available():
                    print("Attempting GPU memory recovery...")
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            else:
                print(f"Error during inference: {e}")
            return None
        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    def get_distance_at_point(self, depth_array, depth_scale, x, y):
        """Get distance at a specific pixel coordinate."""
        h, w = depth_array.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return -1
        raw = int(depth_array[y, x])
        if raw == 0:
            return -1
        return raw * depth_scale

    def get_box_center_distance(self, depth_array, depth_scale, box):
        """Calculate distance to the center of a bounding box."""
        h, w = depth_array.shape
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        distance = self.get_distance_at_point(depth_array, depth_scale, center_x, center_y)

        if distance < 0:
            box_width = x2 - x1
            box_height = y2 - y1
            margin_x = int(box_width * 0.2)
            margin_y = int(box_height * 0.2)

            sample_x1 = max(x1 + margin_x, 0)
            sample_y1 = max(y1 + margin_y, 0)
            sample_x2 = min(x2 - margin_x, w - 1)
            sample_y2 = min(y2 - margin_y, h - 1)

            valid_distances = []
            for py in range(sample_y1, sample_y2, 5):
                for px in range(sample_x1, sample_x2, 5):
                    d = self.get_distance_at_point(depth_array, depth_scale, px, py)
                    if d > 0:
                        valid_distances.append(d)

            if valid_distances:
                distance = np.median(valid_distances)

        return center_x, center_y, distance

    def draw_detection_with_distance(self, frame, box, confidence, class_id, distance):
        """Draw bounding box with class label, confidence, and distance."""
        x1, y1, x2, y2 = map(int, box)
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
        color = self._get_color_for_class(class_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)

        label_text = f"{class_name}"
        if DISPLAY_CONFIDENCE:
            label_text += f": {confidence:.2f}"

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)

        if distance > 0:
            distance_text = f"Distance: {distance:.2f}m"
            distance_color = (0, 255, 0)
        else:
            distance_text = "Distance: N/A"
            distance_color = (0, 0, 255)

        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
        )
        cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5),
                      (x1 + text_width + 5, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 3, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)

        (dist_text_width, dist_text_height), dist_baseline = cv2.getTextSize(
            distance_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE - 0.1, 1
        )
        cv2.rectangle(frame, (x1, y2),
                      (x1 + dist_text_width + 5, y2 + dist_text_height + 5), distance_color, -1)
        cv2.putText(frame, distance_text, (x1 + 3, y2 + dist_text_height + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE - 0.1, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def _get_color_for_class(self, class_id):
        """Get a consistent color for each class."""
        colors = [
            (0, 255, 0),      # car - green
            (255, 0, 0),      # crosswalk - blue
            (0, 165, 255),    # highway_entry - orange
            (255, 165, 0),    # highway_exit - cyan
            (128, 0, 128),    # no_entry - purple
            (0, 255, 255),    # onewayroad - yellow
            (255, 255, 0),    # parking - light blue
            (0, 128, 255),    # pedestrian - red-orange
            (128, 128, 0),    # priority - dark cyan
            (0, 0, 255),      # roadblock - red
            (255, 0, 255),    # roundabout - magenta
            (255, 255, 255),  # stop - white
            (0, 128, 0),      # trafficlight - dark green
        ]
        return colors[class_id % len(colors)]

    def update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

    def draw_fps(self, frame):
        """Draw FPS on frame."""
        if DISPLAY_FPS:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def run(self):
        """Run real-time detection with depth measurement and obstacle avoidance."""
        print("Starting real-time detection with obstacle avoidance...")
        print("Reading frames from unified image shared memory (color + depth)")
        print(f"Model classes: {', '.join(self.class_names)}")

        try:
            frame_count = 0
            inference_skip = 2  # Process every 2nd frame to reduce memory pressure

            while True:
                # Read color + depth from unified image shared memory
                msg = self.image_channel.read()
                if msg is None:
                    time.sleep(0.001)
                    continue

                color_image = msg.image
                depth_array = msg.depth_image
                depth_scale = msg.depth_scale

                # Fallback if no depth available
                if depth_array is None:
                    depth_array = np.zeros(
                        (self.frame_height, self.frame_width), dtype=np.uint16
                    )
                    depth_scale = 0.0

                # Start with color image
                annotated_frame = color_image.copy()

                # Lists to store detections for avoidance system
                all_boxes = []
                all_distances = []
                all_detections = []
                nearest_distance = -1.0

                # Only run inference on every Nth frame to reduce memory pressure
                if frame_count % inference_skip == 0:
                    results = self.get_detections(color_image)

                    if results is None:
                        print("Failed to run inference on this frame - retrying...")
                        continue

                    self.last_results = results
                    self.last_boxes = None
                    self.last_confidences = None
                    self.last_class_ids = None
                    self.last_distances = None

                    if len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                        self.last_boxes = boxes
                        self.last_confidences = confidences
                        self.last_class_ids = class_ids
                        self.last_distances = []

                        for box, conf, class_id in zip(boxes, confidences, class_ids):
                            center_x, center_y, distance = self.get_box_center_distance(
                                depth_array, depth_scale, box
                            )
                            self.last_distances.append(distance)

                            all_boxes.append(box)
                            all_distances.append(distance)
                            all_detections.append(1)

                            annotated_frame = self.draw_detection_with_distance(
                                annotated_frame, box, conf, class_id, distance
                            )

                elif self.last_boxes is not None:
                    for box, conf, class_id, distance in zip(
                        self.last_boxes, self.last_confidences,
                        self.last_class_ids, self.last_distances
                    ):
                        annotated_frame = self.draw_detection_with_distance(
                            annotated_frame, box, conf, class_id, distance
                        )
                        all_boxes.append(box)
                        all_distances.append(distance)
                        all_detections.append(1)

                # Track nearest obstacle distance
                valid_distances = [d for d in all_distances if d > 0]
                if valid_distances:
                    nearest_distance = min(valid_distances)

                # Get LKAS lane detection and update lane boundaries
                lkas_steering = 0.0
                lane_detected = False

                if self.enable_lkas and self.lkas:
                    try:
                        detection = self.lkas.get_detection(timeout=0.01)
                        if detection:
                            if detection.segmentation_mask is not None:
                                annotated_frame = self.lkas_visualizer.draw_segmentation(
                                    annotated_frame, detection.segmentation_mask, alpha=0.4
                                )

                            if detection.left_lane and detection.right_lane:
                                lane_detected = True
                                left_x = detection.left_lane.x1
                                right_x = detection.right_lane.x1

                                if 0 <= left_x < right_x <= self.frame_width:
                                    old_left = self.avoidance.LEFT_LANE_X
                                    old_right = self.avoidance.RIGHT_LANE_X

                                    self.avoidance.LEFT_LANE_X = int(left_x)
                                    self.avoidance.RIGHT_LANE_X = int(right_x)
                                    self.avoidance.LANE_WIDTH_PIXELS = int(right_x) - int(left_x)

                                    if abs(old_left - left_x) > 10 or abs(old_right - right_x) > 10:
                                        print(f"Lane updated: L={int(left_x)}px, R={int(right_x)}px "
                                              f"(width={self.avoidance.LANE_WIDTH_PIXELS}px)")

                        control = self.lkas.get_control(timeout=0.01)
                        if control is not None:
                            lkas_steering = control.steering
                    except Exception as e:
                        print(f"LKAS error: {e}")

                self.lkas_steering = lkas_steering
                self.lkas_lane_detected = lane_detected

                # Process obstacles with avoidance system
                action, avoidance_steering, throttle, enable_lkas_flag = \
                    self.avoidance.process_obstacles(all_detections, all_distances, all_boxes)

                # Combine LKAS steering with obstacle avoidance steering
                if self.enable_lkas and lane_detected and enable_lkas_flag:
                    if action in ('NORMAL', 'MONITOR', 'IGNORE', None):
                        final_steering = lkas_steering
                    else:
                        final_steering = avoidance_steering
                else:
                    final_steering = avoidance_steering

                final_throttle = throttle

                # Write obstacle avoidance status to control shared memory
                # This is read by the decision server to integrate into LKAS control
                obstacle_action = _ACTION_MAP.get(action, ObstacleAction.NORMAL)
                # For lane-change sub-states, map to the base avoidance action
                if action and action.startswith('LANE_CHANGE_'):
                    obstacle_action = ObstacleAction.AVOID_LEFT  # lane changes default left

                obstacle_msg = ObstacleMessage(
                    active=(obstacle_action != ObstacleAction.NORMAL),
                    action=obstacle_action,
                    distance=nearest_distance,
                    steering=avoidance_steering,
                    throttle=throttle,
                    brake=1.0 if action == 'STOP' else 0.0,
                    timestamp=time.time(),
                    frame_id=msg.frame_id,
                )
                self.control_channel.write_obstacle(obstacle_msg)

                # Apply steering and throttle to JetRacer motors (direct control)
                if self.enable_motor_control and self.car is not None:
                    self.car.steering = final_steering
                    self.car.throttle = -final_throttle

                    if abs(final_steering) > 0.01 or abs(final_throttle) > 0.01:
                        mode = "LKAS+AVOID" if (self.enable_lkas and lane_detected) else "AVOID"
                        print(f"[{mode}] steering={final_steering:+.2f}, "
                              f"throttle={-final_throttle:.2f} | action={action}")

                # Draw lane overlay
                annotated_frame = self.avoidance.draw_lane_overlay(annotated_frame)

                # Draw LKAS status if enabled
                if self.enable_lkas:
                    lkas_status = "LKAS: ON" if lane_detected else "LKAS: NO LANES"
                    lkas_color = (0, 255, 0) if lane_detected else (0, 165, 255)
                    cv2.putText(annotated_frame, lkas_status, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, lkas_color, 2)
                    if lane_detected:
                        cv2.putText(annotated_frame, f"LKAS Steer: {lkas_steering:+.2f}",
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Update and draw FPS
                self.update_fps()
                annotated_frame = self.draw_fps(annotated_frame)

                # Draw status panel
                annotated_frame = self.avoidance.draw_status_panel(annotated_frame)

                frame_count += 1

                # Display frame
                cv2.imshow("YOLO Detection with Obstacle Avoidance", annotated_frame)

                # Clear GPU cache every frame
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("\n[USER] Resetting avoidance system...")
                    self.avoidance.reset()

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")

        # Emergency stop motors
        if self.enable_motor_control and self.car is not None:
            print("Stopping motors...")
            self.car.throttle = 0.0
            self.car.steering = 0.0

        # Write final "inactive" obstacle status
        try:
            self.control_channel.write_obstacle(ObstacleMessage(
                active=False,
                action=ObstacleAction.NORMAL,
                timestamp=time.time(),
            ))
        except Exception:
            pass

        # Close LKAS connection
        if self.lkas:
            try:
                self.lkas.close()
                print("LKAS connection closed")
            except Exception:
                pass

        # Close shared memory channels (don't unlink — lkas owns them)
        try:
            self.image_channel.close()
        except Exception:
            pass
        try:
            self.control_channel.close()
        except Exception:
            pass

        cv2.destroyAllWindows()
        print("Done!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLO Object Detection with Obstacle Avoidance and LKAS Integration"
    )
    parser.add_argument('--motors', action='store_true',
                        help='Enable motor control (default: False for safety)')
    parser.add_argument('--no-lkas', action='store_true',
                        help='Disable LKAS integration (standalone mode)')
    parser.add_argument('--conf', type=float, default=CONFIDENCE_THRESHOLD,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=IOU_THRESHOLD,
                        help='IOU threshold')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("YOLO Object Detection + Obstacle Avoidance + LKAS")
    print("(Shared Memory Mode - requires lkas + vehicle to be running)")
    print("=" * 60)
    print(f"Motor Control: {'ENABLED' if args.motors else 'DISABLED'}")
    print(f"LKAS Integration: {'DISABLED' if args.no_lkas else 'ENABLED'}")
    print(f"Confidence: {args.conf}, IOU: {args.iou}")
    print("=" * 60 + "\n")

    # Pre-flight check: show which shared memory segments exist
    import subprocess
    shm_check = subprocess.run(
        ['ls', '/dev/shm/'],
        capture_output=True, text=True
    )
    expected = ['image', 'detection', 'control']
    found = [name for name in expected if name in shm_check.stdout.split()]
    missing = [name for name in expected if name not in shm_check.stdout.split()]
    if found:
        print(f"Shared memory found:   {', '.join(found)}")
    if missing:
        print(f"Shared memory MISSING: {', '.join(missing)}")
        print("  -> Make sure 'lkas --broadcast' and 'vehicle' are running FIRST!")
        print()

    if args.motors:
        print("WARNING: MOTOR CONTROL IS ENABLED!")
        print("The vehicle WILL move! Press 'q' to emergency stop!")
        print("Press Ctrl+C within 3 seconds to abort...\n")
        time.sleep(3)

    try:
        detector = YOLODepthDetectorWithAvoidance(
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            enable_motor_control=args.motors,
            enable_lkas_integration=not args.no_lkas
        )
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
