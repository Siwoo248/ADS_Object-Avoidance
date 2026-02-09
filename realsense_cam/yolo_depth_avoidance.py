"""
Real-time YOLO object detection with RealSense depth and Obstacle Avoidance.
Integrates the obstacle avoidance system with YOLO detection.
"""

import sys
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import torch
import importlib.util
from pathlib import Path

# JetRacer motor control
try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    JETRACER_AVAILABLE = True
except ImportError:
    print("⚠️ JetRacer library not available - running in simulation mode")
    JETRACER_AVAILABLE = False

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
config_path = parent_dir / "Yolo_object_detection" / "config.py"
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


class YOLODepthDetectorWithAvoidance:
    """
    Real-time YOLO detection with RealSense depth and obstacle avoidance.
    Combines object detection, depth measurement, and avoidance decisions.
    """
    
    def __init__(self, model_path=MODEL_PATH, device=0, 
                 conf_threshold=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD,
                 enable_motor_control=False):
        """
        Initialize the YOLO depth detector with avoidance.
        
        Args:
            model_path (str): Path to the YOLO model
            device (int): GPU device index (will use CPU if no CUDA available)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for NMS
            enable_motor_control (bool): Enable actual motor control (default: False for safety)
        """
        # Load YOLO model with validation
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Set device based on CUDA availability
        if torch.cuda.is_available():
            device_str = f'cuda:{device}'
            self.device = device
            print(f"Using GPU device {device}")
        else:
            device_str = 'cpu'
            self.device = 'cpu'
            print("CUDA not available - using CPU for inference")
        
        self.model.to(device=device_str)
        print(f"YOLO model loaded successfully on {device_str}")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = CLASS_NAMES
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        print("Starting RealSense pipeline...")
        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            raise RuntimeError(f"Failed to start RealSense pipeline: {e}")
        
        # Create alignment object
        self.align = rs.align(rs.stream.color)
        
        # Get depth scale
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        
        # Get camera intrinsics
        self.color_intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        # Initialize Obstacle Avoidance System
        print("\nInitializing Obstacle Avoidance System...")
        self.avoidance = ObstacleAvoidanceSystem(frame_width=640, frame_height=480)
        
        # Initialize JetRacer motor control
        self.enable_motor_control = enable_motor_control
        self.car = None
        
        if JETRACER_AVAILABLE and self.enable_motor_control:
            print("\n🚗 Initializing JetRacer motor control...")
            try:
                self.car = NvidiaRacecar()
                self.car.throttle = 0.0
                self.car.steering = 0.0
                print("✅ JetRacer control ENABLED - Motors are ACTIVE!")
                print("⚠️  WARNING: Vehicle WILL move! Press 'q' to stop immediately!")
            except Exception as e:
                print(f"❌ Failed to initialize JetRacer: {e}")
                self.car = None
                self.enable_motor_control = False
        else:
            if not JETRACER_AVAILABLE:
                print("\n🔒 Motor control DISABLED - JetRacer library not available (simulation mode)")
            else:
                print("\n🔒 Motor control DISABLED - Set enable_motor_control=True to enable")
                print("   (This is safer for testing!)")
        
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
        
        print("RealSense camera initialized successfully")
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
            # Clear GPU cache if using CUDA
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
    
    def get_distance_at_point(self, depth_frame, x, y):
        """
        Get distance at a specific pixel coordinate.
        
        Args:
            depth_frame: RealSense depth frame
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            float: Distance in meters (or -1 if out of range)
        """
        distance = depth_frame.get_distance(x, y)
        if distance == 0:
            return -1
        return distance
    
    def get_box_center_distance(self, depth_frame, box):
        """
        Calculate distance to the center of a bounding box.
        Uses a region-based approach to find valid depth data.
        
        Args:
            depth_frame: RealSense depth frame
            box (np.ndarray): Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            tuple: (center_x, center_y, distance_in_meters)
        """
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # First try the center point
        distance = self.get_distance_at_point(depth_frame, center_x, center_y)
        
        # If center point has invalid depth, sample the region around the bounding box
        if distance < 0:
            # Define a region (middle 60% of the bounding box) to sample
            box_width = x2 - x1
            box_height = y2 - y1
            margin_x = int(box_width * 0.2)
            margin_y = int(box_height * 0.2)
            
            sample_x1 = max(x1 + margin_x, 0)
            sample_y1 = max(y1 + margin_y, 0)
            sample_x2 = min(x2 - margin_x, 639)
            sample_y2 = min(y2 - margin_y, 479)
            
            # Collect valid depth values from the region
            valid_distances = []
            for py in range(sample_y1, sample_y2, 5):  # Sample every 5 pixels
                for px in range(sample_x1, sample_x2, 5):
                    d = self.get_distance_at_point(depth_frame, px, py)
                    if d > 0:
                        valid_distances.append(d)
            
            # Use median of valid distances if available
            if valid_distances:
                distance = np.median(valid_distances)
        
        return center_x, center_y, distance
    
    def draw_detection_with_distance(self, frame, box, confidence, class_id, distance):
        """
        Draw bounding box with class label, confidence, and distance.
        
        Args:
            frame (np.ndarray): Input frame
            box (np.ndarray): Bounding box [x1, y1, x2, y2]
            confidence (float): Detection confidence
            class_id (int): Class ID
            distance (float): Distance in meters (-1 if out of range)
            
        Returns:
            np.ndarray: Annotated frame
        """
        x1, y1, x2, y2 = map(int, box)
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
        color = self._get_color_for_class(class_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)
        
        # Prepare label text with confidence
        label_text = f"{class_name}"
        if DISPLAY_CONFIDENCE:
            label_text += f": {confidence:.2f}"
        
        # Get center coordinates
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Draw center point
        cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)
        
        # Prepare distance text
        if distance > 0:
            distance_text = f"Distance: {distance:.2f}m"
            distance_color = (0, 255, 0)  # Green for valid distance
        else:
            distance_text = "Distance: N/A"
            distance_color = (0, 0, 255)  # Red for invalid/out of range
        
        # Draw background rectangles for text
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label_text,
            (x1 + 3, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Draw distance text below bounding box
        (dist_text_width, dist_text_height), dist_baseline = cv2.getTextSize(
            distance_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE - 0.1, 1
        )
        cv2.rectangle(
            frame,
            (x1, y2),
            (x1 + dist_text_width + 5, y2 + dist_text_height + 5),
            distance_color,
            -1
        )
        cv2.putText(
            frame,
            distance_text,
            (x1 + 3, y2 + dist_text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE - 0.1,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        return frame
    
    def _get_color_for_class(self, class_id):
        """
        Get a consistent color for each class.
        
        Args:
            class_id (int): Class ID
            
        Returns:
            tuple: BGR color tuple
        """
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
            cv2.putText(
                frame,
                f"FPS: {self.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        return frame
    
    def run(self):
        """Run real-time detection with depth measurement and obstacle avoidance."""
        print("Starting real-time detection with obstacle avoidance...")
        print(f"Model classes: {', '.join(self.class_names)}")
        
        try:
            frame_count = 0
            inference_skip = 2  # Process every 2nd frame to reduce memory pressure
            
            while True:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                
                # Align depth to color
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Start with color image
                annotated_frame = color_image.copy()
                
                # Lists to store detections for avoidance system
                all_boxes = []
                all_distances = []
                all_detections = []
                
                # Only run inference on every Nth frame to reduce memory pressure
                if frame_count % inference_skip == 0:
                    # Run YOLO inference
                    results = self.get_detections(color_image)
                    
                    # Handle potential inference errors
                    if results is None:
                        print("Failed to run inference on this frame - retrying...")
                        continue
                    
                    # Store results for display
                    self.last_results = results
                    self.last_boxes = None
                    self.last_confidences = None
                    self.last_class_ids = None
                    self.last_distances = None
                    
                    if len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                        
                        # Store for reuse on skipped frames
                        self.last_boxes = boxes
                        self.last_confidences = confidences
                        self.last_class_ids = class_ids
                        self.last_distances = []
                        
                        # Process each detection
                        for box, conf, class_id in zip(boxes, confidences, class_ids):
                            center_x, center_y, distance = self.get_box_center_distance(depth_frame, box)
                            self.last_distances.append(distance)
                            
                            # Store for avoidance system
                            all_boxes.append(box)
                            all_distances.append(distance)
                            all_detections.append(1)
                            
                            # Draw detection with distance
                            annotated_frame = self.draw_detection_with_distance(
                                annotated_frame, box, conf, class_id, distance
                            )
                
                elif hasattr(self, 'last_boxes') and self.last_boxes is not None:
                    # Reuse previous detection results for skipped frames
                    for box, conf, class_id, distance in zip(self.last_boxes, self.last_confidences, 
                                                               self.last_class_ids, self.last_distances):
                        annotated_frame = self.draw_detection_with_distance(
                            annotated_frame, box, conf, class_id, distance
                        )
                        
                        # Store for avoidance system
                        all_boxes.append(box)
                        all_distances.append(distance)
                        all_detections.append(1)
                
                # Process obstacles with avoidance system
                action, steering, throttle = self.avoidance.process_obstacles(
                    all_detections, all_distances, all_boxes
                )
                
                # ✨ Apply steering and throttle to JetRacer motors
                if self.enable_motor_control and self.car is not None:
                    self.car.steering = steering
                    self.car.throttle = -throttle  # Note: JetRacer uses negative for forward
                    
                    # Log motor commands (only when not zero)
                    if abs(steering) > 0.01 or abs(throttle) > 0.01:
                        print(f"🚗 Motors: steering={steering:+.2f}, throttle={-throttle:.2f}")
                
                # Draw lane overlay
                annotated_frame = self.avoidance.draw_lane_overlay(annotated_frame)
                
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
            print("🛑 Stopping motors...")
            self.car.throttle = 0.0
            self.car.steering = 0.0
        
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print("Done!")


def main():
    """Main entry point."""
    try:
        enable_motors = False  # True / False
        
        detector = YOLODepthDetectorWithAvoidance(enable_motor_control=enable_motors)
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()