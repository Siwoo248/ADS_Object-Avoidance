"""
Real-time YOLO object detection inference with camera input.
Detects and annotates objects in real-time using YOLOv8s model.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
from pathlib import Path
from config import (
    MODEL_PATH,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    CAMERA_DEVICE_PATH,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
    CAMERA_ROTATION,
    DEVICE,
    DISPLAY_FPS,
    DISPLAY_CONFIDENCE,
    LINE_THICKNESS,
    FONT_SCALE
)


class YOLOInference:
    """
    Real-time YOLO object detection inference class.
    Handles model loading, inference, and visualization.
    """
    
    def __init__(self, model_path, device=0, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the YOLO inference engine.
        
        Args:
            model_path (str): Path to the YOLO model weights file
            device (int): GPU device index (0 for first GPU)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for NMS
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = CLASS_NAMES
        
        # Load model on GPU
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device=f'cuda:{device}')
        print(f"Model loaded successfully on GPU {device}")
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def preprocess_frame(self, frame):
        """
        Preprocess the camera frame for inference.
        
        Args:
            frame (np.ndarray): Input frame from camera
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        return frame
    
    def run_inference(self, frame):
        """
        Run YOLO inference on a frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Results object containing detections
        """
        # Run inference on GPU
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        return results
    
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            results: YOLO results object
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Extract detections from results
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
            
            # Draw each detection
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                
                # Draw bounding box
                color = self._get_color_for_class(class_id)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)
                
                # Prepare label text
                if DISPLAY_CONFIDENCE:
                    label = f"{class_name}: {conf:.2f}"
                else:
                    label = class_name
                
                # Draw background rectangle for text
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
                )
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width + 5, y1),
                    color,
                    -1
                )
                
                # Draw text label
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 3, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        return annotated_frame
    
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
        if elapsed_time > 1.0:  # Update every second
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def draw_fps(self, frame):
        """
        Draw FPS on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            np.ndarray: Frame with FPS drawn
        """
        if DISPLAY_FPS:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        return frame
    
    def run_camera_inference(self, camera_index=0, device_path=None):
        """
        Run real-time inference on camera feed.
        
        Args:
            camera_index (int): Camera device index (used if device_path is None)
            device_path (str): Linux device path (e.g., '/dev/video4') - takes precedence
        """
        # Determine camera source
        if device_path is not None:
            camera_source = device_path
            source_info = f"device path {device_path}"
        else:
            camera_source = camera_index
            source_info = f"camera index {camera_index}"
        
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera with {source_info}\n"
                f"Please check:\n"
                f"  1. Camera is properly connected\n"
                f"  2. Device path is correct (for /dev/videoX paths)\n"
                f"  3. Camera is not in use by another application\n"
                f"  4. You have proper permissions to access the camera"
            )
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        print(f"Camera opened successfully from {source_info}")
        print(f"Press 'q' to quit.")
        print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"Detections will be drawn in real-time on GPU {self.device}")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Apply rotation if specified
                if CAMERA_ROTATION != 0:
                    if CAMERA_ROTATION == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif CAMERA_ROTATION == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif CAMERA_ROTATION == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Preprocess frame
                frame = self.preprocess_frame(frame)
                
                # Run inference
                results = self.run_inference(frame)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, results)
                
                # Update and draw FPS
                self.update_fps()
                annotated_frame = self.draw_fps(annotated_frame)
                
                # Display the annotated frame
                cv2.imshow("YOLO Real-Time Detection", annotated_frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")


def main():
    """Main function to run YOLO inference."""
    parser = argparse.ArgumentParser(
        description="Real-time YOLO object detection with camera input"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Path to YOLO model weights (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--device-path",
        type=str,
        default=CAMERA_DEVICE_PATH,
        help=f"Linux device path for camera (e.g., /dev/video4) (default: {CAMERA_DEVICE_PATH})"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"Camera index if device-path is not used (default: {CAMERA_INDEX})"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
        help=f"IOU threshold (default: {IOU_THRESHOLD})"
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=DEVICE,
        help=f"GPU device index (default: {DEVICE})"
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=CAMERA_ROTATION,
        choices=[0, 90, 180, 270],
        help=f"Camera rotation in degrees (default: {CAMERA_ROTATION})"
    )
    
    args = parser.parse_args()
    
    # Determine if we should use device path or camera index
    camera_device_path = args.device_path if args.device_path else None
    
    # Create and run inference
    try:
        inference = YOLOInference(
            model_path=args.model,
            device=args.gpu_device,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        inference.run_camera_inference(camera_index=args.camera, device_path=camera_device_path)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure the model file exists at: {args.model}")
        exit(1)
    
    except RuntimeError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
