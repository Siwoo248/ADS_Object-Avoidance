"""
Configuration settings for YOLO real-time inference with camera input.
"""

import os

# Model configuration
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "yolov11/results/runs/detect/train/weights/best.pt"
)

# Class names (13 classes from BFMC dataset)
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

# Inference settings
CONFIDENCE_THRESHOLD = 0.60  # Confidence threshold for detections
IOU_THRESHOLD = 0.45  # IOU threshold for NMS
IMAGE_SIZE = 640  # YOLOv8 input image size

# Camera settings
# Use CAMERA_DEVICE_PATH for Linux /dev/videoX paths (e.g., '/dev/video4')
# Use CAMERA_INDEX for standard camera indexing (0, 1, 2, etc.)
# If both are set, CAMERA_DEVICE_PATH takes precedence
CAMERA_DEVICE_PATH = '/dev/video4'  # Linux device path for camera (set to None to use CAMERA_INDEX instead)
CAMERA_INDEX = 0  # Default camera index (0 for primary camera) - used if CAMERA_DEVICE_PATH is None
FRAME_WIDTH = 640  # Camera frame width
FRAME_HEIGHT = 480  # Camera frame height
FPS = 30  # Target FPS
CAMERA_ROTATION = 0  # Rotation in degrees (0, 90, 180, 270)

# Device configuration
DEVICE = 0  # GPU device (0 for first GPU, -1 for CPU)

# Display settings
DISPLAY_FPS = True  # Show FPS on frame
DISPLAY_CONFIDENCE = True  # Show confidence scores
LINE_THICKNESS = 2  # Thickness of bounding box lines
FONT_SCALE = 0.6  # Font scale for text
