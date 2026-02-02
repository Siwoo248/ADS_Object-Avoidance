# RealSense YOLO Depth Detection

This directory contains tools for real-time object detection with depth measurement using Intel RealSense cameras and YOLO object detection.

## Files

### `check_depth.py`
Basic RealSense depth camera streaming with distance measurement at the center point.

**Features:**
- Real-time depth and color frame streaming
- Distance measurement at frame center
- Visual crosshair at measurement point
- Side-by-side color + depth map visualization

**Usage:**
```bash
python check_depth.py
```

### `yolo_depth_detection.py`
**Advanced real-time YOLO object detection with RealSense depth measurement.**

**Features:**
- ✅ Real-time YOLO object detection using trained traffic sign model
- ✅ Depth measurement at the center of each detected bounding box
- ✅ Distance visualization on each detection (in meters)
- ✅ Color-coded detection boxes by class
- ✅ Real-time FPS display
- ✅ Confidence score display
- ✅ Console output with detected objects and distances
- ✅ Green indicator for valid distances, red for out-of-range

**Usage:**
```bash
python yolo_depth_detection.py
```

**Press 'q' to exit the application.**

## Installation

1. Navigate to the realsense_cam directory:
```bash
cd /home/siwoo/ADS-Skynet/realsense_cam
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

The `requirements.txt` includes:
- **pyrealsense2**: Intel RealSense SDK for Python
- **opencv-python**: Computer vision processing and visualization
- **numpy**: Numerical operations
- **ultralytics**: YOLO object detection framework
- **torch** & **torchvision**: Deep learning framework (required by YOLO)

## Configuration

The YOLO model and detection parameters are configured in the parent directory's `Yolo_object_detection/config.py`:

Key settings:
- `MODEL_PATH`: Path to the trained YOLO model weights
- `CLASS_NAMES`: 13 traffic sign classes (car, stop, traffic light, etc.)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections (default: 0.25)
- `IOU_THRESHOLD`: NMS threshold for duplicate detections (default: 0.45)
- `DEVICE`: GPU device index (0 for first GPU)

## How It Works

### YOLO Depth Detection Pipeline:

1. **Capture Frames**: RealSense camera provides color and depth frames
2. **Align Frames**: Depth frames are aligned to color space for accurate coordinate mapping
3. **YOLO Inference**: Color frame is processed by trained YOLO model
4. **Bounding Boxes**: For each detected object:
   - Calculate center point coordinates
   - Query depth frame at center point
   - Get distance value (with depth scale conversion)
5. **Visualization**: Draw boxes, labels, distances on the frame
6. **Display**: Show annotated frame in real-time

### Distance Measurement:

- Distance is measured at the **center of each bounding box**
- Distance is displayed in **meters** below each detection
- **Green text** = valid distance measurement
- **Red text** = out of range or invalid measurement

## Example Output

When running `yolo_depth_detection.py`:
```
Console Output:
Detections: car: 1.23m, stop: 0.85m, traffic light: 2.15m

Visual Display:
- Real-time video feed with bounding boxes
- Class names and confidence scores
- Distance measurements at each detection
- FPS counter in top-left corner
```

## Troubleshooting

### RealSense Camera Not Found
```bash
# Check connected devices
lsusb | grep RealSense

# Check /dev/video devices
ls -la /dev/video*
```

### YOLO Model Path Issues
Ensure the model path in `Yolo_object_detection/config.py` points to the correct location:
```python
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "BFMC.v1i.yolov8/runs/detect/traffic_signs_full_ads/weights/best.pt"
)
```

### GPU/CUDA Issues
If GPU inference fails, the model will fall back to CPU. To force CPU:
- Modify `DEVICE` in `config.py` to `-1`

### Import Errors
Ensure the parent directory structure is correct:
```
ADS-Skynet/
├── realsense_cam/
│   ├── check_depth.py
│   ├── yolo_depth_detection.py
│   └── requirements.txt
└── Yolo_object_detection/
    ├── config.py
    ├── inference.py
    └── ...
```

## System Requirements

- **Python**: 3.8+
- **OS**: Linux (tested on Ubuntu 20.04+)
- **Hardware**: 
  - Intel RealSense D435/D455 camera (or compatible)
  - NVIDIA GPU with CUDA support (recommended for real-time performance)
  - Minimum 4GB RAM

## Performance Notes

- Real-time FPS depends on GPU capability
- YOLO inference is GPU-accelerated for better performance
- Depth frame alignment adds minimal overhead
- Adjust `CONFIDENCE_THRESHOLD` for faster detection (fewer results)
