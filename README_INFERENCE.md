# Real-Time YOLO Inference with Camera Input

This module provides real-time object detection inference using YOLOv8s model trained on 13 traffic-related classes.

## Features

- ✅ **Real-time Detection**: Process camera feed in real-time
- ✅ **GPU Only**: Optimized for GPU-based inference only
- ✅ **13 Classes**: Detects all trained traffic-related objects
- ✅ **Live Annotation**: Bounding boxes and confidence scores displayed in real-time
- ✅ **FPS Counter**: Real-time performance monitoring
- ✅ **Configurable**: Easy to adjust parameters via `config.py`

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support
- Camera device connected to the system

### Setup

```bash
# Navigate to the Yolo_object_detection folder
cd Yolo_object_detection

# Install required packages
pip install -r requirements.txt

# For GPU support, install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Classes Detected

The model detects the following 13 classes:

1. **car** - Vehicles on the road
2. **crosswalk** - Pedestrian crossing signs
3. **highway_entry** - Highway entrance signs
4. **highway_exit** - Highway exit signs
5. **no_entry** - No entry signs
6. **onewayroad** - One-way road signs
7. **parking** - Parking signs
8. **pedestrian** - Pedestrians
9. **priority** - Priority road signs
10. **roadblock** - Road block signs
11. **roundabout** - Roundabout signs
12. **stop** - Stop signs
13. **trafficlight** - Traffic lights

## Usage

### Basic Usage

Run real-time inference on the default camera configured in `config.py`:

```bash
python inference.py
```

### Using Linux Device Path (for /dev/videoX cameras)

If your camera is connected via a Linux device path (like `/dev/video4`):

```bash
# Using the device path configured in config.py
python inference.py

# Or specify the device path explicitly
python inference.py --device-path /dev/video4
```

### Using Camera Index

If you have a standard camera with index (0, 1, 2, etc.):

```bash
python inference.py --camera 0
```

### Advanced Usage

```bash
# Specify device path with other parameters
python inference.py --device-path /dev/video4 --conf 0.5 --iou 0.45 --gpu-device 0

# Use camera index instead of device path
python inference.py --camera 0 --conf 0.4

# Add camera rotation (useful for mounted cameras)
python inference.py --device-path /dev/video4 --rotation 90

# Specify GPU device
python inference.py --device-path /dev/video4 --gpu-device 0

# Custom model path
python inference.py --device-path /dev/video4 --model path/to/model.pt

# Combine all options
python inference.py --device-path /dev/video4 --conf 0.4 --iou 0.45 --gpu-device 0 --rotation 90
```

### Controls

- **Press 'q'**: Quit the application
- **Press 'Ctrl+C'**: Force exit

## Configuration

Edit `config.py` to customize the inference settings:

```python
# Model configuration
MODEL_PATH = "path/to/best.pt"

# Inference thresholds
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Camera settings
# Use CAMERA_DEVICE_PATH for Linux /dev/videoX paths (e.g., '/dev/video4')
# If set to a non-empty string, it takes precedence over CAMERA_INDEX
CAMERA_DEVICE_PATH = '/dev/video4'  # Linux device path (set to None to use CAMERA_INDEX)
CAMERA_INDEX = 0                      # Standard camera index (used if CAMERA_DEVICE_PATH is None)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
CAMERA_ROTATION = 0                   # Rotation in degrees (0, 90, 180, 270)

# GPU device (0 for first GPU, only GPU is supported)
DEVICE = 0

# Display options
DISPLAY_FPS = True
DISPLAY_CONFIDENCE = True
```

## Output

The inference script will:

1. Open a window showing the camera feed
2. Draw bounding boxes around detected objects
3. Display class names with confidence scores
4. Show real-time FPS counter in the top-left corner
5. Each class has a unique color for easy identification

### Color Scheme

- **Green**: car
- **Blue**: crosswalk
- **Orange**: highway_entry
- **Cyan**: highway_exit
- **Purple**: no_entry
- **Yellow**: onewayroad
- **Light Blue**: parking
- **Red-Orange**: pedestrian
- **Dark Cyan**: priority
- **Red**: roadblock
- **Magenta**: roundabout
- **White**: stop
- **Dark Green**: trafficlight

## Performance Notes

- Runs on GPU only for optimal performance
- Real-time processing requires GPU support
- FPS depends on GPU capabilities and image resolution
- Recommended for NVIDIA GPUs with CUDA support

## Troubleshooting

### Camera not opening - "Failed to open camera with index 0"

This error typically occurs when the camera cannot be accessed. Here are solutions based on your camera type:

#### For Linux Device Path Cameras (/dev/video4, /dev/video0, etc.)

If you're using a camera connected via Linux device path:

1. **Identify your camera device:**
   ```bash
   # List all video devices
   ls -la /dev/video*
   
   # Test camera access with v4l2-ctl
   v4l2-ctl --list-devices
   ```

2. **Update config.py:**
   ```python
   CAMERA_DEVICE_PATH = '/dev/video4'  # Change to your actual device
   CAMERA_INDEX = None  # Not used when CAMERA_DEVICE_PATH is set
   ```

3. **Run inference:**
   ```bash
   python inference.py
   ```

   Or specify the device path via command line:
   ```bash
   python inference.py --device-path /dev/video4
   ```

#### For Standard USB/Built-in Cameras

1. **Update config.py:**
   ```python
   CAMERA_DEVICE_PATH = None  # Not used
   CAMERA_INDEX = 0            # Try 0, 1, 2, etc.
   ```

2. **Run inference:**
   ```bash
   python inference.py --camera 0
   ```

#### General Troubleshooting

- Ensure camera is properly connected
- Check camera is not in use by another application:
  ```bash
  lsof /dev/video*  # Check if camera is being used
  ```
- Check camera permissions:
  ```bash
  # Add user to video group
  sudo usermod -a -G video $USER
  
  # You may need to logout and login for changes to take effect
  ```
- Try different camera indices/devices
- Check camera cable connection
- Restart the application

### Low FPS
- Reduce `FRAME_WIDTH` and `FRAME_HEIGHT` in config.py
- Increase `CONFIDENCE_THRESHOLD` to filter weak detections
- Check GPU utilization and ensure no other heavy processes

### Model loading error
- Verify the model path in `config.py` is correct
- Ensure the model file exists at the specified location
- Check that the model file is not corrupted

### CUDA/GPU errors
- Verify CUDA is installed and compatible with your GPU
- Ensure PyTorch is installed with CUDA support
- Check GPU availability using `nvidia-smi`

## File Structure

```
Yolo_object_detection/
├── inference.py              # Main inference script
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README_INFERENCE.md       # This file
├── BFMC.v1i.yolov8/
│   ├── data.yaml
│   ├── train.py
│   └── runs/
│       └── detect/
│           └── traffic_signs_full_ads/
│               └── weights/
│                   ├── best.pt          # Best model weights (used for inference)
│                   └── last.pt
└── ...
```

## Model Details

- **Model Architecture**: YOLOv8s (Small variant)
- **Input Size**: 640x640 pixels
- **Training Dataset**: BFMC.v1i.yolov8 (13,761 training images)
- **Best mAP@0.5**: 0.977
- **Best mAP@0.5:0.95**: 0.864

## Dependencies

- **ultralytics**: YOLOv8 framework
- **opencv-python**: Computer vision library for image processing
- **numpy**: Numerical computing library
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities for PyTorch

## License

This project uses the BFMC dataset licensed under CC BY 4.0.

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow BFMC Dataset](https://universe.roboflow.com/project/bfmc-ynbzk)
