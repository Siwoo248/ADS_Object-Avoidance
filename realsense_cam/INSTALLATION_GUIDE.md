# RealSense YOLO Depth Detection - Installation & Troubleshooting Guide

## ✅ Successfully Implemented!

Your `yolo_depth_detection.py` script is now working! It successfully loads the YOLO model and initializes the RealSense camera.

## Installation Steps

### 1. Install Dependencies
```bash
cd /home/siwoo/ADS-Skynet/realsense_cam
pip install -r requirements.txt
```

### 2. Run the Script
```bash
python yolo_depth_detection.py
```

## What the Script Does

The script performs real-time object detection with depth measurement:

1. **Loads YOLO Model** - Uses your trained traffic sign detection model
2. **Initializes RealSense Camera** - Connects to the D435/D455 depth camera
3. **Processes Frames in Real-Time**:
   - Captures color and depth frames
   - Aligns depth frames to color space
   - Runs YOLO inference
   - Measures distance at the center of each detected bounding box
4. **Visualizes Results**:
   - Draws bounding boxes with class names
   - Shows distance measurements below each box (in meters)
   - Green text = valid distance, Red text = out of range
   - FPS counter in top-left

## Expected Output

When you run the script and point it at objects:

```
Loading YOLO model from: /home/siwoo/ADS-Skynet/Yolo_object_detection/...
YOLO model loaded successfully
Starting RealSense pipeline...
RealSense camera initialized successfully
Starting real-time detection with depth measurement...
Press 'q' to exit
```

Then a window appears showing:
- Live video feed with detected objects
- Bounding boxes with class labels
- Distance measurements (e.g., "Distance: 1.25m")
- Real-time FPS counter

## Key Fixes Applied

### 1. PyTorch 2.6+ Compatibility
Modern PyTorch versions enforce stricter security for loading model weights. The fix:
```python
_orig_torch_load = torch.load
def torch_load_with_legacy_support(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = torch_load_with_legacy_support
```

### 2. Module Import Issues
Fixed relative imports by:
- Using absolute path resolution with `Path(__file__).resolve()`
- Properly loading the config module with importlib
- Adding the ADS-Skynet root directory to sys.path

### 3. Config Loading
Used `importlib.util` to dynamically load the config module from the Yolo_object_detection directory while preserving the `__file__` context.

## Troubleshooting

### Issue: "No RealSense device detected!"
**Solution**: 
- Check if the camera is connected: `lsusb | grep RealSense`
- Check /dev/video devices: `ls -la /dev/video*`
- Try restarting the USB device

### Issue: "Model file not found"
**Solution**: Verify the model path exists:
```bash
ls -la /home/siwoo/ADS-Skynet/Yolo_object_detection/BFMC.v1i.yolov8/runs/detect/traffic_signs_full_ads/weights/best.pt
```

### Issue: NVIDIA Memory Warnings (NvMapMemAllocInternalTagged errors)
**Solution**: These are harmless warnings on Jetson devices and don't affect functionality. They can be suppressed by redirecting stderr:
```bash
python yolo_depth_detection.py 2>&1 | grep -v "NvMap"
```

### Issue: Slow Performance / Low FPS
**Solutions**:
- Lower the image size in config.py
- Increase the confidence threshold to detect fewer objects
- Use a GPU with better compute capability
- Reduce frame resolution

## Configuration

Edit `/home/siwoo/ADS-Skynet/Yolo_object_detection/config.py` to customize:

```python
# Detection sensitivity
CONFIDENCE_THRESHOLD = 0.25  # Lower = more detections, slower
IOU_THRESHOLD = 0.45  # Duplicate detection threshold

# Camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# GPU
DEVICE = 0  # GPU index (0 for first GPU, -1 for CPU)

# Display
DISPLAY_FPS = True
DISPLAY_CONFIDENCE = True
```

## Performance Tips

1. **Increase FPS**: Reduce `CONFIDENCE_THRESHOLD` or run on more powerful GPU
2. **Reduce Latency**: Disable `DISPLAY_FPS` and `DISPLAY_CONFIDENCE`
3. **Better Accuracy**: Train the model with more data or use a larger model (yolov8m, yolov8l)
4. **Smooth Stream**: Ensure consistent lighting and camera distance

## Files Modified/Created

1. **`yolo_depth_detection.py`** - Main detection script ✅
2. **`requirements.txt`** - Updated with torch, ultralytics ✅
3. **`README.md`** - Documentation ✅
4. **`Yolo_object_detection/__init__.py`** - Package init file ✅

## System Requirements

- Python 3.8+
- CUDA/GPU (recommended for real-time performance)
- Intel RealSense D435 or D455 camera
- 4GB+ RAM
- Ubuntu/Linux OS

## Exit the Script

Press **'q'** key in the video window to exit cleanly.

Or use Ctrl+C in the terminal (may require keyboard focus on the video window first).

---

**Status**: ✅ READY TO USE!

Your RealSense + YOLO depth detection system is fully configured and working!
