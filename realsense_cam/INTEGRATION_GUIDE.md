# YOLO Obstacle Avoidance + LKAS Integration Guide

## Overview

This integration combines three systems for autonomous driving:
1. **LKAS (Lane Keeping Assist System)** - Lane detection and steering control
2. **YOLO Object Detection** - Real-time object detection with depth sensing
3. **Obstacle Avoidance** - Decision-making and maneuver execution

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐
│   Vehicle   │────▶│   LKAS   │────▶│  Viewer  │◀────│  User   │
│  (Camera)   │     │ (Detect  │     │  (Web)   │     │(Browser)│
└─────────────┘     │ +Decide) │     └──────────┘     └─────────┘
                    └──────────┘
                         │
                         │ Lane Detection
                         ▼
                    ┌──────────┐
                    │ Avoidance│
                    │  (YOLO + │
                    │  Depth)  │
                    └──────────┘
                         │
                         │ Combined Steering
                         ▼
                    ┌──────────┐
                    │ JetRacer │
                    │  Motors  │
                    └──────────┘
```

## How It Works

### Steering Combination Logic

The system intelligently combines LKAS steering with obstacle avoidance:

1. **NORMAL/MONITOR** - No obstacles detected
   - Uses LKAS steering only (pure lane keeping)

2. **MICRO_ADJUST** - Small obstacle in outer zone
   - Adds avoidance bias to LKAS steering
   - Stays in lane while avoiding obstacle

3. **LANE_CHANGE** - Large obstacle or obstacle in center
   - Full override: uses avoidance steering only
   - Executes timer-based overtaking maneuver

### Lane Boundary Detection

- **With LKAS**: Lane boundaries are updated dynamically from lane detection
- **Without LKAS**: Uses fixed calibrated boundaries (180px, 460px)

## Running the System

### Prerequisites

Make sure all dependencies are installed:
```bash
# In Yolo_object_detection/realsense_cam/
pip install -r requirements.txt

# LKAS should be installed from /home/siwoo/ads-skynet/lkas
cd /home/siwoo/ads-skynet/lkas
pip install -e .

# Common should be installed
cd /home/siwoo/ads-skynet/common
pip install -e .
```

### 4-Terminal Setup (Full System)

#### Terminal 1: Vehicle (Camera + Motor Control)
```bash
cd /home/siwoo/ads-skynet/vehicle
python -m vehicle.main
```

#### Terminal 2: LKAS (Lane Detection + Decision)
```bash
cd /home/siwoo/ads-skynet/lkas
python -m lkas.run
```

#### Terminal 3: Viewer (Web Interface)
```bash
cd /home/siwoo/ads-skynet/viewer
python -m viewer.run
```
Then open browser to: `http://<jetson-ip>:8000`

#### Terminal 4: Obstacle Avoidance (YOLO + Depth)
```bash
cd /home/siwoo/ads-skynet/Yolo_object_detection/realsense_cam

# Testing mode (no motors, LKAS integration enabled)
python yolo_depth_avoidance.py

# Full autonomous mode (motors enabled)
python yolo_depth_avoidance.py --motors

# Standalone mode (no LKAS integration)
python yolo_depth_avoidance.py --no-lkas

# Standalone + no viewer broadcast
python yolo_depth_avoidance.py --no-lkas --no-zmq
```

### Command Line Options

```bash
python yolo_depth_avoidance.py [OPTIONS]

Options:
  --motors          Enable motor control (default: disabled for safety)
  --no-lkas         Disable LKAS integration (standalone mode)
  --no-zmq          Disable ZMQ broadcasting to viewer
  --conf FLOAT      Confidence threshold (default: 0.5)
  --iou FLOAT       IOU threshold (default: 0.4)
```

## Keyboard Controls

When running `yolo_depth_avoidance.py`:

- **q** - Quit and stop all motors
- **r** - Reset avoidance system state
- **p** - Pause/Resume (emergency stop/resume)

From the web viewer:
- **Stop/Resume** buttons - Control all systems

## Safety Features

### Multiple Safety Layers

1. **Default Disabled Motors** - Motors are OFF by default, must use `--motors` flag
2. **Emergency Stop** - Press 'q' or 'p' to immediately stop
3. **Pause State** - System respects pause commands from viewer
4. **Startup Warning** - 3-second countdown before motors activate

### Failsafe Behavior

- If LKAS connection lost → Falls back to obstacle avoidance only
- If no lanes detected → Uses fixed lane boundaries
- If camera fails → System stops motors automatically
- On any exception → Motors are set to 0.0 in cleanup

## Integration Modes

### Mode 1: Full Integration (Recommended)
```bash
python yolo_depth_avoidance.py --motors
```
- LKAS lane keeping + obstacle avoidance
- ZMQ broadcasting to viewer
- Full autonomous operation

### Mode 2: Testing (Safe)
```bash
python yolo_depth_avoidance.py
```
- All features enabled except motors
- Perfect for testing detection and decisions
- View output in OpenCV window

### Mode 3: Standalone Avoidance
```bash
python yolo_depth_avoidance.py --motors --no-lkas
```
- Obstacle avoidance only, no lane keeping
- Uses fixed lane boundaries
- Still broadcasts to viewer

### Mode 4: Completely Standalone
```bash
python yolo_depth_avoidance.py --no-lkas --no-zmq
```
- No LKAS, no viewer integration
- Obstacle detection and display only
- Useful for debugging YOLO/depth

## Troubleshooting

### "LKAS not available"
- Make sure LKAS is installed: `cd /home/siwoo/ads-skynet/lkas && pip install -e .`
- Check that LKAS servers are running in Terminal 2

### "Common communication modules not available"
- Install common: `cd /home/siwoo/ads-skynet/common && pip install -e .`
- Make sure viewer is running in Terminal 3

### "Failed to start RealSense pipeline"
- Check camera connection: `realsense-viewer`
- Try unplugging and replugging the camera
- Check USB bandwidth: Use USB 3.0 port

### Motors not responding
- Did you use `--motors` flag?
- Check JetRacer hardware connections
- Verify NVIDIA racecar library is installed

### Lane boundaries incorrect
- **With LKAS**: Check LKAS detection quality in viewer
- **Without LKAS**: Calibrate values in `obstacle_avoidance.py`:
  ```python
  self.LEFT_LANE_X = 180   # Adjust based on your camera
  self.RIGHT_LANE_X = 460  # Adjust based on your camera
  ```

## Performance Tips

1. **Reduce inference skip**: Edit `inference_skip = 2` for faster detection
2. **Lower resolution**: Reduce camera resolution for better FPS
3. **Use GPU**: Make sure CUDA is available for YOLO inference
4. **Optimize LKAS**: Tune LKAS detection parameters in viewer

## Next Steps

1. **Calibrate lane boundaries** - Drive and observe lane positions in pixels
2. **Tune avoidance thresholds** - Adjust `CRITICAL_DISTANCE`, etc. in `obstacle_avoidance.py`
3. **Test maneuvers** - Start with `--no-motors` mode first
4. **Integrate with MPC controller** - Connect to LKAS MPC for smoother steering

## Technical Details

### Shared Memory Communication

- **camera_feed** - Vehicle → LKAS (images)
- **detection_results** - LKAS → Avoidance (lane detection)
- **control_commands** - LKAS → Vehicle (steering commands)

The avoidance system reads `detection_results` to get lane boundaries and reads `control_commands` to get LKAS steering, then combines with its own avoidance steering.

### ZMQ Ports

- **5562** - Vehicle status publishing (to LKAS broker)
- **5561** - Action commands (from viewer via LKAS broker)
- **5560** - Parameter updates (from viewer via LKAS broker)

All communication goes through the LKAS broker for centralized management.

## Credits

- LKAS system based on ads-skynet architecture
- YOLO integration using Ultralytics
- Obstacle avoidance with RealSense depth sensing
- Combined by Claude Code for Jetson Nano autonomous driving