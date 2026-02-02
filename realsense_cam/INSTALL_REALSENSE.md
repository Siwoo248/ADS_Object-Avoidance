# RealSense SDK Fix - Quick Start Guide

## The Problem
Your RealSense SDK installation is broken:
- ❌ C++ librealsense backend: **Missing/corrupted**
- ✅ pyrealsense2 Python bindings: Installed but not working
- ✅ Camera firmware: Responding but SDK can't communicate

**Result:** `RuntimeError: bad optional access` and `No device connected`

## The Solution
Reinstall the C++ backend with a **minimal, depth-only** build optimized for Jetson.

## Instructions

### 1️⃣ Prerequisites (if you don't have librealsense source)

If you get an error about "librealsense source not found", clone it first:

```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
```

Then skip to Step 2 below.

### 2️⃣ Run the Installation Script

```bash
cd /home/siwoo/ADS-Skynet/realsense_cam
bash install_realsense.sh
```

**What it does:**
- Removes broken RealSense installations
- Rebuilds librealsense C++ (minimal, optimized)
- Verifies the installation
- Tests Python bindings

**Time estimate:** 15-25 minutes (depends on Jetson board)

### 3️⃣ After Installation Completes

**Physically replug your camera** (unplug USB, wait 5 seconds, plug back in)

### 4️⃣ Verify Everything Works

Run the diagnostic:
```bash
cd /home/siwoo/ADS-Skynet/realsense_cam
python diagnose_camera.py
```

**Expected output:**
```
✓ Found 1 device(s)
✓ Pipeline created successfully
✓ Depth stream configured
✓ Color stream configured
✓ Pipeline started successfully
✓ Frames received
✅ ALL TESTS PASSED - CAMERA IS WORKING!
```

### 5️⃣ Run Your YOLO Script

Once diagnostic passes:
```bash
python yolo_depth_detection.py
```

---

## Troubleshooting

### Build failed with "CMake error"
- Check you have `cmake` installed: `apt-get install cmake`
- Check you have `build-essential`: `apt-get install build-essential`

### Python import error after script
- The script installs to system directories, restart terminal or shell:
  ```bash
  exec bash
  ```

### Still getting "No device connected"
1. Try different USB port
2. Use a powered USB hub
3. Run: `lsusb | grep Intel` to verify camera is detected

### Script needs to restart?
- The script handles sudo automatically
- Just run it and follow prompts

---

## What Changed

**Before (broken):**
- librealsense C++ ❌ (deleted)
- pyrealsense2 ✅ (orphaned)
- Result: SDK crash

**After (fixed):**
- librealsense C++ ✅ (minimal, rebuilt)
- pyrealsense2 ✅ (working)
- Result: Full depth + color streaming

---

## Important Notes

- **No IMU:** This build intentionally excludes IMU support (your D455 doesn't have it anyway)
- **Depth-only mode:** Optimized for YOLO + depth detection
- **Python-focused:** Built specifically for pyrealsense2
- **Minimal dependencies:** No extra visualization tools or examples

---

## If You Need Help

Check the diagnostic output:
```bash
python diagnose_camera.py
```

It will tell you exactly which step failed and why.

---

**Good luck! 🎯**
