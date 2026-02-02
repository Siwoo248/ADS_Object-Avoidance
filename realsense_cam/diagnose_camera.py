"""
Diagnostic script to test RealSense camera connection and troubleshoot issues.
Does not modify any main scripts.
"""

import pyrealsense2 as rs
import time
import sys

def test_device_detection():
    """Test if device is detected by context"""
    print("\n" + "="*60)
    print("STEP 1: Device Detection")
    print("="*60)
    
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("❌ FAILED: No devices found!")
            return False
        
        print(f"✓ Found {len(devices)} device(s)")
        
        # Skip reading device info as it may cause "bad optional access" error
        # Just confirm devices exist
        print("  (Device details skipped to avoid hang)")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_pipeline_creation():
    """Test if pipeline can be created"""
    print("\n" + "="*60)
    print("STEP 2: Pipeline Creation")
    print("="*60)
    
    try:
        pipeline = rs.pipeline()
        print("✓ Pipeline created successfully")
        return True, pipeline
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, None


def test_config_creation(pipeline):
    """Test if config can be created"""
    print("\n" + "="*60)
    print("STEP 3: Config Creation")
    print("="*60)
    
    try:
        config = rs.config()
        print("✓ Config created successfully")
        return True, config
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, None


def test_stream_configuration(config):
    """Test if streams can be configured"""
    print("\n" + "="*60)
    print("STEP 4: Stream Configuration")
    print("="*60)
    
    try:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        print("✓ Depth stream configured (640x480, z16, 30fps)")
        
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        print("✓ Color stream configured (640x480, bgr8, 30fps)")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_pipeline_start_with_retry(pipeline, config, max_retries=5):
    """Test if pipeline can start with retry logic"""
    print("\n" + "="*60)
    print("STEP 5: Pipeline Start (with retry)")
    print("="*60)
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\nAttempt {attempt}/{max_retries}...")
            profile = pipeline.start(config)
            print(f"✓ Pipeline started successfully on attempt {attempt}!")
            return True, profile, pipeline
        except RuntimeError as e:
            print(f"  ❌ Failed: {e}")
            
            if attempt < max_retries:
                wait_time = 2 * attempt  # Exponential backoff: 2s, 4s, 6s, 8s, 10s
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                
                # Reset pipeline
                try:
                    pipeline.stop()
                except:
                    pass
                
                pipeline = rs.pipeline()
                config_new = rs.config()
                config_new.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config_new.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                config = config_new
            else:
                print(f"\n❌ FAILED: Could not start pipeline after {max_retries} attempts")
                return False, None, pipeline
    
    return False, None, pipeline


def test_frame_capture(pipeline):
    """Test if frames can be captured"""
    print("\n" + "="*60)
    print("STEP 6: Frame Capture")
    print("="*60)
    
    try:
        print("Waiting for frames...")
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if depth_frame:
            print(f"✓ Depth frame received ({depth_frame.get_width()}x{depth_frame.get_height()})")
        else:
            print("❌ No depth frame")
            return False
        
        if color_frame:
            print(f"✓ Color frame received ({color_frame.get_width()}x{color_frame.get_height()})")
        else:
            print("❌ No color frame")
            return False
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    """Run all diagnostic tests"""
    print("\n" + "█"*60)
    print("  RealSense Camera Diagnostic Tool")
    print("█"*60)
    
    # Test 1: Device detection
    if not test_device_detection():
        print("\n❌ DIAGNOSIS: Camera not detected by pyrealsense2")
        print("   Solutions:")
        print("   1. Check USB connection")
        print("   2. Reinstall pyrealsense2: pip install --upgrade pyrealsense2")
        print("   3. Check /dev/video* devices: ls -la /dev/video*")
        sys.exit(1)
    
    # Test 2: Pipeline creation
    success, pipeline = test_pipeline_creation()
    if not success:
        print("\n❌ DIAGNOSIS: Cannot create pipeline")
        sys.exit(1)
    
    # Test 3: Config creation
    success, config = test_config_creation(pipeline)
    if not success:
        print("\n❌ DIAGNOSIS: Cannot create config")
        sys.exit(1)
    
    # Test 4: Stream configuration
    if not test_stream_configuration(config):
        print("\n❌ DIAGNOSIS: Cannot configure streams")
        sys.exit(1)
    
    # Test 5: Pipeline start with retry
    success, profile, pipeline = test_pipeline_start_with_retry(pipeline, config, max_retries=5)
    if not success:
        print("\n" + "█"*60)
        print("  DIAGNOSIS: Camera hardware issue")
        print("█"*60)
        print("\nThe camera is detected but won't respond to initialization.")
        print("\nPossible causes:")
        print("1. Camera firmware is unresponsive")
        print("2. USB connection is unstable (try different port)")
        print("3. Insufficient USB power (try powered USB hub)")
        print("4. Camera driver conflict")
        print("\nTry:")
        print("  1. Unplug camera, wait 30s, plug back in")
        print("  2. Try a different USB port")
        print("  3. Use a powered USB hub")
        print("  4. Restart your system")
        sys.exit(1)
    
    # Test 6: Frame capture
    if not test_frame_capture(pipeline):
        print("\n❌ DIAGNOSIS: Pipeline started but cannot receive frames")
        print("   This may be a firmware issue")
        sys.exit(1)
    
    pipeline.stop()
    
    # All tests passed
    print("\n" + "█"*60)
    print("  ✅ ALL TESTS PASSED - CAMERA IS WORKING!")
    print("█"*60)
    print("\nYour RealSense camera is functioning correctly.")
    print("You can now run: python yolo_depth_detection.py")
    

if __name__ == "__main__":
    main()
