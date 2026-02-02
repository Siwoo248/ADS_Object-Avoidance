#!/usr/bin/env python3
import pyrealsense2 as rs
import cv2
import numpy as np
import time

print("Testing RealSense camera connection...")

# Try multiple times
for attempt in range(3):
    print(f"\nAttempt {attempt + 1}/3...")
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable streams with lower resolution first
        config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 15)
        
        print("Starting pipeline...")
        profile = pipeline.start(config)
        
        print("✓ Pipeline started successfully!")
        
        # Try to get a frame
        print("Waiting for frames...")
        for i in range(5):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if depth_frame and color_frame:
                print(f"  Frame {i+1}: Depth size: {depth_frame.get_width()}x{depth_frame.get_height()}, Color size: {color_frame.get_width()}x{color_frame.get_height()}")
        
        print("\n✓ SUCCESS! Camera is working!")
        pipeline.stop()
        break
        
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        time.sleep(1)
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        time.sleep(1)
else:
    print("\n✗ Failed to connect after 3 attempts")
