#!/usr/bin/env python3
import pyrealsense2 as rs
import time

print("Initializing RealSense context...")

ctx = rs.context()
devices = ctx.query_devices()

print(f"Found {len(devices)} device(s)")

if len(devices) == 0:
    print("ERROR: No devices found!")
    exit(1)

device = devices[0]
print(f"Using device: {device}")

# Try to get device info without .get_info()
try:
    pipeline = rs.pipeline()
    print("Pipeline created")
    
    config = rs.config()
    print("Config created")
    
    # Don't specify resolution - let device choose
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)
    print("Streams enabled in config")
    
    print("Starting pipeline...")
    profile = pipeline.start(config)
    
    print("✓ SUCCESS! Pipeline started!")
    print(f"Depth stream: {profile.get_stream(rs.stream.depth)}")
    print(f"Color stream: {profile.get_stream(rs.stream.color)}")
    
    # Get first frame
    print("Waiting for first frame...")
    frames = pipeline.wait_for_frames()
    print(f"✓ Got frame!")
    
    pipeline.stop()
    print("Pipeline stopped")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
