import pyrealsense2 as rs
import time

def main():
    """
    Real-time RealSense D455 IMU sensor streaming.
    Displays accelerometer and gyroscope data in real-time.
    """
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get the device to check its name
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("Error: No RealSense device detected!")
        return
    
    device = devices[0]
    print(f"Device found: {device.get_info(rs.camera_info.name)}")
    
    # Check if device has IMU sensors
    imu_sensors = []
    for sensor in device.query_sensors():
        if rs.stream.accel in sensor.get_stream_profiles() or \
           rs.stream.gyro in sensor.get_stream_profiles():
            imu_sensors.append(sensor)
    
    if not imu_sensors:
        print("Error: No IMU sensor found on this device!")
        print("Note: D455 should have an IMU with accelerometer and gyroscope.")
        return
    
    print(f"Found {len(imu_sensors)} IMU sensor(s)")
    
    # Enable IMU streams (accelerometer and gyroscope)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
    
    # Start streaming
    print("Starting IMU streaming...")
    profile = pipeline.start(config)
    
    print("IMU streaming started. Press Ctrl+C to exit.\n")
    
    # Data storage for averaging
    accel_data = [0, 0, 0]
    gyro_data = [0, 0, 0]
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            
            # Get accelerometer frame
            if frames.get_accel_frame():
                accel_frame = frames.get_accel_frame()
                accel_data = accel_frame.as_motion_frame().get_motion_data()
            
            # Get gyroscope frame
            if frames.get_gyro_frame():
                gyro_frame = frames.get_gyro_frame()
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            
            # Get timestamps
            accel_timestamp = frames.get_accel_frame().get_timestamp() if frames.get_accel_frame() else 0
            gyro_timestamp = frames.get_gyro_frame().get_timestamp() if frames.get_gyro_frame() else 0
            
            # Calculate magnitude
            accel_magnitude = (accel_data.x**2 + accel_data.y**2 + accel_data.z**2)**0.5
            gyro_magnitude = (gyro_data.x**2 + gyro_data.y**2 + gyro_data.z**2)**0.5
            
            # Display data
            print(f"\n{'='*70}")
            print(f"Timestamp: {time.time() - start_time:.3f}s")
            print(f"{'='*70}")
            
            print(f"\nAccelerometer (m/s²):")
            print(f"  X: {accel_data.x:8.4f}  |  Y: {accel_data.y:8.4f}  |  Z: {accel_data.z:8.4f}")
            print(f"  Magnitude: {accel_magnitude:.4f} m/s²")
            
            print(f"\nGyroscope (rad/s):")
            print(f"  X: {gyro_data.x:8.4f}  |  Y: {gyro_data.y:8.4f}  |  Z: {gyro_data.z:8.4f}")
            print(f"  Magnitude: {gyro_magnitude:.4f} rad/s")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        pipeline.stop()
        elapsed_time = time.time() - start_time
        print(f"\nIMU streaming stopped.")
        print(f"Total frames: {frame_count}")
        print(f"Duration: {elapsed_time:.2f}s")
        if frame_count > 0:
            print(f"Average FPS: {frame_count/elapsed_time:.2f}")


if __name__ == "__main__":
    main()
