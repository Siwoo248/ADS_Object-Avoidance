import pyrealsense2 as rs

def main():
    """
    Diagnostic script to list all sensors and their capabilities on RealSense device.
    """
    # Create context
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found!")
        return
    
    device = devices[0]
    print(f"Device: {device.get_info(rs.camera_info.name)}")
    print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
    print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}\n")
    
    # List all sensors
    sensors = device.query_sensors()
    print(f"Found {len(sensors)} sensor(s):\n")
    
    for idx, sensor in enumerate(sensors):
        print(f"Sensor {idx}: {sensor.get_info(rs.camera_info.name)}")
        
        # List all stream profiles
        profiles = sensor.get_stream_profiles()
        print(f"  Available streams: {len(profiles)}")
        
        for profile in profiles:
            stream_type = profile.stream_type()
            stream_name = str(stream_type).split('.')[-1]
            print(f"    - {stream_name}")
        
        print()
    
    # Specifically check for IMU streams across all sensors
    print("\n" + "="*60)
    print("Checking for IMU streams (ACCEL/GYRO):")
    print("="*60)
    
    has_accel = False
    has_gyro = False
    
    for idx, sensor in enumerate(sensors):
        profiles = sensor.get_stream_profiles()
        for profile in profiles:
            if profile.stream_type() == rs.stream.accel:
                print(f"✓ Found ACCEL stream in sensor {idx}")
                has_accel = True
            if profile.stream_type() == rs.stream.gyro:
                print(f"✓ Found GYRO stream in sensor {idx}")
                has_gyro = True
    
    if not has_accel and not has_gyro:
        print("✗ No IMU sensors (ACCEL/GYRO) found")
        print("\nNote: The D455 may not have an integrated IMU.")
        print("Check if your device is a D455 or if IMU firmware needs to be updated.")
    else:
        if has_accel:
            print("✓ Accelerometer available")
        if has_gyro:
            print("✓ Gyroscope available")


if __name__ == "__main__":
    main()
