import pyrealsense2 as rs

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get the device to check its name
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("Error: No RealSense device detected!")
        return
    
    # Safer device info access
    print(devices)
    device = devices[0]
    try:
        device_name = device.get_info(rs.camera_info.name)
        print(f"Device found: {device_name}")
    except RuntimeError:
        print("Device found but name unavailable")
        print(f"Device serial: {device.get_info(rs.camera_info.serial_number)}")

if __name__ == "__main__":
    main()