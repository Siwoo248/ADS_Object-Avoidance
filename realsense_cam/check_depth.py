import pyrealsense2 as rs
import cv2
import numpy as np

def main():
    """
    Real-time RealSense camera streaming with depth visualization.
    Shows RGB feed with distance measurement at center point.
    """
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get the device to check its name
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("Error: No RealSense device detected!")
        return
    
    try:
        device_name = devices[0].get_info(rs.camera_info.name)
        print(f"Device found: {device_name}")
    except RuntimeError as e:
        print(f"Device detected but cannot read name: {e}")
        print("Proceeding with streaming anyway...")
    
    # Enable depth and color streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Error starting pipeline: {e}")
        print("Make sure the RealSense camera is properly connected and not in use by another application.")
        return
    
    # Create an align object to align depth to color frame
    align = rs.align(rs.stream.color)
    
    # Get the depth scale for unit conversion
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    print("Streaming started. Press 'q' to exit.")
    
    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get frame dimensions
            height, width = color_image.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            # Get distance at center point
            distance = depth_frame.get_distance(center_x, center_y)
            distance_m = distance * depth_scale * 1000  # Convert to meters
            
            # Normalize depth image for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # Draw crosshair at center
            cv2.circle(color_image, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.line(color_image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
            cv2.line(color_image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
            
            # Add distance text on color image
            if distance > 0:
                distance_text = f"Distance: {distance_m:.3f} m"
                cv2.putText(color_image, distance_text, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(color_image, "Distance: Out of Range", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Combine images side by side
            images = np.hstack((color_image, depth_colormap))
            
            # Display the frame
            cv2.imshow("RealSense Color and Depth", images)
            
            # Print distance to console
            if distance > 0:
                print(f"Distance at center: {distance_m:.3f} meters", end="\r")
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nCamera streaming stopped.")

if __name__ == "__main__":
    main()

