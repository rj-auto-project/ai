# import cv2
# from ultralytics import YOLO
# import subprocess

# # Load the YOLOv8 model
# model = YOLO('/home/annone/ai/models/yolov8n.pt')

# # Define the video capture
# input_video_path = 0  # 0 for webcam

# # Open the video file or webcam
# cap = cv2.VideoCapture(input_video_path)

# # RTSP URL of your MediaMTX server
# rtsp_url = 'rtsp://localhost:8554/stream'  # Update this URL as needed

# # Construct the FFmpeg command
# ffmpeg_cmd = [
#     'ffmpeg',
#     '-y',  # Overwrite output files without asking
#     '-f', 'rawvideo',  # Input format
#     '-pix_fmt', 'bgr24',  # Pixel format
#     '-s', '640x480',  # Video resolution (adjust as needed)
#     '-r', '25',  # Frame rate
#     '-i', '-',  # Input from stdin
#     '-c:v', 'libx264',  # Video codec
#     '-preset', 'ultrafast',  # Encoding speed
#     '-tune', 'zerolatency',  # Tune for low latency
#     '-f', 'rtsp',  # Output format
#     rtsp_url  # RTSP output URL
# ]

# # Start the FFmpeg process
# process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform object detection
#     results = model(frame)

#     # Draw bounding boxes and labels on the frame
#     annotated_frame = results[0].plot()

#     # Write the frame to FFmpeg's stdin
#     process.stdin.write(annotated_frame.tobytes())

# # Clean up
# cap.release()
# process.stdin.close()
# process.wait()



import cv2

# RTSP link
rtsp_url = "rtsp://localhost:8554/stream"  # Replace with your RTSP URL

# Open the video stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the video stream opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video stream")
    exit()

# Read and display frames in a loop
while True:
    ret, frame = cap.read()
    cv2.resize(frame, (640,480))

    if not ret:
        print("Error: Unable to read frame from stream")
        break

    # Display the frame
    cv2.imshow('RTSP Stream', frame)

    # Exit the stream on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
