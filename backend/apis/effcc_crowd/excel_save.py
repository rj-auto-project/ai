import cv2
import torch
from ultralytics import YOLO
import pandas as pd
from datetime import datetime, timedelta
import signal
import sys
import base64
import io
from PIL import Image

# Define a function to save detection results to an Excel file
def save_results_to_excel(detection_results, output_path='detection_results.xlsx'):
    df = pd.DataFrame(detection_results)
    df.to_excel(output_path, index=False)
    print(f"Detection results saved to {output_path}")

# Signal handler to catch Ctrl+C (SIGINT)
def signal_handler(sig, frame):
    print('Interrupted! Saving detection results...')
    save_results_to_excel(detection_results)
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
id=int(input("Enter the camera id: "))
# Load the YOLOv8 model (assuming the model is already trained)
model = YOLO('best.pt')  # Replace 'best.pt' with the appropriate model file

# Define the video capture
input_video_path = 'testing1.mkv'
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare to save the results
detection_results = []
start_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Collect detection results with timestamp and class IDs
    current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    elapsed_time_sec = current_time_sec - start_time_sec
    timestamp = (datetime.now() + timedelta(seconds=elapsed_time_sec)).strftime('%H:%M:%S')

    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, class_id = result
        tensor_list = result.cpu().numpy().tolist()

        # Check for duplicates
        is_duplicate = False
        for detection in detection_results:
            if detection['timestamp'] == timestamp and detection['tensor'][-1] == class_id:
                is_duplicate = True
                break

        if not is_duplicate:
            detection_results.append({
                'timestamp': timestamp,
                'camera_id': id,
                'tensor': tensor_list
            })

# If the loop ends naturally, save the results
save_results_to_excel(detection_results)

# Release resources
cap.release()
cv2.destroyAllWindows()
