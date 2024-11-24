import cv2
from datetime import datetime
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import numpy as np

app = FastAPI()

def extract_frame(video_path, timestamp_str, video_start_time, output_image_path=None):
    # Parse the timestamp string
    print(video_path)
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

    # Calculate the total seconds from the start of the video
    time_difference = timestamp - video_start_time
    time_in_seconds = time_difference.total_seconds()

    # Load the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return None

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame number to capture
    frame_number = int(time_in_seconds * fps)

    # Set the video to the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    if ret:
        print(f"Frame at {timestamp_str} saved as ")
        cap.release()
        return frame
    else:
        print(f"Failed to capture frame at {timestamp_str}")
        cap.release()
        return None
