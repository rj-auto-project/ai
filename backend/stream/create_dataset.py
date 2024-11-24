import time 
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort.sort import Sort
import time
# from module import generate_custom_string
from collections import defaultdict
import math
import redis
from function import ViolationDetector  # Import your class from the first file
import subprocess
import os
# Initialize the ViolationDetector instance
var = ViolationDetector()




# Initialize device and model
PARENT_DIR = "/home/annone/ai"
r = redis.Redis(host='localhost', port=6379, db=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("/home/annone/ai/models/objseg50e.pt")
model.to(device)
print(f"{device} as Computation Device initiated")
tracker = Sort()

# CONSTANTS PER CAMERA
width, height = 640,480
camera_ip = "198.78.45.89"
camera_id = 2
fps = 30
class_list = ['auto', 'bike-rider', 'bolero', 'bus', 'car', 'hatchback', 'jcb', 'motorbike-rider', 'omni', 'pickup',
              'scooty-rider', 'scorpio', 'sedan', 'suv', 'swift', 'thar', 'tractor', 'truck', 'van']
previous_positions = defaultdict(lambda: {"x": 0, "y": 0, "time": 0})
null_mask = np.zeros((height, width), dtype=np.uint8)


track_ids_inframe = {}
custom_track_ids = {}
tracks_left_frame = []
known_track_ids = []

save_classes = ["scooty", "car", "woman", "thar", "bikerider", "scorpio", "bolero", "suv",
    "tractor", "truck", "van", "omni", "child", "hatchback", "sedan",
    "scootyrider", "bus", "apache", "bullet", "jcb", "motorbike-rider", "person",
    "pulsar", "swift", "licenseplate"]
save_every_nth = 20  # Save every 5th occurrence

# Initialize directories for cropped images
for cls in save_classes:
    os.makedirs(f"./cropped_images/{cls}", exist_ok=True)

# STREAMING CONSTANTS
rtsp_url = 'rtsp://localhost:8554/stream1'  # Update this URL as needed
orig_w,orig_h,resized_w,  resized_h = 1280,920,640,480
# Construct the FFmpeg command
ffmpeg_cmd = [
    'ffmpeg',
    '-y',  # Overwrite output files without asking
    '-f', 'rawvideo',  # Input format
    '-pix_fmt', 'bgr24',  # Pixel format
    '-s', '640x480',  # Video resolution (adjust as needed)
    '-r', '25',  # Frame rate
    '-i', '-',  # Input from stdin
    '-c:v', 'libx264',  # Video codec
    '-preset', 'ultrafast',  # Encoding speed
    '-tune', 'zerolatency',  # Tune for low latency
    '-f', 'rtsp',  # Output format
    rtsp_url  # RTSP output URL
]
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
frame_counts = defaultdict(int)

# Function to calculate distance in pixels
def calculate_pixel_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to calculate speed (assuming pixel distance and time interval)
def calculate_speed(pixel_distance, time_interval):
    return pixel_distance / time_interval  # Speed in pixels per second
    
# Function to process frames in batches
def process_frame_batch(frames):
    resized_frames = [cv2.resize(frame, (640 // 32 * 32, 480 // 32 * 32)) for frame in frames]
    frames_tensor = torch.from_numpy(np.stack(resized_frames)).permute(0, 3, 1, 2).float().to(device) / 255.0

    with torch.no_grad():
        batch_results = model(frames_tensor, device=device)

    return batch_results, resized_frames

import uuid

# Function to generate a custom track ID based on YOLO class, confidence, and a unique UUID
def generate_custom_track_id(label, confidence):
    return f"{label}_{confidence:.2f}_{uuid.uuid4()}"

# Function to track objects and draw segmentation polygons
def track_objects_and_save(frames, batch_results, frame_time):
    global camera_ip, previous_positions, fps, camera_id, track_ids_inframe, custom_track_ids, known_track_ids, roi_points, tracks_left_frame,orig_w,resized_w, orig_h, resized_h

    tracked_frames = []
    current_track_ids = []

    for frame, result in zip(frames, batch_results):
        detections = []
        scale_x, scale_y = orig_w / resized_w, orig_h / resized_h
        # scale_x, scale_y = ori, 2
        print(orig_w,resized_w, orig_h, resized_h)
        
        if result.masks:
            for mask, box in zip(result.masks.xy, result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                score = box.conf[0].item()
                label = model.names[int(box.cls[0])]
                detections.append([x1, y1, x2, y2, score, int(box.cls[0])])

        tracks = tracker.update(np.array(detections))

        for i, track in enumerate(tracks):
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            label = model.names[int(track[5])]
            frame_counts[track_id] += 1  # Increment frame count for the tracked object

            # If the object is in the save_classes list and it's the nth occurrence, save the cropped image
            if label in save_classes and frame_counts[track_id] % save_every_nth == 0:
                cropped_image = frame[y1:y2, x1:x2]
                image_save_path = f"./cropped_images/{label}/{track_id}_{frame_counts[track_id]}.jpg"
                cv2.imwrite(image_save_path, cropped_image)
                print(f"Saved cropped image for {label} with track ID {track_id} at {image_save_path}")

            # Append the current frame to the tracked frames
            tracked_frames.append(frame)

            # Add current track ID to the list of track IDs in the current frame
            current_track_ids.append(track_id)

            # Draw bounding box on the frame
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f"ID: {track_id} {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Check for tracks that have left the frame
    tracks_left_frame = set(custom_track_ids.keys()) - set(current_track_ids)
    return tracked_frames
def process_tracked_obj():
    # Insert data into Redis for tracks that left the frame
    global tracks_left_frame, custom_track_ids
    while tracks_left_frame:
        track_id = next(iter(tracks_left_frame))
        # track_data = custom_track_ids[track_id]
        # r.set(track_data['custom_track_id'], str(track_data))  # Insert into Redis as a string or JSON

        # Remove the track ID from the custom_track_ids since it left the frame
        del custom_track_ids[track_id]
        tracks_left_frame.remove(track_id)
    # print(custom_track_ids.keys())

# Function to stream video and process frames in batches
def stream_process(video_path, batch_size=8):
    global orig_w,orig_h
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(orig_w, orig_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("/home/annone/ai/data/output.mp4", fourcc, fps, (640, 480))

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Record the current time in seconds for tracking purposes
        frame_time = time.time()

        frames.append(frame)

        if len(frames) >= batch_size:
            batch_results, resized_frames = process_frame_batch(frames)
            tracked_frames = track_objects_and_save(frames, batch_results, frame_time)

            for tracked_frame in tracked_frames:
                # tracked_frame = cv2.resize(tracked_frame, (640, 480))
                # process.stdin.write(tracked_frame.tobytes())
                cv2.imshow("Tracked Frame", tracked_frame)
                out.write(tracked_frame)
            
            frames = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


video_path = '/home/annone/ai/data/wrongway.mp4'

stream_process( video_path, batch_size=2)
print(tracks_left_frame, len(custom_track_ids))
process_tracked_obj()
print(tracks_left_frame, len(custom_track_ids))