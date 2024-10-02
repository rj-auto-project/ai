# import pandas as pd
# import psycopg2
# from datetime import datetime, timedelta
# import time
# from db import Database
# import random
# import uuid
# # Function to insert a row into the PostgreSQL table
# conn = Database.get_connection()
# cursor = conn.cursor()
# def insert_row(row):
#     global cursor, conn
#     try:

#         data = (row["camera_ip"], row["camera_id"],row["detection_id"],  row['license_number'],row['prediction_confidence'], row['track_id'],row['boxCoords'], row['detectionClass'], row["ownerName"])
#         print(data)
#         insert_query = '''
#         INSERT INTO "AnprLogs" (camera_ip,camera_id,detection_id, license_number, prediction_confidence, "trackId", "boxCoords", "detectionClass", "ownerName")
#         VALUES (%s, %s, %s, %s,%s,%s,%s,%s,%s);
#         '''
#         cursor.execute(insert_query, data)
#         conn.commit()

#         print(f"Inserted row: {row.to_dict()} at {datetime.now().strftime('%H:%M:%S')}")
#     except (Exception, psycopg2.Error) as error:
#         print("Error while connecting to PostgreSQL", error)
#     # finally:
#     #     if conn:
#     #         cursor.close()
#     #         conn.close()

# import os

# # Step 1: Read the CSV file
# csv_file = '/home/annone/ai/backend/stream/test.csv'
# df = pd.read_csv(csv_file)
# # Step 2: Start a countdown timer from 5:00 (HH:MM)
# start_time = datetime.strptime("05:00", "%M:%S")
# current_time = start_time

# # Convert the 'Time' column in the DataFrame to datetime objects
# df['Time'] = pd.to_datetime(df['Time'], format='%M:%S').dt.time

# # Step 3: Countdown loop
# while current_time.time() <= df['Time'].max():
#     current_formatted_time = current_time.strftime("%M:%S")
    
#     # Check if the current time matches any row in the CSV
#     matching_rows = df[df['Time'] == current_time.time()]
    
#     for _, row in matching_rows.iterrows():
#         row = row[:-1]
#         row["camera_id"] = "2"
#         row["camera_ip"] = "128.9.9.12"
#         row["detection_id"] = random.randint(178,999)
#         row['prediction_confidence'] = random.random()
#         row["track_id"] = str(uuid.uuid4())
#         row["boxCoords"] = [f"{random.randint(0,1080)}" f"{random.randint(0,1080)}" f"{random.randint(0,1080)}" f"{random.randint(0,1080)}"]
#         row["ownerName"] = "ramu kaka"
#         for i in row:
#             print(type(i))
#         print(row)
#         insert_row(row)
    
#     # Wait for one second
#     time.sleep(1)
    
#     # Increment the timer by one second
#     current_time += timedelta(seconds=1)
    
#     # Print the current timer (optional)
#     print(f"Timer: {current_formatted_time}")
    

# print("Finished processing all rows.")
# if conn:
#     cursor.close()
#     conn.close()















import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort.sort import Sort
import time
from collections import defaultdict
import math
import redis
import uuid
import subprocess
import os 

# Initialize device and model
r = redis.Redis(host='localhost', port=6379, db=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_file = os.path.abspath("/home/annone/ai/models/objseg50e.pt")
model = YOLO(model_file)

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
known_track_ids = []

# STREAMING CONSTANTS
rtsp_url = 'rtsp://localhost:8554/stream1'  # Update this URL as needed

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


# Function to generate a custom track ID based on YOLO class, confidence, and a unique UUID
def generate_custom_track_id(label, confidence):
    return f"{label}_{confidence:.2f}_{uuid.uuid4()}"

# Function to track objects and draw segmentation polygons
def track_objects(frames, batch_results, frame_time):
    global camera_ip, previous_positions, fps, camera_id, track_ids_inframe, custom_track_ids, known_track_ids

    tracked_frames = []
    current_track_ids = []  # To keep track of the tracks currently in the frame

    for frame, result in zip(frames, batch_results):
        detections = []
        img_bin = []
        labels = []
        confs = []

        if result.masks:
            for mask, box in zip(result.masks.xy, result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].item()
                label = model.names[int(box.cls[0])]
                detections.append([x1, y1, x2, y2, score, int(box.cls[0])])
                labels.append(label)
                confs.append(score)
                cv2.polylines(frame, [np.array(mask, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
                cv2.putText(frame, f"{label} ({score:.2f})", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        tracks = tracker.update(np.array(detections))

        for i, track in enumerate(tracks):
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])

            # If the object is new, generate a custom track ID and store initial data
            if track_id not in custom_track_ids:
                custom_id = generate_custom_track_id(labels[i], confs[i])
                custom_track_ids[track_id] = {
                    "custom_track_id": custom_id,
                    "camera_id": camera_id,
                    "camera_ip": camera_ip,
                    "first_appearance": frame_time,  # Store first appearance time
                    "last_appearance": frame_time,   # Initialize last appearance time
                    "dbbox": [[x1, y1, x2, y2]],
                    "dlabel": [labels[i]],
                    "dconf": [confs[i]],
                }
            else:
                # Append the new frame data to the existing object in the dict
                custom_track_ids[track_id]["dbbox"].append([x1, y1, x2, y2])
                custom_track_ids[track_id]["dlabel"].append(labels[i])
                custom_track_ids[track_id]["dconf"].append(confs[i])
                custom_track_ids[track_id]["last_appearance"] = frame_time  # Update last appearance time

            # Add current track ID to the list of track IDs in the current frame
            current_track_ids.append(track_id)

            # Display the custom track ID on the frame
            cv2.putText(frame, f"ID: {custom_track_ids[track_id]['custom_track_id']}", (x1, y1 - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            # Append the current frame to the tracked frames
            tracked_frames.append(frame)

    # Check for tracks that are no longer in the current frame (left the frame)
    tracks_left_frame = set(custom_track_ids.keys()) - set(current_track_ids)

    # Insert data into Redis for tracks that left the frame
    for track_id in tracks_left_frame:
        track_data = custom_track_ids[track_id]
        del custom_track_ids[track_id]

    return tracked_frames, list(custom_track_ids.keys())


# Function to stream video and process frames in batches
def stream_process(camera_id, camera_ip, video_path, batch_size=8):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("./data/output.mp4", fourcc, fps, (640,480))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frames = []
    t1 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Record the current time in seconds for tracking purposes
        frame_time = time.time()

        frames.append(frame)

        if len(frames) >= batch_size:
            batch_results, resized_frames = process_frame_batch(frames)

            tracked_frames, track_id_list = track_objects(resized_frames, batch_results, frame_time)
            for tracked_frame in tracked_frames:
                cv2.imshow("Tracked Frame", tracked_frame)
                process.stdin.write(tracked_frame.tobytes())
                print("LLL")
                # out.write(tracked_frame)
            frames = []
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    t2 = time.time()
    print(t2-t1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(custom_track_ids)

# Example usage
video_path = '/home/annone/ai/data/wrongway.mp4'
cam_ip = '127.0.0.1'
cam_id = "1"
stream_process(cam_id, cam_ip, video_path, batch_size=2)