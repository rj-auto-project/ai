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

# Initialize the ViolationDetector instance
var = ViolationDetector()


# Initialize device and model
PARENT_DIR = "/home/annone/ai"
# r = redis.Redis(host='localhost', port=6379, db=0)
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
known_track_ids = []



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
def track_objects(frames, batch_results, frame_time):
    global camera_ip, previous_positions, fps, camera_id, track_ids_inframe, custom_track_ids, known_track_ids, roi_points

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
            bbox = f"{x1}, {y1}, {x2}, {y2}"
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label =model.names[track[5]] 
            # print(roi_points)
            if label in class_list:
                var.check_illegal_parking(track_id, cx, cy, label)
                var.detect_traffic_violation(track_id, cx, cy, label)  
                var.detect_wrong_way_violation(track_id, cx, cy, label)       
            
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
            incident_type = None
            if track_id in var.violated_objects:
                incident_type = var.INCIDENT_TYPES["TRAFFIC_VIOLATION"]
                print(f"Saving violation: {incident_type} for track ID: {track_id}{label}")
                if track_id not in var.logged_traffic:  # Check if already logged
                    # var.save_violation_to_db(camera_id, track_id, camera_ip, bbox, incident_type)
                    var.logged_traffic.add(track_id)  # Mark as logged

            # Check for wrong way violation
            if track_id in var.violated_objects_wrong:
                incident_type = var.INCIDENT_TYPES["WRONG_WAY"]
                print(f"Saving violation: {incident_type} for track ID: {track_id}{label}")
                if track_id not in var.logged_wrong:  # Check if already logged
                    # var.save_violation_to_db(camera_id, track_id, camera_ip, bbox, incident_type)
                    var.logged_wrong.add(track_id)
            if track_id in var.static_objects and var.static_objects[track_id]["violated"]:
                incident_type = var.INCIDENT_TYPES["ILLEGAL_PARKING"]
                print(f"Saving violation: {incident_type} for track ID: {track_id}{label}")
                if track_id not in var.logged_parking:  # Check if already logged
                    # var.save_violation_to_db(camera_id, track_id, camera_ip, bbox, incident_type)
                    var.logged_parking.add(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Parking Violation", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # if incident_type:  # If an incident type is detected, save it
            #     print(f"Saving violation: {incident_type} for track ID: {track_id}")
            #     var.save_violation_to_db(camera_id, track_id, camera_ip, bbox, incident_type)
            # Display the custom track ID on the frame
            # cv2.putText(frame, f"ID: {custom_track_ids[track_id]['custom_track_id']}", (x1, y1 - 30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            # print(label,track_id)
            cv2.putText(frame, f"ID:{track_id}{label}", (x1, y1 - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            # Append the current frame to the tracked frames
            tracked_frames.append(frame)

    # Check for tracks that are no longer in the current frame (left the frame)
    tracks_left_frame = set(custom_track_ids.keys()) - set(current_track_ids)

    # Insert data into Redis for tracks that left the frame
    for track_id in tracks_left_frame:
        track_data = custom_track_ids[track_id]
        # r.set(track_data['custom_track_id'], str(track_data))  # Insert into Redis as a string or JSON

        # Remove the track ID from the custom_track_ids since it left the frame
        del custom_track_ids[track_id]

    return tracked_frames, list(custom_track_ids.keys())


# Function to stream video and process frames in batches
def stream_process( video_path, batch_size=8):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("/home/annone/ai/data/output.mp4", fourcc, fps, (640,480))
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
                var.draw_lines_for_traffic_violation(tracked_frame,var.total_parking_violations)
                
                cv2.imshow("Tracked Frame", tracked_frame)
                out.write(tracked_frame)
            frames = []
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    t2 = time.time()
    print(t2-t1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(custom_track_ids)


video_path = '/home/annone/ai/data/wrongway.mp4'

stream_process( video_path, batch_size=2)