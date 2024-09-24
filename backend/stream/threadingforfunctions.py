import threading 
# from illegal_parking import check_illegal_parking
# from traffic_violation import detect_traffic_violations
# from wrong_way_driving import detect_wrong_way_violation
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
offset=10
count = 0
total_parking_violations = 0 
wrong_way_violation_count = 0
traffic_violation_count = 0  
crossed_objects = {}  # Track objects that have crossed lines
violated_objects = set()  # Track objects that have already violated
static_objects = {}  # To track objects that are stationary
stationary_frame_threshold = 200
ww_red_line = [
    [(133, 251), (438, 251)],  # First red line
    # [(42, 200), (368, 200)],   # Second red line
    # [(417, 182), (640, 182)]   # Third red line
]

ww_green_line = [
    [(44, 390), (525, 390)],   # First green line (paired with first red line)
    # [(213, 110), (368, 110)],  # Second green line (paired with second red line)
    # [(404, 118), (561, 118)]    # Third green line (paired with third red line)
]

# parking
def check_illegal_parking(track_id, cx, cy):
    """Check if an object is illegally parked based on stationary duration."""
    global static_objects, total_parking_violations
    print("illegal_parking")
    # Initialize tracking information if not already present
    if track_id not in static_objects:
        static_objects[track_id] = {"position": (cx, cy), "frames": 0, "violated": False}
    else:
        last_position = static_objects[track_id]["position"]

        # Check if the object has remained stationary (within a certain offset)
        if abs(cx - last_position[0]) <= offset and abs(cy - last_position[1]) <= offset:
            static_objects[track_id]["frames"] += 1
            # If the object is stationary for more than the threshold and not already marked as a violation
            if static_objects[track_id]["frames"] > stationary_frame_threshold and not static_objects[track_id]["violated"]:
                static_objects[track_id]["violated"] = True  # Mark the object as a violation
                total_parking_violations += 1  # Increase the parking violation count
                print(f"Object {track_id} marked as parking violation. Total Violations: {total_parking_violations}")
        else:
            # If the object has moved, reset its tracking information
            static_objects[track_id]["position"] = (cx, cy)
            static_objects[track_id]["frames"] = 0
            static_objects[track_id]["violated"] = False
# wrong way
def detect_wrong_way_violation(track_id, cx, cy):
    global wrong_way_violation_count, violated_objects
    print("wrong way")
    # Initialize tracking for the object if not already done
    if track_id not in crossed_objects:
        crossed_objects[track_id] = {'red': set(), 'green': set()}

    # Check if the object crosses any red line
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_red_line):
        if min(y_start, y_end) - offset <= cy <= max(y_start, y_end) + offset:
            if min(x_start, x_end) <= cx <= max(x_start, x_end):
                crossed_objects[track_id]['red'].add(i)

    # Check if the object crosses any green line
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_green_line):
        if min(y_start, y_end) - offset <= cy <= max(y_start, y_end) + offset:
            if min(x_start, x_end) <= cx <= max(x_start, x_end):
                crossed_objects[track_id]['green'].add(i)

    # Detect wrong-way violation (crossing green line after red line)
    if any(
        i in crossed_objects[track_id]['green'] and
        i in crossed_objects[track_id]['red'] and
        track_id not in violated_objects
        for i in crossed_objects[track_id]['green']
    ):
        # cv2.putText(frame, "Wrong Way Violation", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        wrong_way_violation_count += 1
        violated_objects.add(track_id)
# red light violation
def detect_traffic_violation(track_id, cx, cy):
    """Check if an object violates traffic rules by crossing lines."""
    global traffic_violation_count, crossed_objects, violated_objects
    print("Checking traffic violation")
    # Initialize tracking for the object if not already done
    if track_id not in crossed_objects:
        crossed_objects[track_id] = set()

    # Check if the object crosses any red line
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_red_line):
        if min(y_start, y_end) - offset <= cy <= max(y_start, y_end) + offset:
            if min(x_start, x_end) <= cx <= max(x_start, x_end):
                crossed_objects[track_id].add(f"red_{i}")
                print("red")

    # Check if the object crosses any green line after crossing a red line (indicating a violation)
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_green_line):
        if min(y_start, y_end) - offset <= cy <= max(y_start, y_end) + offset:
            if min(x_start, x_end) <= cx <= max(x_start, x_end):
                if any(f"red_{j}" in crossed_objects[track_id] for j in range(len(ww_red_line))) and track_id not in violated_objects:
                    # cv2.putText(frame, "Traffic Violation", (cx, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    traffic_violation_count += 1
                    violated_objects.add(track_id)
                    print("green")


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
                detections.append([x1, y1, x2, y2, score])
                labels.append(label)
                confs.append(score)
                cv2.polylines(frame, [np.array(mask, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
                cv2.putText(frame, f"{label} ({score:.2f})", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        tracks = tracker.update(np.array(detections))

        for i, track in enumerate(tracks):
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2


            # over_speeding = threading.Thread(target=, args=)
            # illegal_parking = threading.Thread(target=check_illegal_parking, args=(track_id,cx,cy))
            # traffic_violation = threading.Thread(target=detect_traffic_violation, args=(track_id,cx,cy))
            # wrong_way_driving = threading.Thread(target=detect_wrong_way_violation, args=(track_id,cx,cy))
            # license_plate = threading.Thread(target=, args=)
            # static_object_detection = threading.Thread(target=, args=)
            # over_speeding.start()
            # illegal_parking.start()
            # traffic_violation.start()
            # wrong_way_driving.start()
            # license_plate.start()
            # static_object_detection.start()
            # if track_id in violated_objects:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for violated objects
            #     cv2.putText(frame, f"{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # else:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for non-violated objects
            #     cv2.putText(frame, f"{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            # if track_id in violated_objects:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #     cv2.putText(frame, f"{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # else:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     cv2.putText(frame, f"{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            # # Draw bounding boxes around detected objects for illegal parking    
            # if static_objects[track_id]["violated"]:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #     cv2.putText(frame, "Parking Violation", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # else:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            

            # # If the object is new, generate a custom track ID and store initial data
            # if track_id not in custom_track_ids:
            #     custom_id = generate_custom_track_id(labels[i], confs[i])
            #     custom_track_ids[track_id] = {
            #         "custom_track_id": custom_id,
            #         "camera_id": camera_id,
            #         "camera_ip": camera_ip,
            #         "first_appearance": frame_time,  # Store first appearance time
            #         "last_appearance": frame_time,   # Initialize last appearance time
            #         "dbbox": [[x1, y1, x2, y2]],
            #         "dlabel": [labels[i]],
            #         "dconf": [confs[i]],
            #     }
            # else:
            #     # Append the new frame data to the existing object in the dict
            #     custom_track_ids[track_id]["dbbox"].append([x1, y1, x2, y2])
            #     custom_track_ids[track_id]["dlabel"].append(labels[i])
            #     custom_track_ids[track_id]["dconf"].append(confs[i])
            #     custom_track_ids[track_id]["last_appearance"] = frame_time  # Update last appearance time

            # # Add current track ID to the list of track IDs in the current frame
            # current_track_ids.append(track_id)

            # # Display the custom track ID on the frame
            # cv2.putText(frame, f"ID: {custom_track_ids[track_id]['custom_track_id']}", (x1, y1 - 30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

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

def drawnlinesfortrafficviolation(frame,total_violations):
    """Draw the red and green lines on the frame and display the traffic violation count."""
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_red_line):
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.putText(frame, f"Red {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_green_line):
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(frame, f"Green {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the traffic violation count
    cv2.rectangle(frame, (0, 0), (350, 60), (0, 255, 255), -1)  # Yellow background
    cv2.putText(frame, f'Traffic Violations - {traffic_violation_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Parking Violations - {total_violations}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Wrong Way Violations - {wrong_way_violation_count}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
# Function to stream video and process frames in batches
def stream_process(camera_id, camera_ip, video_path, batch_size=8):
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
                drawnlinesfortrafficviolation(tracked_frame,total_parking_violations)
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

# Example usage
video_path = '/home/annone/ai/data/T-pole wrong way.mp4'
cam_ip = '127.0.0.1'
cam_id = "1"
stream_process(cam_id, cam_ip, video_path, batch_size=2)







