import cv2
import time
import torch
from ultralytics import YOLO
import numpy as np
import redis
from module import generate_custom_string, process_raw_d_logs, process_d_logs, process_raw_cc_logs, process_cc_logs
import concurrent.futures
import time
import random
import datetime
import uuid

PARENT_DIR = "/home/annone/ai"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("/home/annone/ai/models/50e.pt")
model.to(device)

r = redis.Redis(host='localhost', port=6379, db=0)
custom_track_ids = {}
print(device)

def clear_redis_database():
    r.flushdb()
    print("Redis database cleared.")

# Thread Creator
def create_thread_pool(num_threads):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
    return executor

# Submit Thread to Thread pool
def submit_tasks(executor, num_tasks, task):
    for i in range(num_tasks):
        executor.submit(task, i)

# global_variables
class_list = ['auto', 'bike-rider', 'bolero', 'bus', 'car', 'hatchback', 'jcb', 'motorbike-rider', 'omni', 'pickup',
              'scooty-rider', 'scorpio', 'sedan', 'suv', 'swift', 'thar', 'tractor', 'truck', 'van']


# Additional dictionary to store the last positions of each tracked object
last_positions = {}

# red_line_y = 166
# red_line_coord = ()
# blue_line_y = 234
# blue_line_coord = 2

# text_color = (0, 0, 0)
# yellow_color = (0, 255, 255)
# red_color = (0, 0, 255)
# blue_color = (255, 0, 0)


### RED LIGHT VIOLATION DETECTION

offset = 10
count = 0  # Frame counter
traffic_violation_count = 0  # Counter for traffic violations
rlv_crossed_objects = {}  # To track which objects have crossed lines
rlv_violated_objects = set()  # To track objects that have already violated traffic rules

red_lines = [
    [(794,711), (735,131)],  # Red Line 1
    [(402, 462), (560, 447)],  # Red Line 2
]

green_lines = [
    [(735,131), (1116 ,268)],  # Green Line 1
]

# Precompute y and x ranges for red and green lines to avoid recomputing them in each iteration
red_line_ranges = [
    (
        min(y_start, y_end) - offset, max(y_start, y_end) + offset,
        min(x_start, x_end), max(x_start, x_end)
    )
    for (x_start, y_start), (x_end, y_end) in red_lines
]

green_line_ranges = [
    (
        min(y_start, y_end) - offset, max(y_start, y_end) + offset,
        min(x_start, x_end), max(x_start, x_end)
    )
    for (x_start, y_start), (x_end, y_end) in green_lines
]

### ILLEGAL PARKING DETECTION

# illegal parking coords
illegal_parking_polygon = np.array([[300, 300], [610, 300], [610, 670], [300, 670]])

# Threshold for the number of consecutive frames for illegal parking
STATIONARY_FRAMES_THRESHOLD = 200
stationary_objects = {}  # To keep track of stationary frame counts
STATIONARY_OFFSET = 50


### WRONG WAY DRIVING DETECTION

# Initialize counters and tracking variables
wrong_way_violation_count = 0
ww_crossed_objects = {}  # Track objects that have crossed lines
ww_violated_objects = set()  # Track objects that have already violated
ww_offset = 10

# Define red and green lines as lists of tuples
ww_red_lines = [
    # [(133, 251), (438, 251)],  # First red line
    [(624,401), (180,376)],   # Second red line
    [(734,508), (1120,527)]   # Third red line
]

ww_green_lines = [
    # [(44, 390), (525, 390)],   # First green line (paired with first red line)
    [(383,187), (660,215)],  # Second green line (paired with second red line)
    [(1122,274), (759,275)]    # Third green line (paired with third red line)
]


def is_within_offset(last_position, current_loc, offset):
    """Check if the current location is within the offset range of the last position."""
    if last_position is None:
        return False
    distance = np.linalg.norm(np.array(current_loc) - np.array(last_position))
    return distance <= offset

# Function to check if a point is inside the polygon
def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def draw_lines_and_text(frame):
    """Draw the red and green lines on the frame and display the traffic violation count."""
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(red_lines):
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.putText(frame, f"Red {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(green_lines):
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(frame, f"Green {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the traffic violation count
    cv2.rectangle(frame, (0, 0), (350, 50), (0, 255, 255), -1)
    cv2.putText(frame, f'Traffic Violations - {traffic_violation_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def calculate_speed(last_position, current_loc, fps):
    """ Calculate the speed of the object based on the distance covered between frames """
    if last_position is None:
        return 0
    distance = np.linalg.norm(np.array(current_loc) - np.array(last_position))
    speed = distance * fps  # Speed in pixels per second
    return speed

def is_within_range(x, y, y_min, y_max, x_min, x_max):
    return y_min <= y <= y_max and x_min <= x <= x_max


def stream_process(camera_id, camera_ip, video_path):
    global traffic_violation_count, ww_offset, wrong_way_violation_count
    # clear_redis_database()
    executor = create_thread_pool(6)
    # executor.submit(process_raw_d_logs)
    # executor.submit(process_d_logs)
    # executor.submit(process_raw_cc_logs)
    # executor.submit(process_cc_logs)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
    null_mask = np.zeros((720, 1280), dtype=np.uint8)
    interval = 2  # Save frame every 2 seconds
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter("/home/annone/ai/backend/stream/output.mp4", fourcc, fps, (width, height))
    while cap.isOpened():
        current_datetime = datetime.datetime.now()
        ret, ori_frame = cap.read()
        width, height = 1280, 720
        if not ret:
            break
        vehicle_count = 0
        frame = cv2.resize(ori_frame, (width, height))
        results = model.track(frame, persist=True, device=device)
        print(fps)
        for result in results:
            if result.masks:
                track_ids = (
                    result.boxes.id.int().cpu().tolist()
                    if result.boxes.id is not None
                    else []
                )
                for mask, box, track_id in zip(result.masks.xy, result.boxes, track_ids):
                    points = np.int32([mask])
                    confidence = box.conf[0]
                    label = model.names[int(box.cls[0])]
                    if track_id not in custom_track_ids:
                        custom_track_ids[track_id] = generate_custom_string(
                            cam_ip, track_id
                        )
                    custom_id = custom_track_ids[track_id]
                    cv2.fillPoly(null_mask, [points], 255)
                    polygon_image = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                    polygon_image[:, :, 3] = null_mask
                    # Calculate the object's centroid for speed detection
                    x, y, w, h = cv2.boundingRect(points)
                    cx, cy = (x + w // 2, y + h // 2)
                    current_loc = (cx, cy)
                    cv2.circle(frame,(cx,cy),2,(0, 255, 0))
                    # Calculate speed based on the last known position
                    last_position = last_positions.get(track_id)
                    speed = calculate_speed(last_position, current_loc, fps)
                    last_positions[track_id] = current_loc
                    cropped_polygon_image = polygon_image[y:y+h, x:x+w]
                    # unique_id = str(uuid.uuid4())
                    detection_img_name = f"{random.randint(1,999)}_{custom_id}_{random.randint(1,999)}"
                    _, buffer = cv2.imencode('.png', cropped_polygon_image)
                    r.set(f"{detection_img_name}:image", buffer.tobytes())

                    # Annotate frame with bounding box, label, confidence, and speed
                    # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    # cv2.putText(frame, f'{label}, Speed: {speed:.2f} px/s',
                    #             (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # (camera_id, camera_ip, timestamp, box_coords, detection_class, track_id, class_confidence, metadata)
                    data_string = f"{camera_id}|{camera_ip}|{current_datetime}|[{x},{y},{w},{h}]|{label}|{custom_id}|{confidence}|{detection_img_name}"
                    r.set(f"{detection_img_name}:raw_d_log", data_string)
                    
                    ### ILLEGAL PARKING
                    if is_point_in_polygon(current_loc, illegal_parking_polygon):
                        # Consider the object stationary if within offset range
                        if speed < 1e-2 or is_within_offset(last_position, current_loc, STATIONARY_OFFSET):
                            # Increment the stationary frame count
                            if track_id in stationary_objects:
                                stationary_objects[track_id] += 1
                            else:
                                stationary_objects[track_id] = 1

                            # Mark as illegal parking if the object is stationary for 1500 frames
                            if stationary_objects[track_id] >= STATIONARY_FRAMES_THRESHOLD:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                cv2.putText(frame, "Illegal Parking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        else:
                            # Reset the stationary frame count if the object starts moving
                            stationary_objects[track_id] = 0
                    else:
                        # If the object moves out of the polygon, remove it from tracking
                        if track_id in stationary_objects:
                            del stationary_objects[track_id]

                    ### RED LIGHT VIOLATION = rlv
                    if track_id not in rlv_crossed_objects:
                        rlv_crossed_objects[track_id] = set()

                    for i, (y_min, y_max, x_min, x_max) in enumerate(red_line_ranges):
                        if is_within_range(cx, cy, y_min, y_max, x_min, x_max):
                            rlv_crossed_objects[track_id].add(f"red_{i}")

                    # Process each green line and check for traffic violations
                    for i, (y_min, y_max, x_min, x_max) in enumerate(green_line_ranges):
                        if is_within_range(cx, cy, y_min, y_max, x_min, x_max):
                            if f"red_{i}" in rlv_crossed_objects[track_id] and track_id not in rlv_violated_objects:
                                cv2.putText(frame, "Traffic Violation", (x+w, y+h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                traffic_violation_count += 1
                                rlv_violated_objects.add(track_id)

                    # WRONG WAY DRIVING
                    if track_id not in ww_crossed_objects:
                        ww_crossed_objects[track_id] = {'red': set(), 'green': set()}

                    # Check if the object crosses any red line
                    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_red_lines):
                        if min(y_start, y_end) - ww_offset <= cy <= max(y_start, y_end) + ww_offset:
                            if min(x_start, x_end) <= cx <= max(x_start, x_end):
                                ww_crossed_objects[track_id]['red'].add(i)

                    # Check if the object crosses any green line
                    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_green_lines):
                        if min(y_start, y_end) - ww_offset <= cy <= max(y_start, y_end) + ww_offset:
                            if min(x_start, x_end) <= cx <= max(x_start, x_end):
                                ww_crossed_objects[track_id]['green'].add(i)

                                # Detect wrong-way violation (crossing green line after red line)
                                if i in ww_crossed_objects[track_id]['green'] and i in ww_crossed_objects[track_id]['red'] and track_id not in ww_violated_objects:
                                    cv2.putText(frame, "Wrong Way Violation", (x+w, y+h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                    wrong_way_violation_count += 1  # Increase violation count
                                    ww_violated_objects.add(track_id)  # Mark object as violated
                                    wrong_way_detected = True  # Set violation flag

                    if label in class_list:
                        vehicle_count += 1

        # vehicle count
        cv2.putText(frame, f"vehicle count {vehicle_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # cv2.putText(frame, f"crowd count {crowd_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # cv2.putText(frame, f"vehicle count {vehicle_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        


        ### WRONG WAY DRIVING
        # draw red line
        for (start_point, end_point) in ww_red_lines:
            cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
            cv2.putText(frame, "Red", (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # Draw green lines
        for (start_point, end_point) in ww_green_lines:
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(frame, "Green", (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ### ILLEGAL PARKING AREA DRWAING
        cv2.polylines(frame, [illegal_parking_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        ### RED LIGHT VIOLATION LINE DRAWING
        # draw red line
        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(red_lines):
            cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
            cv2.putText(frame, f"Red {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # Draw green lines
        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(green_lines):
            cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(frame, f"Green {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        current_time = time.time()
        if current_time - start_time >= interval:
            frame_name = f'{int(current_time * 1e6)}'
            _, frame_img = cv2.imencode('.png', cropped_polygon_image)
            r.set(f"{frame_name}:image", frame_img.tobytes())
            start_time = current_time
            r.set(f"{frame_name}:raw_cc_log", f"{camera_ip}|{current_datetime}|{frame_name}")

        out.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# STATIC ARGUMENTS

# traffic violtion, over speeding, illegal parking,  ---wrong way
video_path = '/home/annone/ai/data/traffic_light.mp4'
cam_ip = '127.0.0.1'
cam_id = "1"
stream_process(camera_id=cam_id, camera_ip=cam_ip, video_path=video_path)
