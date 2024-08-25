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
model = YOLO("/home/annone/ai/models/yolov8n-seg.pt")
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

def red_light_violation():
    pass

def over_speeding():
    pass

def wrong_way_driving():
    print("test")

# global_variables
down = {}
up = {}
counter_down = []
counter_up = []
wrong_way_count = 0

# Additional dictionary to store the last positions of each tracked object
last_positions = {}

red_line_y = 166
red_line_coord = ()
blue_line_y = 234
blue_line_coord = 2

text_color = (0, 0, 0)
yellow_color = (0, 255, 255)
red_color = (0, 0, 255)
blue_color = (255, 0, 0)

offset = 10

def calculate_speed(last_position, current_position, fps):
    """ Calculate the speed of the object based on the distance covered between frames """
    if last_position is None:
        return 0
    distance = np.linalg.norm(np.array(current_position) - np.array(last_position))
    speed = distance * fps  # Speed in pixels per second
    return speed

def stream_process(camera_id, camera_ip, video_path):
    global wrong_way_count
    # clear_redis_database()
    executor = create_thread_pool(6)
    executor.submit(process_raw_d_logs)
    executor.submit(process_d_logs)
    executor.submit(process_raw_cc_logs)
    executor.submit(process_cc_logs)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
    null_mask = np.zeros((480, 640), dtype=np.uint8)
    interval = 2  # Save frame every 2 seconds
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    while cap.isOpened():
        current_datetime = datetime.datetime.now()
        ret, ori_frame = cap.read()
        if not ret:
            break
        vehicle_count = 0
        width, height = 640, 480
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
                    cx, cy, cw, ch = cv2.boundingRect(points)
                    current_position = (cx + cw // 2, cy + ch // 2)
                    # Calculate speed based on the last known position
                    last_position = last_positions.get(track_id)
                    speed = calculate_speed(last_position, current_position, fps)
                    # speed= "0.99"
                    last_positions[track_id] = current_position
                    cropped_polygon_image = polygon_image[cy:cy+ch, cx:cx+cw]
                    unique_id = str(uuid.uuid4())
                    detection_img_name = f"{unique_id}_{custom_id}_{random.randint(1,999)}"
                    _, buffer = cv2.imencode('.png', cropped_polygon_image)
                    r.set(f"{detection_img_name}:image", buffer.tobytes())

                    # Annotate frame with bounding box, label, confidence, and speed
                    cv2.rectangle(frame, (int(cx), int(cy)), (int(cx + cw), int(cy + ch)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label}, Speed: {speed:.2f} px/s',
                                (int(cx), int(cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # (camera_id, camera_ip, timestamp, box_coords, detection_class, track_id, class_confidence, metadata)
                    data_string = f"{camera_id}|{camera_ip}|{current_datetime}|[{cx},{cy},{cw},{ch}]|{label}|{custom_id}|{confidence}|{detection_img_name}"
                    r.set(f"{detection_img_name}:raw_d_log", data_string)
                    if label in [
                        "auto",
                        "motorbike",
                        "bike-rider",
                        "car",
                        "suv",
                        "hatchback",
                        "sedan",
                        "scooty-rider",
                        "scooty",
                        "bus",
                        "truck",
                        "tractor",
                        "loader",
                    ]:
                        vehicle_count += 1

        current_time = time.time()
        if current_time - start_time >= interval:
            frame_name = f'{int(current_time * 1e6)}'
            _, frame_img = cv2.imencode('.png', cropped_polygon_image)
            r.set(f"{frame_name}:image", frame_img.tobytes())
            # cv2.imwrite(f"{PARENT_DIR}/backend/stream/cc_temp/{frame_name}.jpg", frame)
            start_time = current_time
            r.set(f"{frame_name}:raw_cc_log", f"{camera_ip}|{current_datetime}|{frame_name}")
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # executor.shutdown(wait=False)

# STATIC ARGUMENTS

video_path = '/home/annone/ai/data/traffic_light.mp4'
cam_ip = '127.0.0.1'
cam_id = "1"
stream_process(camera_id=cam_id, camera_ip=cam_ip, video_path=video_path)
