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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("/home/annone/ai-camera/backend/stream/seg.pt")
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

red_line_y = 166
red_line_coord = ()
blue_line_y = 234
blue_line_coord = 2

text_color = (0, 0, 0)
yellow_color = (0, 255, 255)
red_color = (0, 0, 255)
blue_color = (255, 0, 0)

offset = 10


def stream_process(camera_id,camera_ip, video_path):
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
    while cap.isOpened():
        current_datetime = datetime.datetime.now()
        ret, ori_frame = cap.read()
        if not ret:
            break
        vehicle_count = 0
        width, height = 640, 480
        frame = cv2.resize(ori_frame, (width, height))
        results = model.track(frame, persist=True,device=device)
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

                    # create polygon/mask image of detection
                    cv2.fillPoly(null_mask, [points], 255)
                    polygon_image = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                    polygon_image[:, :, 3] = null_mask
                    cx, cy, cw, ch = cv2.boundingRect(points)
                    cropped_polygon_image = polygon_image[cy:cy+ch, cx:cx+cw]
                    detection_img_name = f"{random.randint(1,999)}_{custom_id}_{random.randint(1,999)}"
                    cv2.imwrite(f"/home/annone/ai-camera/backend/stream/d_temp/{detection_img_name}.png",cropped_polygon_image)

                    cv2.polylines(frame, points, True, (255, 0, 0), 1)
                    # color_number = classes_ids.index(int(box.cls[0]))
                    # cv2.fillPoly(frame, points, (0, 255, 0))
                    (x, y, w, h) = (box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2] - box.xyxy[0][0], box.xyxy[0][3] - box.xyxy[0][1])
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID: {custom_id}, {label}, Conf: {confidence:.2f}', 
                                    (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # (camera_id, camera_ip, timestamp, box_coords, detection_class, track_id, class_confidence, metadata)
                    data_string = f"{camera_id}|{camera_ip}|{current_datetime}|[{x},{y},{w},{h}]|{label}|{custom_id}|{confidence}|{detection_img_name}.png"
                    r.set(f"{detection_img_name}:raw_d_log",data_string)
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


                        cx = (x + w) // 2
                        cy = (y + h) // 2
                        x2 = x + w
                        y2 = y + h
                        # Check for going down
                        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                            down[track_id] = time.time()
                        if track_id in down:
                            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                                elapsed_time = time.time() - down[track_id]
                                if counter_down.count(track_id) == 0:
                                    if cx < 650:
                                        wrong_way_count += 1
                                        cv2.putText(frame, "Wrong Way ", (x2, y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                    counter_down.append(track_id)
                                    distance = 20  # meters
                                    a_speed_ms = distance / elapsed_time
                                    a_speed_kh = a_speed_ms * 3.6
                                    cv2.putText(frame, f"{int(a_speed_kh)} Km/h", (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

                        # Check for going up
                        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                            up[track_id] = time.time()
                        else:
                            return
                        if track_id in up:
                            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                                elapsed1_time = time.time() - up[track_id]
                                if counter_up.count(track_id) == 0:
                                    if cx > 650:
                                        wrong_way_count += 1
                                        cv2.putText(frame, "Wrong Way", (x2, y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                    counter_up.append(track_id)
                                    distance1 = 20  # meters
                                    a_speed_ms1 = distance1 / elapsed1_time
                                    a_speed_kh1 = a_speed_ms1 * 3.6
                                    cv2.putText(frame, f"{int(a_speed_kh1)} Km/h", (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)
        cv2.line(frame, (284, 166), (614, 166), red_color, 2)
        cv2.putText(frame, ('Red Line'), (168, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.line(frame, (254, 234), (680, 234), blue_color, 2)
        cv2.putText(frame, ('Blue Line'), (236, 234), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, ('Wrong Way - ' + str(wrong_way_count)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        current_time = time.time()
        if current_time - start_time >= interval:
            frame_name = f'{int(current_time * 1e6)}'
            cv2.imwrite(f"/home/annone/ai-camera/backend/stream/cc_temp/{frame_name}.jpg", frame)
            start_time = current_time
            r.set(f"{frame_name}:raw_cc_log",f"{camera_ip}|{current_datetime}|{frame_name}")
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # executor.shutdown(wait=False)

# STATIC ARGUMENTS

video_path = '/home/annone/ai-camera/traffic_light.mp4'
cam_ip = '127.0.0.1'
cam_id = "1"
stream_process(camera_id=cam_id,camera_ip=cam_ip,video_path=video_path)