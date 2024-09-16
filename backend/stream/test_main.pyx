# opencv-python -v 4.10.0.84
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
import tracemalloc
import pickle

tracemalloc.start()

# global variables

PARENT_DIR = "/home/annone/ai"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("/home/annone/ai/models/objseg50e.pt")
model.to(device)
r = redis.Redis(host='localhost', port=6379, db=0)
custom_track_ids = {}
track_ids_inframe = {}
print(device)

def clear_redis_database():
    r.flushdb()
    print("Redis database cleared.")

def create_thread_pool(num_threads):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
    return executor

def submit_tasks(executor, num_tasks, task):
    for i in range(num_tasks):
        executor.submit(task, i)

def shutdown_thread(executor):
    executor.shutdown()
    print("Thread pool shut down.")
class_list = ['auto', 'bike-rider', 'bolero', 'bus', 'car', 'hatchback', 'jcb', 'motorbike-rider', 'omni', 'pickup',
              'scooty-rider', 'scorpio', 'sedan', 'suv', 'swift', 'thar', 'tractor', 'truck', 'van']
last_positions = {}

### RED LIGHT VIOLATION DETECTION

offset = 10
count = 0
traffic_violation_count = 0
rlv_crossed_objects = {}
rlv_violated_objects = set()

red_lines = [
    [(1052,222), (1027,1080)],
    # [(402, 462), (560, 447)],
]

green_lines = [
    [(1052,222), (1680,433)],
]
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

illegal_parking_polygon = np.array([[300, 300], [610, 300], [610, 670], [300, 670]])
STATIONARY_FRAMES_THRESHOLD = 200
stationary_objects = {}
STATIONARY_OFFSET = 50


### WRONG WAY DRIVING DETECTION

wrong_way_violation_count = 0
ww_crossed_objects = {}
ww_violated_objects = set()
ww_offset = 10

ww_red_lines = [
    [(301,337), (1040,428)],
    # [(734,508), (1120,527)]
]

ww_green_lines = [
    [(1052,222), (532,193)],
    # [(1040,428), (1680,433)]
]


def is_within_offset(last_position, current_loc, offset):
    """Check if the current location is within the offset range of the last position."""
    if last_position is None:
        return False
    distance = np.linalg.norm(np.array(current_loc) - np.array(last_position))
    return distance <= offset

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def draw_lines_and_text(frame):
    """Draw the red and green lines on the frame and display the traffic violation count."""
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(red_lines):
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)
        cv2.putText(frame, f"Red {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)

    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(green_lines):
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
        cv2.putText(frame, f"Green {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

    cv2.rectangle(frame, (0, 0), (350, 50), (0, 255, 255), 1)
    cv2.putText(frame, f'Traffic Violations - {traffic_violation_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def calculate_speed(last_position, current_loc, fps):
    """ Calculate the speed of the object based on the distance covered between frames """
    if last_position is None:
        return 0
    distance = np.linalg.norm(np.array(current_loc) - np.array(last_position))
    speed = distance * fps
    return speed

def is_within_range(x, y, y_min, y_max, x_min, x_max):
    return y_min <= y <= y_max and x_min <= x <= x_max


### SETUP VARIABLES AND FUNCTION (OPTIMIZED AND READY TO USE)

red_line_ranges_np = np.array(red_line_ranges)
green_line_ranges_np = np.array(green_line_ranges)
ww_red_lines_np = np.array(ww_red_lines)
ww_green_lines_np = np.array(ww_green_lines)

def check_traffic_violation(cx, cy, track_id, rlv_crossed_objects, rlv_violated_objects, frame):
    for i, (y_min, y_max, x_min, x_max) in enumerate(red_line_ranges_np):
        if is_within_range(cx, cy, y_min, y_max, x_min, x_max):
            rlv_crossed_objects.setdefault(track_id, set()).add(f"red_{i}")
            cv2.putText(frame, "crossed red", (cx + 10, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
            print("crossed red")
            break

    for i, (y_min, y_max, x_min, x_max) in enumerate(green_line_ranges_np):
        if is_within_range(cx, cy, y_min, y_max, x_min, x_max):
            if f"red_{i}" in rlv_crossed_objects.get(track_id, set()) and track_id not in rlv_violated_objects:
                cv2.putText(frame, "crossed green", (cx + 10, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
                print("crossed green")
                return True

    return False

def check_wrong_way_violation(cx, cy, track_id, ww_crossed_objects, ww_violated_objects, ww_red_lines_np, ww_green_lines_np, ww_offset, frame):
    crossed_red = False
    crossed_green = False

    ww_crossed_objects.setdefault(track_id, {'red': set(), 'green': set()})

    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_red_lines_np):
        if min(y_start, y_end) - ww_offset <= cy <= max(y_start, y_end) + ww_offset and min(x_start, x_end) <= cx <= max(x_start, x_end):
            ww_crossed_objects[track_id]['red'].add(i)
            crossed_red = True
            break

    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_green_lines_np):
        if min(y_start, y_end) - ww_offset <= cy <= max(y_start, y_end) + ww_offset and min(x_start, x_end) <= cx <= max(x_start, x_end):
            ww_crossed_objects[track_id]['green'].add(i)
            crossed_green = True
            break

    if crossed_red and crossed_green and track_id not in ww_violated_objects:
        return True
    return False

def stream_process(camera_id, camera_ip, video_path):
    global traffic_violation_count, ww_offset, wrong_way_violation_count
    # clear_redis_database()
    # executor = create_thread_pool(6)
    # executor.submit(process_raw_d_logs)
    # executor.submit(process_raw_d_logs)
    # executor.submit(process_raw_d_logs)
    # executor.submit(process_raw_d_logs)
    # executor.submit(process_d_logs)
    # executor.submit(process_raw_cc_logs)
    # executor.submit(process_cc_logs)


    width, height = 1920, 1080
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
    null_mask = np.zeros((height, width), dtype=np.uint8)
    interval = 2
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter("/home/annone/ai/data/output.mp4", fourcc, fps, (width, height))

    t1 = time.time()
    while cap.isOpened():
        ret, ori_frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(ori_frame, (width, height))
        results = model.track(frame, persist=True, device=device)
        print(fps)
        vehicle_count = 0
        crowd_count = 0
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
                #     if track_id not in custom_track_ids:
                #         custom_track_ids[track_id] = generate_custom_string(
                #             camera_ip, track_id
                #         )
                #     custom_id = custom_track_ids[track_id]
                #     cv2.fillPoly(null_mask, [points], 255)
                #     polygon_image = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                #     polygon_image[:, :, 3] = null_mask
                    x, y, w, h = cv2.boundingRect(points)
                #     cx, cy = (x + w // 2, y + h // 2)
                #     current_loc = (cx, cy)
                #     last_position = last_positions.get(track_id)
                #     speed = calculate_speed(last_position, current_loc, fps)
                #     last_positions[track_id] = current_loc
                #     cropped_polygon_image = polygon_image[y:y+h, x:x+w].tobytes()
                #     detection_img_name = f"{random.randint(1,999)}_{custom_id}_{random.randint(1,999)}"
                #     # _, buffer = cv2.imencode('.png', cropped_polygon_image)
                #     # r.set(f"{detection_img_name}:image", buffer.tobytes())
                #     cv2.circle(frame,(cx,cy),2,(0, 255, 0))
                #     cv2.putText(frame, f'{label}, Speed: {speed:.2f} px/s',
                #                 (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                #     data_string = f"{camera_id}|{camera_ip}|[{x},{y},{w},{h}]|{label}|{custom_id}|{confidence}|{detection_img_name}"
                #     if custom_id not in track_ids_inframe:
                #         track_ids_inframe[custom_id] = {
                #         "camera_id":camera_id,
                #         "camera_ip":camera_ip,
                #         "dbbox":[[x,y,w,h]],
                #         "dlabel":[label],
                #         "dconf":[confidence],
                #         "dimg":[cropped_polygon_image]
                #         }
                #     else:
                #         track_ids_inframe[custom_id]["dbbox"].append([x,y,w,h])
                #         track_ids_inframe[custom_id]["dlabel"].append(label)
                #         track_ids_inframe[custom_id]["dconf"].append(confidence)
                #         track_ids_inframe[custom_id]["dimg"].append(cropped_polygon_image)

                #     ended_ids = track_ids_inframe.keys() - custom_track_ids
                #     for id in ended_ids:
                #         serialized_data = pickle.dumps(track_ids_inframe[id])
                #         r.set(f"{id}:new",serialized_data) 

#                     """

# {
# "23vgv4gv3432sfsfd4432":
#     {
#         cam_id:2,
#         cam_ip:12.23.23.34,
#         bbox:[[1,2,3,4],[24,5,6,2]],
#         dlabel:[auto,bike],
#         flabel:"auto",
#         dconfidenc:[0.28982,0.999882]
#         fconf:"0.99982"
#         dcolor:[[red,yellow],[blue, red, green]]
#         fcolor:red,yellow
#         dimg:[utf8, utf8]
#     }
# }
# """
                    # r.set(f"{detection_img_name}:raw_d_log", data_string)

                    # ### ILLEGAL PARKING
                    # if is_point_in_polygon(current_loc, illegal_parking_polygon):
                    #     if speed < 1e-2 or is_within_offset(last_position, current_loc, STATIONARY_OFFSET):
                    #         if track_id in stationary_objects:
                    #             stationary_objects[track_id] += 1
                    #         else:
                    #             stationary_objects[track_id] = 1
                    #         if stationary_objects[track_id] >= STATIONARY_FRAMES_THRESHOLD:
                    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    #             cv2.putText(frame, "Illegal Parking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
                    #     else:
                    #         stationary_objects[track_id] = 0
                    # else:
                    #     if track_id in stationary_objects:
                    #         del stationary_objects[track_id]

                    # ### RED LIGHT VIOLATION = rlv

                    # if track_id not in rlv_crossed_objects:
                    #     rlv_crossed_objects[track_id] = set()

                    # for i, (y_min, y_max, x_min, x_max) in enumerate(red_line_ranges):
                    #     if is_within_range(cx, cy, y_min, y_max, x_min, x_max):
                    #         rlv_crossed_objects[track_id].add(f"red_{i}")

                    # for i, (y_min, y_max, x_min, x_max) in enumerate(green_line_ranges):
                    #     if is_within_range(cx, cy, y_min, y_max, x_min, x_max):
                    #         if f"red_{i}" in rlv_crossed_objects[track_id] and track_id not in rlv_violated_objects:
                    #             cv2.putText(frame, "Traffic Violation", (x+w, y+h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
                    #             traffic_violation_count += 1
                    #             rlv_violated_objects.add(track_id)

                    # ### WRONG WAY DRIVING
                    # if track_id not in ww_crossed_objects:
                    #     ww_crossed_objects[track_id] = {'red': set(), 'green': set()}

                    # for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_red_lines):
                    #     if min(y_start, y_end) - ww_offset <= cy <= max(y_start, y_end) + ww_offset:
                    #         if min(x_start, x_end) <= cx <= max(x_start, x_end):
                    #             ww_crossed_objects[track_id]['red'].add(i)

                    # for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_green_lines):
                    #     if min(y_start, y_end) - ww_offset <= cy <= max(y_start, y_end) + ww_offset:
                    #         if min(x_start, x_end) <= cx <= max(x_start, x_end):
                    #             ww_crossed_objects[track_id]['green'].add(i)

                    #             if i in ww_crossed_objects[track_id]['green'] and i in ww_crossed_objects[track_id]['red'] and track_id not in ww_violated_objects:
                    #                 cv2.putText(frame, "Wrong Way Violation", (x+w, y+h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
                    #                 wrong_way_violation_count += 1
                    #                 ww_violated_objects.add(track_id)
                    #                 wrong_way_detected = True
                    cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), thickness=1)

                    if label in class_list:
                        vehicle_count += 1
                    if label in ['man', 'woman', 'child', 'person']:
                        crowd_count += 1
            # frame = result.plot(font_size=0.5, kpt_line=True, masks=True)
        cv2.putText(frame, f"vehicle count {vehicle_count}", (246, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"crowd count {crowd_count}", (246, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)    

        # ### WRONG WAY DRIVING
        # for (start_point, end_point) in ww_red_lines:
        #     cv2.line(frame, start_point, end_point, (0, 0, 255), 1)
        #     cv2.putText(frame, "ww", (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # for (start_point, end_point) in ww_green_lines:
        #     cv2.line(frame, start_point, end_point, (0, 255, 0), 1)
        #     cv2.putText(frame, "ww", (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ## ILLEGAL PARKING AREA DRWAING
        # cv2.polylines(frame, [illegal_parking_polygon], isClosed=True, color=(0, 255, 255), thickness=1)

        # ## RED LIGHT VIOLATION LINE DRAWING
        # for i, ((x_start, y_start), (x_end, y_end)) in enumerate(red_lines):
        #     cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)
        #     cv2.putText(frame, f"rlv {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # for i, ((x_start, y_start), (x_end, y_end)) in enumerate(green_lines):
        #     cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
        #     cv2.putText(frame, f"rlv {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # current_time = time.time()
        # if current_time - start_time >= interval:
        #     frame_name = f'{int(current_time * 1e6)}'
            # _, frame_img = cv2.imencode('.png', cropped_polygon_image)
            # r.set(f"{frame_name}:image", frame_img.tobytes())
            # start_time = current_time
            # r.set(f"{frame_name}:raw_cc_log", f"{camera_ip}|{frame_name}")
        # out.write(frame)
        # result_frame = frame.download()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # shutdown_thread(executor)
            break
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    t2 = time.time()
    print(t2-t1)


# STATIC ARGUMENTS

# video_path = '/home/annone/ai/data/T-pole wrong way.mp4'
# cam_ip = '127.0.0.1'
# cam_id = "1"
# stream_process(cam_id, cam_ip,video_path)



# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')

# for stat in top_stats:
#     print(stat)
# traffic violtion, over speeding, illegal parking,  ---wrong way
# if __name__ == "__main__":
#     import multiprocessing
#     multiprocessing.set_start_method('spawn')
#     video_path = '/home/annone/ai/data/traffic_light.mp4'
#     cam_ip = '127.0.0.1'
#     cam_id = "1"
#     p1 = multiprocessing.Process(target=stream_process, args= (cam_id, cam_ip,video_path))
#     p1.start()
#     # p2.start()
#     p1.join()
#     # p2.join()

#     print("Both functions have completed.")