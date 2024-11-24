import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort.sort import Sort
import time
from collections import defaultdict
from module import ViolationDetector
import subprocess
import uuid
import threading
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

parent_dir = os.getenv("PARENT_DIR")
camera_ip = os.getenv("CAM_IP")
camera_id = os.getenv("CAM_ID")
processed_rtsp_url = os.getenv("PROCESSED_RTSP_URL")
cam_rtsp_url = os.getenv("CAM_RTSP")
print(cam_rtsp_url)
var = ViolationDetector()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(f"{parent_dir}/models/obj_seg_02.pt")
model.to(device)
print(f"{device} as Computation Device initiated")
tracker = Sort()

orig_w, orig_h, resized_w, resized_h = (
    1280,
    920,
    int(os.getenv("RESIZED_RESOLUTION_WIDTH")),
    int(os.getenv("RESIZED_RESOLUTION_HEIGHT")),
)

fps = 30
class_list = [
    "auto",
    "bike-rider",
    "bolero",
    "bus",
    "car",
    "hatchback",
    "jcb",
    "motorbike-rider",
    "omni",
    "pickup",
    "scooty-rider",
    "scorpio",
    "sedan",
    "suv",
    "swift",
    "thar",
    "tractor",
    "truck",
    "van",
]
previous_positions = defaultdict(lambda: {"x": 0, "y": 0, "time": 0})
null_mask = np.zeros((resized_h, resized_w), dtype=np.uint8)


track_ids_inframe = {}
custom_track_ids = {}
tracks_left_frame = []
known_track_ids = []

# Construct the FFmpeg command
# ffmpeg_cmd = [
#     'ffmpeg',
#     '-y',  # Overwrite output files without asking
#     '-f', 'rawvideo',  # Input format
#     '-pix_fmt', 'bgr24',  # Pixel format
#     '-s', '640x480',  # Video resolution (adjust as needed)
#     '-r', '25',  # Frame rate
#     '-i', '-',  # Input from stdin
#     '-c:v', 'libx264',  # Video codec
#     '-preset', 'ultrafast',  # Encoding speed
#     '-tune', 'zerolatency',  # Tune for low latency
#     '-f', 'rtsp',  # Output format
#     processed_rtsp_url  # RTSP output URL
# ]
# process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, shell=True)


# Batch Processing
def process_frame_batch(frames):
    resized_frames = [
        cv2.resize(frame, (resized_w // 32 * 32, resized_h // 32 * 32))
        for frame in frames
    ]
    frames_tensor = (
        torch.from_numpy(np.stack(resized_frames))
        .permute(0, 3, 1, 2)
        .float()
        .to(device)
        / 255.0
    )

    with torch.no_grad():
        batch_results = model(frames_tensor, device=device)

    return batch_results


def generate_custom_track_id(label, confidence):
    return f"{label}_{confidence:.2f}_{uuid.uuid4()}"


# Function to track objects and draw segmentation polygons
def track_objects(frames, batch_results, frame_time):
    global camera_ip, previous_positions, fps, camera_id, track_ids_inframe, custom_track_ids, known_track_ids, tracks_left_frame, orig_w, resized_w, orig_h, resized_h

    tracked_frames = []
    current_track_ids = []

    for frame, result in zip(frames, batch_results):
        detections = []
        labels = []
        confs = []
        scale_x, scale_y = orig_w / resized_w, orig_h / resized_h
        now = datetime.now()
        file_name = now.strftime('%Y-%m-%d %H-%M-%S')
        if result.masks:
            for mask, box in zip(result.masks.xy, result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                )
                score = box.conf[0].item()
                label = model.names[int(box.cls[0])]
                detections.append([x1, y1, x2, y2, score, int(box.cls[0])])
                labels.append(label)
                confs.append(score)
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                )
                if label == "license-plate" or (isinstance(label, int) and label == 17):
                    cropped_plate = frame[y1 - 10 : y2 + 10, x1 - 10 : x2 + 10]
                    cv2.imwrite(
                        f"{parent_dir}/data/lp/{x1}_{y2}_{uuid.uuid4()}.jpg",
                        cropped_plate,
                    )
                if label == "human":
                    cropped_plate = frame[y1 - 10 : y2 + 10, x1 - 10 : x2 + 10]
                    cv2.imwrite(
                        f"{parent_dir}/data/human/{x1}_{y2}_{uuid.uuid4()}.jpg",
                        cropped_plate,
                    )
        tracks = tracker.update(np.array(detections))

        for i, track in enumerate(tracks):
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            bbox = f"{x1}, {y1}, {x2}, {y2}"
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = model.names[track[5]]

            if label in class_list:
                var.check_illegal_parking(track_id, cx, cy, label)
                var.detect_traffic_violation(track_id, cx, cy, label)
                var.detect_wrong_way_violation(track_id, cx, cy, label)

            if track_id not in custom_track_ids:
                custom_id = generate_custom_track_id(labels[i], confs[i])
                custom_track_ids[track_id] = {
                    "custom_track_id": custom_id,
                    "camera_id": camera_id,
                    "camera_ip": camera_ip,
                    "first_appearance": frame_time,
                    "last_appearance": frame_time,
                    "dbbox": [[x1, y1, x2, y2]],
                    "dlabel": [labels[i]],
                    "dconf": [confs[i]],
                }
            else:
                custom_track_ids[track_id]["dbbox"].append([x1, y1, x2, y2])
                custom_track_ids[track_id]["dlabel"].append(labels[i])
                custom_track_ids[track_id]["dconf"].append(confs[i])
                custom_track_ids[track_id]["last_appearance"] = frame_time

            current_track_ids.append(track_id)
            incident_type = None
            
            # red light violation
            if track_id in var.violated_objects:
                incident_type = var.INCIDENT_TYPES["TRAFFIC_VIOLATION"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "red light violation",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                if track_id not in var.logged_traffic:
                    var.save_violation_to_db(
                        camera_id, track_id, camera_ip, bbox, incident_type,  f"{file_name}_{track_id}.jpeg"
                    )
                    var.logged_traffic.add(track_id)
                    cv2.imwrite(f"{parent_dir}/data/red_light_violation/{file_name}_{track_id}.jpeg",frame)

            # wrong way violation
            if track_id in var.violated_objects_wrong:
                incident_type = var.INCIDENT_TYPES["WRONG_WAY"]
                if track_id not in var.logged_wrong:
                    var.save_violation_to_db(
                        camera_id, track_id, camera_ip, bbox, incident_type, f"{file_name}_{track_id}.jpeg"
                    )
                    var.logged_wrong.add(track_id)
                    cv2.imwrite(f"{parent_dir}/data/wrong_way_driving/{file_name}_{track_id}.jpeg", frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "wrong way driving",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
            # illegal parking
            if (
                track_id in var.static_objects
                and var.static_objects[track_id]["violated"]
            ):
                incident_type = var.INCIDENT_TYPES["ILLEGAL_PARKING"]
                if track_id not in var.logged_parking:
                    var.save_violation_to_db(
                        camera_id, track_id, camera_ip, bbox, incident_type, f"{file_name}_{track_id}.jpeg"
                    )
                    var.logged_parking.add(track_id)
                    cv2.imwrite(f"{parent_dir}/data/illegal_parking/{file_name}_{track_id}.jpeg",frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "illegal parking",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
            tracked_frames.append(frame)

    tracks_left_frame = set(custom_track_ids.keys()) - set(current_track_ids)
    return tracked_frames, list(custom_track_ids.keys())

# stream function
def stream_process(rtsp_url: str, batch_size: int = 8):
    global orig_w, orig_h
    cap = cv2.VideoCapture(rtsp_url)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{parent_dir}/data/output.mp4", fourcc, fps, (640, 480))
    if not cap.isOpened():
        print(f"Error opening video file: {rtsp_url}")
        return

    frames = []
    t1 = time.time()
    start_time = time.time()
    interval = 120 # 2 min
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = time.time()

        frames.append(frame)
        elapsed_time = frame_time - start_time

        if elapsed_time >= interval:
            cv2.imwrite(f"{parent_dir}/data/municipal/{frame_time}_{camera_ip}", frame)
            start_time = time.time()

        if len(frames) >= batch_size:
            batch_results = process_frame_batch(frames)
            tracked_frames, track_id_list = track_objects(
                frames, batch_results, frame_time
            )

            for tracked_frame in tracked_frames:
                var.draw_lines_for_traffic_violation(
                    tracked_frame
                )
                tracked_frame = cv2.resize(tracked_frame, (640, 480))
                # process.stdin.write(tracked_frame.tobytes())
                # cv2.imshow("Tracked Frame", tracked_frame)
                # out.write(tracked_frame)
            frames = []
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    t2 = time.time()
    print(t2 - t1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# update the code to save logs in db
def process_tracked_obj():
    global tracks_left_frame, custom_track_ids
    while tracks_left_frame:
        track_id = next(iter(tracks_left_frame))
        print(custom_track_ids[track_id])
        del custom_track_ids[track_id]
        tracks_left_frame.remove(track_id)

def process_urination():
    while True:
        imgs = [cv2.imread(f"{parent_dir}/data/human/{file}") for file in os.listdir(f"{parent_dir}/data/human")[:10]]
        model_path = f"{parent_dir}/models/pee_spit.pth"
        results = var.process_image(imgs, model_path, "uri_000", "000000", "00000")
        print(results)
        time.sleep(5)

# process_lp_images()
stream_process( cam_rtsp_url, batch_size=2)

# thread1 = threading.Thread(target=stream_process, args=(cam_rtsp_url, 2))
# thread2 = threading.Thread(target=process_urination)
# thread1.start()
# thread2.start()
# thread1.join()
# thread2.join()