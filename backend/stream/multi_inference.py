import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort.sort import Sort
from module import ViolationDetector
import uuid
import threading
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
video_num = 0
parent_dir = os.getenv("PARENT_DIR")

class ObjectTracker:
    def __init__(self):
        self.tracker = Sort()
        self.custom_track_ids = {}

    def update(self, detections):
        return self.tracker.update(np.array(detections))

class CameraStream:
    def __init__(self, rtsp_url,camera_ip, camera_id, model_path, violation_detector,thread_id):
        self.thread_id = thread_id
        self.rtsp_url = rtsp_url
        self.camera_ip = camera_ip
        self.camera_id = camera_id
        self.model = YOLO(model_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.violation_detector = [violation_detector for i in self.rtsp_url]
        self.orig_w, self.orig_h = 1280, 920
        self.resized_w = int(os.getenv("RESIZED_RESOLUTION_WIDTH"))
        self.resized_h = int(os.getenv("RESIZED_RESOLUTION_HEIGHT"))
        self.frames = []
        self.tracker = [ObjectTracker() for i in self.rtsp_url]
        self.class_list = [
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

    def process_frame_batch(self, frames):
        resized_frames = [cv2.resize(frame, (self.resized_w // 32 * 32, self.resized_h // 32 * 32)) for frame in frames]
        frames_tensor = (
            torch.from_numpy(np.stack(resized_frames))
            .permute(0, 3, 1, 2)
            .float()
            .to(self.model.device)
            / 255.0
        )
        with torch.no_grad():
            return self.model(frames_tensor, device=self.model.device)

    def track_objects(self, frames, batch_results, frame_time):
        tracked_frames = []
        current_track_ids = []

        for frame, result in zip(frames, batch_results):
            detections, labels, confs = [], [], []
            scale_x, scale_y = self.orig_w / self.resized_w, self.orig_h / self.resized_h

            if result.masks:
                for mask, box in zip(result.masks.xy, result.boxes):
                    x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy() * np.array([scale_x, scale_y, scale_x, scale_y])).astype(int)
                    score = box.conf[0].item()
                    label = self.model.names[int(box.cls[0])]
                    detections.append([x1, y1, x2, y2, score, int(box.cls[0])])
                    labels.append(label)
                    confs.append(score)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if len(detections) == 0:
                continue
            tracks = self.tracker[0].update(detections)

            for i, track in enumerate(tracks):
                track_id = int(track[4])
                x1, y1, x2, y2 = map(int, track[:4])
                current_track_ids.append(track_id)

                if label in self.class_list:
                    self.violation_detector[0].check_illegal_parking(track_id, (x1 + x2) // 2, (y1 + y2) // 2, label)
                    self.violation_detector[0].detect_traffic_violation(track_id, (x1 + x2) // 2, (y1 + y2) // 2, label)
                    self.violation_detector[0].detect_wrong_way_violation(track_id, (x1 + x2) // 2, (y1 + y2) // 2, label)

                if track_id not in self.tracker[0].custom_track_ids:
                    self.tracker[0].custom_track_ids[track_id] = {
                        "custom_track_id": self.generate_custom_track_id(labels[i], confs[i]),
                        "first_appearance": frame_time,
                        "last_appearance": frame_time,
                        "dbbox": [[x1, y1, x2, y2]],
                        "dlabel": [labels[i]],
                        "dconf": [confs[i]],
                    }
                else: 
                    self.tracker[0].custom_track_ids[track_id]["dbbox"].append([x1, y1, x2, y2])
                    self.tracker[0].custom_track_ids[track_id]["dlabel"].append(labels[i])
                    self.tracker[0].custom_track_ids[track_id]["dconf"].append(confs[i])
                    self.tracker[0].custom_track_ids[track_id]["last_appearance"] = frame_time

                self.handle_incidents(track_id, x1, y1, x2, y2, frame, labels[i], frame_time)

            tracked_frames.append(frame)

        return tracked_frames, current_track_ids

    def handle_incidents(self, track_id, x1, y1, x2, y2, frame, label, frame_time):
        bbox = f"{x1}, {y1}, {x2}, {y2}"
        incident_type = None
        now = datetime.now()
        file_name = now.strftime('%Y-%m-%d %H-%M-%S')

        # Red light violation
        if track_id in self.violation_detector[0].violated_objects:
            incident_type = self.violation_detector[0].INCIDENT_TYPES["TRAFFIC_VIOLATION"]
            self.violation_detector[0].save_violation_to_db(self.camera_id[0], track_id, self.camera_ip[0], bbox, incident_type, f"{file_name}_{track_id}.jpeg")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "red light violation", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Wrong way violation
        if track_id in self.violation_detector[0].violated_objects_wrong:
            incident_type = self.violation_detector[0].INCIDENT_TYPES["WRONG_WAY"]
            self.violation_detector[0].save_violation_to_db(self.camera_id[0], track_id, self.camera_ip[0], bbox, incident_type, f"{file_name}_{track_id}.jpeg")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "wrong way driving", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Illegal parking
        if track_id in self.violation_detector[0].static_objects and self.violation_detector[0].static_objects[track_id]["violated"]:
            incident_type = self.violation_detector[0].INCIDENT_TYPES["ILLEGAL_PARKING"]
            self.violation_detector[0].save_violation_to_db(self.camera_id[0], track_id, self.camera_ip[0], bbox, incident_type, f"{file_name}_{track_id}.jpeg")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "illegal parking", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    def generate_custom_track_id(self, label, confidence):
        return f"{label}_{confidence:.2f}_{uuid.uuid4()}"
    
    def connect_stream(self, url:str):
        cap = cv2.VideoCapture(url)
        while not cap.isOpened():
            print(f"Failed to connect. Retrying in 5 seconds...")
            time.sleep(5)
            cap = cv2.VideoCapture(url)  # Re-attempt connection
        print(f"Successfully connected!")
        return cap

    def run(self, batch_size = 3):
        global video_num
        print(self.rtsp_url)
        caps = [self.connect_stream(url = f"{video_url}") for video_url in self.rtsp_url]
        for i in self.rtsp_url:
            video_num = video_num + 1 
        # caps = [cv2.VideoCapture(self.rtsp_url[0])]
        for c in caps:
            if not c.isOpened():
                print(f"Error opening video file: {self.rtsp_url[0]}")
                return
        frames = []
        start_time = time.time()
        interval = 120  # 2 min
        t1 = time.time()
        while True:
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_time = time.time()
                frames.append(frame)
            if time.time() - start_time >= interval:
                cv2.imwrite(f"{parent_dir}/data/municipal/{frame_time}_{self.camera_id}.jpeg", frame)
                start_time = time.time()

            if len(frames) >= batch_size:  # Process in batches of 8
                batch_results = self.process_frame_batch(frames)
                tracked_frames, track_id_list = self.track_objects(frames, batch_results, frame_time)
                for tracked_frame in tracked_frames:
                    self.violation_detector[0].draw_lines_for_traffic_violation(tracked_frame)
                    tracked_frame = cv2.resize(tracked_frame, (640, 480))
                    print(self.camera_id)
                    cv2.imwrite("/home/annone/ai/data/img.jpeg",tracked_frame)
                    cv2.imshow(f"Tracked Frame", tracked_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frames = []

            print(self.thread_id)
        t2 = time.time()
        print(t2-t1, self.camera_ip)
        cap.release()
        cv2.destroyAllWindows()

class MultiCameraProcessor:
    def __init__(self, cam_rtsp_urls,cam_ids,cam_ips, model_path):
        self.cam_rtsp_urls = cam_rtsp_urls
        self.cam_ids = cam_ids
        self.cam_ips = cam_ips
        self.model_path = model_path
        self.violation_detector = ViolationDetector()

    def start_streams(self):
        threads = []
        for i in range(1):
            camera_stream = CameraStream(rtsp_url=cam_rtsp_urls, camera_ip= self.cam_ips,camera_id= self.cam_ids,model_path= self.model_path, violation_detector= self.violation_detector, thread_id=i)
            thread = threading.Thread(target=camera_stream.run)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

if __name__ == "__main__":
    cam_rtsp_urls = os.getenv("CAM_RTSPS").split(',')
    cam_ids = os.getenv("CAM_IDS").split(",")
    cam_ips = os.getenv("CAM_IPS").split(",")
    model_path = f"{os.getenv('PARENT_DIR')}/models/obj_seg_02.pt"
    camera_stream = CameraStream(rtsp_url=cam_rtsp_urls,camera_ip=cam_ips,camera_id=cam_ids,model_path=model_path,violation_detector=ViolationDetector(),thread_id=1)
    thread = threading.Thread(target=camera_stream.run)
    thread.start()
    thread.join()
    # processor = MultiCameraProcessor(cam_rtsp_urls= cam_rtsp_urls,cam_ids=cam_ids,cam_ips=cam_ips,model_path=model_path)
    # processor.start_streams()
    print(video_num)
