import cv2
import psycopg2
import numpy as np
from detection_keypoint import DetectKeypoint
from classification_keypoint import KeypointClassification
import os
import cv2
from lp_util import read_license_plate
from db import Database
import ast
from dotenv import load_dotenv
import time
class ViolationDetector:
    def __init__(self):
        self.offset = 3
        self.static_obj_offset = 5
        self.red_light_line_offset = 10
        self.total_parking_violations = 0
        self.wrong_way_violation_count = 0
        self.traffic_violation_count = 0
        self.crossed_objects_wrong = {}
        self.violated_objects_wrong = set()
        self.logged_wrong = set() 
        self.crossed_objects = {}
        self.logged_parking = set()
        self.violated_objects = set()
        self.logged_traffic = set() 
        self.static_objects = {}
        self.stationary_frame_threshold = 50
        self.INCIDENT_TYPES = {
                        "TRAFFIC_VIOLATION": "REDLIGHT_VIOLATION",
                        "ILLEGAL_PARKING": "ILLEGAL_PARKING",
                        "WRONG_WAY": "WRONG_WAY_DRIVING"
                        }
        load_dotenv()
        #  illegal parking area coords
        self.roi_area = ast.literal_eval(os.getenv("ILLEGAL_PARKING_ZONE_COORDS"))
        print(self.roi_area)
        # wrong way lines pair
        self.ww_line_pair = ast.literal_eval(os.getenv("WRONG_WAY_LINE_PAIR"))
        self.ww_red_line = []
        self.ww_green_line = []

        # traffic violation line pair
        self.tv_line_pair = ast.literal_eval(os.getenv("RED_LIGHT_LINE_PAIR"))
        self.tv_red_line = []
        self.tv_green_line = []
        
        for lines in self.ww_line_pair:
            self.ww_green_line.append(lines[1])
            self.ww_red_line.append(lines[0])

        for lines in self.tv_line_pair:
            self.tv_green_line.append(lines[1])
            self.tv_red_line.append(lines[0])
        # Database connection setup
        # self.db_connection = Database.get_connection()
        # self.db_cursor = self.db_connection.cursor()   
    
    def save_violation_to_db(self, camera_id, track_id, camera_ip, bbox, incident_type, thumbnail):
        """Save violation details to the PostgreSQL database."""
        try:
            insert_query = """
                INSERT INTO "IncidentLogs" ("cameraId", "trackId", "camera_ip", "boxCoords", "incidentType","thumbnail") 
                VALUES (%s, %s, %s, %s, %s,%s)
            """
            # self.db_cursor.execute(insert_query, (camera_id, track_id, camera_ip, bbox, incident_type,thumbnail))
            # self.db_connection.commit()
            print(f"Violation saved to DB: Track ID {track_id}")
        except Exception as e:
            print(f"Error saving violation to DB: {e}")
            self.db_connection.rollback()
    
    
    def close_db_connection(self):
        """Close the database connection gracefully."""
        self.db_cursor.close()
        self.db_connection.close()
    def is_point_in_polygon(self, point, polygon):
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def check_illegal_parking(self, track_id, cx, cy, label):
        inside_roi = False
        for roi_points in self.roi_area: 
            if self.is_point_in_polygon((cx, cy), np.array(roi_points,dtype=np.int32)):
                inside_roi = True
                break
        if inside_roi:
            if track_id not in self.static_objects:
                self.static_objects[track_id] = {"position": (cx, cy), "frames": 0, "violated": False, "label": label}
            else:
                last_position = self.static_objects[track_id]["position"]
                print(self.static_objects[track_id]["frames"])
                if abs(cx - last_position[0]) <= self.static_obj_offset and abs(cy - last_position[1]) <= self.static_obj_offset:
                    self.static_objects[track_id]["frames"] += 1
                    if self.static_objects[track_id]["frames"] > self.stationary_frame_threshold and not self.static_objects[track_id]["violated"]:
                        self.static_objects[track_id]["violated"] = True
                        self.total_parking_violations += 1
                        print(f"Object {track_id} marked as parking violation. Total Violations: {self.total_parking_violations}")
                else:
                    self.static_objects[track_id]["position"] = (cx, cy)
                    self.static_objects[track_id]["frames"] = 0
                    self.static_objects[track_id]["violated"] = False
    def detect_wrong_way_violation(self, track_id, cx, cy, label):
        if  track_id not in self.crossed_objects_wrong:
            self.crossed_objects_wrong[track_id] = {'red_crossed': False, 'green': set(), "label": label}

        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.ww_red_line):
            if min(y_start, y_end) - self.offset <= cy <= max(y_start, y_end) + self.offset:
                if min(x_start, x_end) <= cx <= max(x_start, x_end):
                    self.crossed_objects_wrong[track_id]['red_crossed'] = True
        if self.crossed_objects_wrong[track_id]['red_crossed']:
            for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.ww_green_line):
                if min(y_start, y_end) - self.offset <= cy <= max(y_start, y_end) + self.offset:
                    if min(x_start, x_end) <= cx <= max(x_start, x_end):
                        self.crossed_objects_wrong[track_id]['green'].add(i)

        if track_id not in self.violated_objects_wrong and self.crossed_objects_wrong[track_id]['green'] and self.crossed_objects_wrong[track_id]['red_crossed']:
            self.wrong_way_violation_count += 1
            self.violated_objects_wrong.add(track_id)


    def detect_traffic_violation(self, track_id, cx, cy, label):
        if track_id not in self.crossed_objects:
            self.crossed_objects[track_id] = {'red': set(), 'green': set(), "label": label}

        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.tv_red_line):
            if min(y_start, y_end) - self.red_light_line_offset <= cy <= max(y_start, y_end) + self.red_light_line_offset:
                if min(x_start, x_end) <= cx <= max(x_start, x_end):
                    self.crossed_objects[track_id]['red'].add(f"red_{i}")

        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.tv_green_line):
            if min(y_start, y_end) - self.red_light_line_offset <= cy <= max(y_start, y_end) + self.red_light_line_offset:
                if min(x_start, x_end) <= cx <= max(x_start, x_end):
                    if any(f"red_{j}" in self.crossed_objects[track_id]['red'] for j in range(len(self.tv_red_line))) and track_id not in self.violated_objects:
                        self.traffic_violation_count += 1
                        self.violated_objects.add(track_id)
    def process_image(self,cropped_images, model_path, track_id, cam_id, camera_ip):
        detection_keypoint = DetectKeypoint()
        classification_keypoint = KeypointClassification(model_path)
        images = cropped_images
        if images is None:
            raise ValueError(f"Image not found at the path: {cropped_images}")
        results = detection_keypoint(images)
        input_classifications = [result[10:] if result != None else [] for result in results]
        classification_results = classification_keypoint(input_classifications)
        return classification_results

    def draw_lines_for_traffic_violation(self, frame):
        for roi_points in self.roi_area:
            roi_points = np.array(roi_points,dtype=np.int32)
            cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=1)
            # cv2.putText(frame, "Illegal Parking Area", (roi_points[0][0],roi_points[0][1] - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.tv_red_line):
            cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)
            # cv2.putText(frame, f"Red {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)

        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.tv_green_line):
            cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
            # cv2.putText(frame, f"Green {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.ww_red_line):
            cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)
            # cv2.putText(frame, f"Red {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)

        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.ww_green_line):
            cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
            # cv2.putText(frame, f"Green {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

        # Display the violation counts
        # cv2.rectangle(frame, (0, 0), (350, 60), (0, 255, 255), -1)
        # cv2.putText(frame, f'Traffic Violations - {self.traffic_violation_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(frame, f'Parking Violations - {self.total_parking_violations}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(frame, f'Wrong Way Violations - {self.wrong_way_violation_count}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def process_license_plate(image, camera_ip, time):
    license_plate_text = read_license_plate(image, camera_ip, time[0])
    return license_plate_text

def process_lp_images(input_folder="/home/annone/ai-camera/ML_Backend/data/lp", camera_ip = "0.0.0.0"):
    while True:
        if len(os.listdir(input_folder)) > 0:
            file_name = os.listdir(input_folder)[0]
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(input_folder, file_name)
                print(f"Processing: {file_name}")
                img_f = cv2.imread(file_path)
                img = cv2.resize(img_f, (410, 190))
                time = file_name.split(".")[:-1]
                lp = process_license_plate(img, camera_ip, time)
                print(lp)
                os.remove(file_path)

def save_count(vehicle_count :int, crowd_count :int, timestamp):
    conn = Database.get_connection()
    cur = conn.cursor()
    insert_query = """
        INSERT INTO "CrowdCount" (count, timestamp, confidence, "vehicleCount", "cameraId")
        VALUES (%s, %s, %s, %s, %s)
        """
    cur.execute(insert_query, (crowd_count, timestamp,1, vehicle_count, os.getenv("CAM_ID")))
    conn.commit()
    Database.return_connection(connection=conn)