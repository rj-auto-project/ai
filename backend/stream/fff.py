import cv2
import psycopg2
import numpy as np
from collections import deque
from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification


class ViolationDetector:
    def __init__(self):  
        self.offset = 10
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
        self.static_obj = {}
        self.obj_static_frames = 3  
        self.static_objects = {}
        self.stationary_frame_threshold = 200
        self.cam_ip = '127.0.0.1'
        self.cam_id = "1"
        self.INCIDENT_TYPES = {
                        "TRAFFIC_VIOLATION": "REDLIGHT_VIOLATION",
                        "ILLEGAL_PARKING": "ILLEGAL_PARKING",
                        "WRONG_WAY": "WRONG_WAY_DRIVING"
                        }
         # Define your ROI points and lines
        self.roi_points = np.array([[0, 0], [1280, 0], [1280, 960], [0, 960]], dtype=np.int32)
        self.ww_red_line = [[(266, 502), (876, 502)]]
        self.ww_green_line = [[(88, 780), (1050, 780)]]
        
        
        # Database connection setup
        self.db_connection = psycopg2.connect(
            host='34.47.148.81',
            database='logs',
            user="root",
            password="team123",
            port='8080'
        )
        self.db_cursor = self.db_connection.cursor()   
    
    def save_violation_to_db(self, camera_id, track_id, camera_ip, bbox, incident_type):
        """Save violation details to the PostgreSQL database."""
        try:
            insert_query = """
                INSERT INTO "IncidentLogs" ("cameraId", "trackId", "camera_ip", "boxCoords", "incidentType") 
                VALUES (%s, %s, %s, %s, %s)
            """
            self.db_cursor.execute(insert_query, (camera_id, track_id, camera_ip, bbox, incident_type))
            self.db_connection.commit()
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
    
    def check_static(self, track_id, cx, cy, label):
        # Initialize tracking information if not already present
        if track_id not in self.static_obj:
            self.static_obj[track_id] = {
                "positions": [(None, None), (None, None), (None, None)],  # Start with None values
                "frames": 0,
                "violated": False,
                "label": label
            }
    
        position_history = self.static_obj[track_id]["positions"]
    
        # Shift positions and insert the new position
        position_history[2] = position_history[1]
        position_history[1] = position_history[0]
        position_history[0] = (cx, cy)
    
        # Check if the last three positions are valid for comparison
        if None not in position_history[0] and None not in position_history[1] and None not in position_history[2]:
            # Check if the object has remained stationary for the last 3 frames (within the offset)
            stationary = True
            for i in range(2):  # Check for x and y coordinates
                if (abs(position_history[0][i] - position_history[1][i]) > self.offset or
                    abs(position_history[0][i] - position_history[2][i]) > self.offset):
                    stationary = False
                    break
                
            if stationary:
                self.static_obj[track_id]["frames"] += 1
                if self.static_obj[track_id]["frames"] > self.obj_static_frames and not self.static_obj[track_id]["violated"]:
                    self.static_obj[track_id]["violated"] = True  # Mark the object as a violation
                    print(f"Object {track_id} marked as parking violation.")
            else:
                # If the object has moved, reset the frame count and violation status
                self.static_obj[track_id]["frames"] = 0
                self.static_obj[track_id]["violated"] = False
        else:
            print(f"Position history for track_id {track_id} contains None values. Current positions: {position_history}")
    



    
    # using frame to call person is static_object 
    # def check_static_person(self, track_id, cx, cy, label):
    
    #     # Initialize tracking information if not already present
    #     if track_id not in self.static_obj:
    #         # Store the position history for the object (last 3 frames)
    #         self.static_obj[track_id] = {
    #             "positions": [(cx, cy)], 
    #             "frames": 0, 
    #             "violated": False, 
    #             "label": label, 
    #             "image_count": 0  # Counter for number of images saved
    #         }

    #     position_history = self.static_obj[track_id]["positions"]

    #     # Append the new position to the history, keeping only the last 3 frames
    #     if len(position_history) >= 3:
    #         position_history.pop(0)  # Remove the oldest position if more than 3
    #     position_history.append((cx, cy))

    #     # Check if the object has remained stationary for the last 3 frames (within the offset)
    #     if all(abs(cx - pos[0]) <= self.offset and abs(cy - pos[1]) <= self.offset for pos in position_history):
    #         self.static_obj[track_id]["frames"] += 1
    #         if self.static_obj[track_id]["frames"] > self.obj_static_frames and not self.static_obj[track_id]["violated"]:
    #             self.static_obj[track_id]["violated"] = True  # Mark the object as a violation
    #             print(f"Object {track_id} marked as parking violation.")
    #     else:
    #         # If the object has moved, reset the frame count and violation status
    #         self.static_obj[track_id]["frames"] = 0
    #         self.static_obj[track_id]["violated"] = False



    def check_illegal_parking(self, track_id, cx, cy, label):
        if not self.is_point_in_polygon((cx, cy), self.roi_points):
            return
        
        if track_id not in self.static_objects:
            self.static_objects[track_id] = {"position": (cx, cy), "frames": 0, "violated": False, "label": label}
        else:
            last_position = self.static_objects[track_id]["position"]
            if abs(cx - last_position[0]) <= self.offset and abs(cy - last_position[1]) <= self.offset:
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
            self.crossed_objects_wrong[track_id] = {'red': set(), 'green': set(), "label": label}

        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.ww_red_line):
            if min(y_start, y_end) - self.offset <= cy <= max(y_start, y_end) + self.offset:
                if min(x_start, x_end) <= cx <= max(x_start, x_end):
                    self.crossed_objects_wrong[track_id]['red'].add(i)

        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.ww_green_line):
            if min(y_start, y_end) - self.offset <= cy <= max(y_start, y_end) + self.offset:
                if min(x_start, x_end) <= cx <= max(x_start, x_end):
                    self.crossed_objects_wrong[track_id]['green'].add(i)

        if any(
            i in self.crossed_objects_wrong[track_id]['green'] and
            i in self.crossed_objects_wrong[track_id]['red'] and
            track_id not in self.violated_objects_wrong
            for i in self.crossed_objects_wrong[track_id]['green']
        ):
            self.wrong_way_violation_count += 1
            self.violated_objects_wrong.add(track_id)

    def detect_traffic_violation(self, track_id, cx, cy, label):
        if track_id not in self.crossed_objects:
            self.crossed_objects[track_id] = {'red': set(), 'green': set(), "label": label}

        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.ww_red_line):
            if min(y_start, y_end) - self.offset <= cy <= max(y_start, y_end) + self.offset:
                if min(x_start, x_end) <= cx <= max(x_start, x_end):
                    self.crossed_objects[track_id]['red'].add(f"red_{i}")

        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.ww_green_line):
            if min(y_start, y_end) - self.offset <= cy <= max(y_start, y_end) + self.offset:
                if min(x_start, x_end) <= cx <= max(x_start, x_end):
                    if any(f"red_{j}" in self.crossed_objects[track_id]['red'] for j in range(len(self.ww_red_line))) and track_id not in self.violated_objects:
                        self.traffic_violation_count += 1
                        self.violated_objects.add(track_id)  
    def process_image(self,cropped_image, model_path, track_id, cam_id, camera_ip):

        # Initialize the detection and classification models
        detection_keypoint = DetectKeypoint()
        classification_keypoint = KeypointClassification(model_path)

        # Read the image
        image = cv2.imread(cropped_image)
        if image is None:
            raise ValueError(f"Image not found at the path: {cropped_image}")

        # Detect keypoints
        results = detection_keypoint(image)
        results_keypoint = detection_keypoint.get_xy_keypoint(results)

        # Prepare input for classification
        input_classification = results_keypoint[10:]  # Adjust based on your needs
        results_classification = classification_keypoint(input_classification)

        # Visualize keypoints
        image_draw = results.plot(boxes=False)

        # Draw bounding box and classification label
        if results.boxes.xyxy.size(0) > 0:  # Check if there are any detected boxes
            x_min, y_min, x_max, y_max = results.boxes.xyxy[0].numpy()
            bbox = f"{x_min}, {y_min}, {x_max}, {y_max}"
            cv2.rectangle(image_draw, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

            label = results_classification.upper()
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image_draw, (int(x_min), int(y_min) - 20), (int(x_min) + w, int(y_min)), (0, 0, 255), -1)
            cv2.putText(image_draw, label, (int(x_min), int(y_min) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)

            incident_type = results_classification
            self.save_violation_to_db(cam_id, track_id, camera_ip, bbox, incident_type)
            
        # Print classification result
        print(f'Keypoint classification: {results_classification}')

        # Show the image
        cv2.imshow("frame", image_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()                    
                        
    def draw_lines_for_traffic_violation(self, frame, total_parking_violations):
        cv2.polylines(frame, [self.roi_points], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, "Illegal Parking Area", (self.roi_points[0][0], self.roi_points[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.ww_red_line):
            cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
            cv2.putText(frame, f"Red {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        for i, ((x_start, y_start), (x_end, y_end)) in enumerate(self.ww_green_line):
            cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(frame, f"Green {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the violation counts
        cv2.rectangle(frame, (0, 0), (350, 60), (0, 255, 255), -1)
        cv2.putText(frame, f'Traffic Violations - {self.traffic_violation_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Parking Violations - {self.total_parking_violations}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Wrong Way Violations - {self.wrong_way_violation_count}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
