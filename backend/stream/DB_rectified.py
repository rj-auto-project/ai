import cv2
from ultralytics import YOLO
import psycopg2
from psycopg2 import pool
from datetime import datetime

# Define PostgreSQL connection parameters
DB_HOST = '34.47.148.81'
DB_NAME = 'logs'
DB_USER = 'root'
DB_PASS = 'team123'
DB_PORT = '8080'

# Connection pool for PostgreSQL
connection_pool = psycopg2.pool.SimpleConnectionPool(1, 10, host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)
class_names = ['fire', 'GARBAGE', 'POTHOLE', 'smoke', 'vehicle pollution', 'WATERLOGGING']
def get_db_connection():
    """Get a connection from the pool."""
    try:
        return connection_pool.getconn()
    except Exception as e:
        print(f"Error getting connection from pool: {e}")
        return None

def release_db_connection(conn):
    """Release a connection back to the pool."""
    if conn:
        connection_pool.putconn(conn)



def save_detections_to_db(trackId, metaCoords,boxCoords, incidentType, cameraId, camera_ip):
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO "IncidentLogs" ("trackId", "metaCoords", "boxCoords", "incidentType", "cameraId", "camera_ip")
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (trackId, metaCoords,boxCoords, incidentType, cameraId, camera_ip,))
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"Error saving detections to database: {e}")
    finally:
        release_db_connection(conn)

def update_detection_alert(trackId, metaCoords,boxCoords, incidentType, cameraId, camera_ip):
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO "IncidentLogs" ("trackId", "metaCoords", "boxCoords", "incidentType", "cameraId", "camera_ip")
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (trackId, metaCoords,boxCoords, incidentType, cameraId, camera_ip,))
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"Error saving detections to database: {e}")
    finally:
        release_db_connection(conn)

def check_existing_detection(issue, midpoint, camera_id, camera_ip,conf_threshold=0.5, threshold=20):
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        mid_x, mid_y = midpoint

        # SQL query to fetch all matching records with less or more than the midpoint threshold
        cursor.execute('''
    SELECT * FROM "IncidentLogs" 
    WHERE "incidentType" = %s 
    AND (SQRT(POWER("metaCoords"[1] - %s, 2) + POWER("metaCoords"[2] - %s, 2)) <= %s
    OR SQRT(POWER("metaCoords"[1] - %s, 2) + POWER("metaCoords"[2] - %s, 2)) >= %s)
    AND "cameraId" = %s
    AND camera_ip = %s
''', (issue, mid_x, mid_y, threshold, mid_x, mid_y, threshold, camera_id, camera_ip,))

        # Fetch all matching rows
        matching_detections = cursor.fetchall()
        cursor.close()
        return matching_detections
    except Exception as e:
        print(f"Error checking for existing detection: {e}")
        return []
    finally:
        release_db_connection(conn)

def find_midpoint(bbox):
    # Unpack the bounding box coordinates
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate the midpoint
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    
    return [x_mid, y_mid]

def detect_municipal_issues(image_path, model_path, camera_id, camera_ip, conf_threshold=0.5):
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    results = model(img, conf=conf_threshold)
    for result in results:
        for box in result.boxes:
            class_id = box.cls.item()
            confidence = box.conf.item()
            bbox = box.xyxy.tolist()[0]
            midpoint = find_midpoint(bbox)
            # print(class_id)
            issue = class_names[int(class_id)]

            # Check if the detection already exists in the database
            if not check_existing_detection(issue, midpoint, camera_id, camera_ip, conf_threshold):
                detection_time = datetime.now()  # Current timestamp
                trackId = confidence  # Assuming confidence as trackId, modify if necessary
                save_detections_to_db(trackId, midpoint,f"{[bbox]}", str(issue).upper()[:-1], camera_id, camera_ip)
                print("true")
            else:
                print("pothhole already exist")

    print("Detections processed.")

# Example usage
if __name__ == "__main__":
    image_path = "/home/annone/Downloads/images.jpeg"
    model_path = '/home/annone/ai/models/Municipal_issues.pt'
    camera_id = '1'
    camera_ip = '127.0.0.1'

    # # Create database table if it doesn't exist
    # create_database_table()

    # # Perform detection and save detections
    detected_issues = detect_municipal_issues(image_path, model_path, camera_id, camera_ip)

    # if detected_issues:
    #     print(f"{len(detected_issues)} new detections saved.")
