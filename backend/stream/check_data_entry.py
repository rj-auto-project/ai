import cv2
import numpy as np
import psycopg2
from datetime import datetime
from sort.sort import Sort  # Import the SORT tracker
from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification

# Initialize the SORT tracker
tracker = Sort()

def save_to_database(cam_id, track_id, camera_ip, bbox, incident_type):
    # Connect to PostgreSQL database
    conn = psycopg2.connect(
        dbname='logs',  # Replace with your database name
        user='root',    # Replace with your username
        password='team123', # Replace with your password
        host='34.47.148.81',        # Adjust host if necessary
        port='8080'
    )
    cursor = conn.cursor()

    # Insert data into the table
    cursor.execute(''' 
        INSERT INTO "IncidentLogs" ( "cameraId", "trackId", "camera_ip", "boxCoords", "incidentType") 
        VALUES (%s, %s, %s, %s, %s)
    ''', ( cam_id, track_id, camera_ip, bbox, incident_type))

    # Commit and close the connection
    conn.commit()
    cursor.close()
    conn.close()

def process_image(image_path, model_path, cam_id, camera_ip):
    # Initialize the detection and classification models
    detection_keypoint = DetectKeypoint()
    classification_keypoint = KeypointClassification(model_path)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at the path: {image_path}")

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
        box_coords = [x_min, y_min, x_max, y_max, 1.0]  # Add a dummy confidence score

        # Update the SORT tracker with the detected bounding box
        box_coords = np.array([box_coords])  # Convert to a 2D array
        tracked_objects = tracker.update(box_coords)

        for obj in tracked_objects:
            track_id = int(obj[4])  # Retrieve the track ID from SORT
            cv2.rectangle(image_draw, (int(obj[0]), int(obj[1])), (int(obj[2]), int(obj[3])), (0, 0, 255), 2)

            label = f"{results_classification.upper()} (ID: {track_id})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image_draw, (int(obj[0]), int(obj[1]) - 20), (int(obj[0]) + w, int(obj[1])), (0, 0, 255), -1)
            cv2.putText(image_draw, label, (int(obj[0]), int(obj[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)

            # Get the current timestamp
            
            incident_type = results_classification  # Adjust this based on your needs
     

            # Save results to the database
            save_to_database(cam_id,  track_id, camera_ip, bbox, incident_type)

    # Print classification result
    print(f'Keypoint classification: {results_classification}')

    # Show the image
    cv2.imshow("frame", image_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "C:/Users/Laptop/Desktop/YoloV8-Pose-Keypoint-Classification-master/test/peeeing in public1.jpg"
    model_path = 'C:/Users/Laptop/Desktop/YoloV8-Pose-Keypoint-Classification-master/model/pee_spit.pth'
    cam_id = "1"  # Example camera ID
    camera_ip = "24.67.98.23"  # Add your camera IP
    process_image(image_path, model_path, cam_id, camera_ip)
