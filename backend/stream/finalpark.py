import cv2
import os
import torch
from ultralytics import YOLO
import time

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define global variables
count = 0
static_objects = {}  # To track objects that are stationary
stationary_frame_threshold = 200  # Number of frames an object should be stationary to trigger a parking violation
total_parking_violations = 0  # Total count of parking violations

def setup_model(model_path):
    # Load the YOLO model and move it to the appropriate device (GPU if available, otherwise CPU)
    model = YOLO(model_path)
    model.to(device)
    return model

def setup_video(video_path, output_path):
    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None, None, None
    
    # Set up the video writer for saving the processed video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (1020, 500))
    return cap, out, fps

def create_frame_folder(folder_name):
    # Create a folder to save individual frames if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def process_frame(frame, model, offset=10):
    global static_objects, total_parking_violations
    frame = cv2.resize(frame, (1020, 500))  # Resize the frame to the desired dimensions
    results = model.track(frame, persist=True, device=device)  # Perform object detection and tracking

    # Iterate over the results for each object detected
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get the bounding box coordinates
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []

        # Iterate over each detected object
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)  # Extract bounding box coordinates
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Calculate the center of the bounding box

            # If the object is seen for the first time, initialize its tracking information
            if track_id not in static_objects:
                static_objects[track_id] = {"position": (cx, cy), "frames": 0, "violated": False}
            else:
                last_position = static_objects[track_id]["position"]
 
                # Check if the object has remained stationary (within a certain offset)
                if abs(cx - last_position[0]) <= offset and abs(cy - last_position[1]) <= offset:
                    static_objects[track_id]["frames"] += 1
                    # If the object is stationary for more than the threshold and not already marked as a violation
                    if static_objects[track_id]["frames"] > stationary_frame_threshold and not static_objects[track_id]["violated"]:
                        static_objects[track_id]["violated"] = True  # Mark the object as a violation
                        total_parking_violations += 1  # Increase the parking violation count
                        print(f"Object {track_id} marked as parking violation. Total Violations: {total_parking_violations}")
                else:
                    # If the object has moved, reset its tracking information
                    static_objects[track_id]["position"] = (cx, cy)
                    static_objects[track_id]["frames"] = 0
                    static_objects[track_id]["violated"] = False

            # Draw bounding box: Red for violated objects, Green otherwise
            if static_objects[track_id]["violated"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Parking Violation", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

def draw_parking_violation_text(frame, total_violations):
    # Display the total number of parking violations on the frame
    cv2.rectangle(frame, (0, 0), (250, 90), (0, 255, 255), -1)
    cv2.putText(frame, f'Parking Violations - {total_violations}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def save_frame(frame, folder_name, count):
    # Save the processed frame to the specified folder
    frame_filename = f'{folder_name}/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)

def run_detection(video_path, output_path, model):
    global count, total_parking_violations
    cap, out, fps = setup_video(video_path, output_path)  # Set up video input and output
    if cap is None or out is None:
        return

    create_frame_folder('test_frames')  # Create a folder to save frames

    while cap.isOpened():
        ret, ori_frame = cap.read()
        if not ret:
            break
        count += 1

        # Process the frame for object detection and tracking
        frame = process_frame(ori_frame, model)
        draw_parking_violation_text(frame, total_parking_violations)  # Display total violations
        save_frame(frame, 'test_frames', count)  # Save the frame

        out.write(frame)  # Write the frame to the output video
        cv2.imshow("RGB", frame)  # Display the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = 'I:/RJ/models/seg.pt'
    video_path = "I:/RJ/test_videos/traffic_light.mp4"
    output_path = 'I:/RJ/output/park.mp4'

    model = setup_model(model_path)  # Load the YOLO model
    run_detection(video_path, output_path, model)  # Start the detection and tracking process
