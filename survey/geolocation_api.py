from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
import uuid
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from datetime import datetime, timedelta
from db import Database
from pydantic import BaseModel
import math


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the YOLO model
MODEL_PATH = '/home/annone/ai/survey/data/weights/best.pt'  # Update path
model = YOLO(MODEL_PATH)

# JSON data file
JSON_DATA_FILE = "/home/annone/ai/survey/temp_database.json"
coordinate_text_file = '/home/annone/ai/survey/coordinates_with_timestamp.txt'
OUTPUT_IMAGE_DIR = "/home/annone/ai/survey/static/images/"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
conn = Database.get_connection()
cursor = conn.cursor()
# Helper: Read and write to JSON file
def log_into_json(file_path, data):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(data)

    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)

# save into postgres
def save_to_postgres(thumbnail, class_name, survey_id, geo_coords, distance ):
    query = """
    INSERT INTO "SurveyReport" ("thumbnail", "className", "surveyId", "location", "distance")
    VALUES (%s, %s, %s, %s, %s);
    """
    try:
        print(type(geo_coords))
        geo_coords = json.dumps([geo_coords[0],geo_coords[1]])
        data = (f"{thumbnail}", f"{class_name}", f"{survey_id}", geo_coords, f"{distance}")
        cursor.execute(query, data)
        conn.commit()
        print("Data saved successfully.")

    except Exception as e:
        print(f"Error saving data: {e}")

# Helper: Load coordinates with timestamps
def load_coordinates(coordinate_text_file):
    coordinates = []
    with open(coordinate_text_file, 'r') as f:
        for line in f:
            lat, lon, time_str = [i.split(": ")[1] for i in line.strip().split(',')]
            coordinates.append({
                "timestamp": datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f"),
                "latitude": float(lat),
                "longitude": float(lon)
            })
    return coordinates

# calculate distance between two geo coords
def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.

    Parameters:
        coord1 (tuple): (latitude, longitude) of the first location in decimal degrees.
        coord2 (tuple): (latitude, longitude) of the second location in decimal degrees.

    Returns:
        float: Distance between the two coordinates in kilometers.
    """
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    # Compute differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

# Example usage
# point1 = (26.9150048, 75.7416909)  # Latitude and longitude of point 1
# point2 = (26.9123942, 75.7882451)  # Latitude and longitude of point 2

# Function to generate unique file names
def get_next_file_name(base_path):
    i = 1
    while os.path.exists(f"./{base_path}{i}.txt"):
        i += 1
    return f"{base_path}{i}.txt"

# File base path to store the coordinates
base_path = "coordinates"

# WebSocket handler to receive location data
@app.websocket("/ws")   
async def websocket_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection
    await websocket.accept()
    
    # Generate a new file for this session
    current_file = get_next_file_name(base_path)
    print(f"Writing coordinates to: {current_file}")

    try:
        while True:
            # Receive data from the WebSocket
            data = await websocket.receive_text()
            if not data.strip():
                await websocket.send_text("Empty data received.")
                continue
            try:
                location_data = json.loads(data)  # Attempt to parse JSON
            except json.JSONDecodeError:
                await websocket.send_text("Invalid JSON format.")
                continue
            # Parse the received JSON data
            location_data = json.loads(data)
            for loc in location_data:
                latitude = loc.get("latitude")
                longitude = loc.get("longitude")
                timestamp = loc.get("timestamp")
                # Write the coordinates to the current file
                with open(current_file, "a") as file:
                    file.write(f"Latitude: {latitude}, Longitude: {longitude}, Timestamp: {timestamp}\n")
            # Send confirmation back to the client
            await websocket.send_text(f"Coordinates received and written to {current_file}")
    
    except WebSocketDisconnect:
        print("Client disconnected")


# Route to process video
class FilePaths(BaseModel):
    text_file: str
    video_file: str

@app.post("/process-video/")
async def process_video(paths: FilePaths):
    text_file_path = paths.text_file
    coordinate_text_file = text_file_path
    input_video_path = paths.video_file
    output_video_path = input_video_path.replace(".mp4", "_processed.mp4")

    try:
        # Extract file modified time
        file_modified_time = os.path.getmtime(input_video_path)
        initial_time = datetime.fromtimestamp(file_modified_time)

        # Load coordinates
        coordinates = load_coordinates(coordinate_text_file)

        # Open video file
        cap = cv2.VideoCapture(input_video_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        starting_lat = coordinates[0]["latitude"]
        starting_long =  coordinates[0]["longitude"]
        # Initialize SORT tracker
        tracker = Sort()
        known_track_id = []
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate frame timestamp            
            frame_time = initial_time + timedelta(seconds=frame_index / fps)

            # Run YOLO model
            results = model(frame)
            class_list = model.names

            # Prepare detections for SORT
            detections = []
            for detection in results[0].boxes:
                confidence = detection.conf.item()
                class_id = int(detection.cls.item())
                class_name = model.names[class_id]

                if confidence < 0.3 or class_name == "drainage":
                    continue

                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                detections.append([x1, y1, x2, y2, confidence, class_id])

            detections = np.array(detections) if len(detections) > 0 else np.empty((0, 6))

            # Update tracker
            tracked_objects = tracker.update(detections)

            # Annotate frame
            annotated_frame = frame.copy()

            for obj in tracked_objects:
                x1, y1, x2, y2, track_id, class_id = map(int, obj[:6])
                label = f"ID {track_id}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                nearest_coord = min(coordinates, key=lambda coord: abs(coord["timestamp"] - frame_time))
                lat, lon = nearest_coord["latitude"], nearest_coord["longitude"]
                distance = haversine((starting_lat, starting_long),(lat,lon))
                if track_id not in known_track_id:
                    json_data = {
                        "detection_id": str(uuid.uuid4()),
                        "thumbnail": f"{track_id}.jpg",
                        "class_name": f"{class_list[class_id]}",
                        "location": [lat, lon],
                        "survey_id": "1",
                        "distance": distance,
                        "track_id": str(track_id)
                    }
                    log_into_json(JSON_DATA_FILE, json_data)
                    cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, f"{track_id}.jpg"), annotated_frame)
                    known_track_id.append(track_id)

            out.write(annotated_frame)
            frame_index += 1

        # Release resources
        cap.release()
        out.release()

        return JSONResponse(content={"message": "Video processed successfully", "output_video": output_video_path})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    # finally:
        # if os.path.exists(input_video_path):
        #     os.remove(input_video_path)


@app.post("/save-to-db/")
def save_to_db():
    file_path = "/home/annone/ai/survey/temp_database.json"
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                file_data = json.load(file)  # Load JSON data

            # Keep track of rows that fail to save
            unsaved_rows = []

            for data in file_data:
                try:
                    thumbnail = data['thumbnail']
                    class_name = data['class_name']
                    location = data['location']
                    survey_id = data['survey_id']
                    distance = "100"

                    # Save to database
                    save_to_postgres(
                        thumbnail=thumbnail, 
                        class_name=class_name, 
                        survey_id=survey_id, 
                        geo_coords=location, 
                        distance=distance
                    )
                except Exception as e:
                    print(f"Error saving to database: {e}")
                    unsaved_rows.append(data)  # Keep track of unsaved rows

            # Overwrite JSON file with only unsaved rows
            with open(file_path, 'w') as file:
                json.dump(unsaved_rows, file, indent=4)

            print("Database save complete. Unsaved rows (if any) retained in the JSON file.")

        except json.JSONDecodeError:
            print(f"Error: The file {file_path} is empty or contains invalid JSON.")
    else:
        print(f"Error: The file {file_path} does not exist.")
    
# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5346)
