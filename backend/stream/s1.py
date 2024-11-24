import cv2
from ultralytics import YOLO
import zmq
import pickle

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Set up ZeroMQ publisher
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")  # Use any available port

cap = cv2.VideoCapture(0)  # Replace 0 with your video source if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Prepare data for color detection
    data = {
        "frame": frame,
        "detections": []
    }
    
    # Extract bounding boxes and class labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Coordinates of bounding box
            cls = result.names[int(box.cls)]  # Detected class name
            data["detections"].append({"class": cls, "box": (x1, y1, x2, y2)})

    # Serialize and send frame with detection data
    socket.send(pickle.dumps(data))

cap.release()
