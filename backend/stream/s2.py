import cv2
import zmq
import pickle
import numpy as np

# Set up ZeroMQ subscriber
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")  # Use the IP and port from Script 1
socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all messages

# Function to detect color (e.g., red, green, blue)
def detect_color(frame, box):
    x1, y1, x2, y2 = box
    region = frame[y1:y2, x1:x2]

    # Convert the region to HSV color space for easier color detection
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV (example for red, adjust as needed)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv_region, lower_red, upper_red)

    # Check if there’s a significant red presence in the region
    if cv2.countNonZero(mask_red) > 0:
        return "Red"
    return "Color not detected"  # Or "None" if no specific color is detected

while True:
    # Receive and deserialize data
    data = pickle.loads(socket.recv())
    frame = data["frame"]
    detections = data["detections"]

    # Process each detection for color detection
    for detection in detections:
        obj_class = detection["class"]
        box = detection["box"]

        # Perform color detection within the bounding box
        detected_color = detect_color(frame, box)
        
        # Annotate the frame with the detected color and object class
        x1, y1, x2, y2 = box
        label = f"{obj_class}: {detected_color}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow("Color Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
