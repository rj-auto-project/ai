import cv2
from datetime import datetime
import uuid
# RTSP stream URL
rtsp_url = "rtmp://122.200.18.78:80/live"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream opened successfully
if not cap.isOpened():
    print("Error: Unable to open the RTSP stream.")
    exit()

# Get the width, height, and FPS of the stream
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(uuid.uuid4())
# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MJPG' or 'MP4V'
out = cv2.VideoWriter('/home/annone/ai/backend/stream/output.avi', fourcc, fps, (frame_width, frame_height))

print("Press 'q' to stop recording and save the video.")

# Loop to read frames from the stream and write them to the file
while True:
    timestamp = datetime.now()
    ret, frame = cap.read()

    if timestamp.minute % 30:
        cv2.imwrite(f"/home/annone/ai/backend/stream/frame/{uuid.uuid4()}.jpeg",frame)

    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Write the frame to the output file
    out.write(frame)

    # Display the current frame (optional)
    cv2.imshow('RTSP Stream', frame)

    # Press 'q' to quit and stop saving
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and VideoWriter objects and close windows
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved successfully.")
