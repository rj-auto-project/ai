import torch
import cv2
import time
import torchvision.transforms as transforms
from PIL import Image
from timmML_025.models.factory import create_model
import pyautogui
from ultralytics import YOLO

corp_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

import warnings
warnings.filterwarnings("ignore")

# Initialize the EfficientNet model
model = create_model('efficientnet_lite0')  # This is modified lite0.
PATH_model = "D:/local-intelligent-cameras/backend/apis/effcc_crowd/student_025.pt"

# Load the state dictionary to the appropriate device
model.load_state_dict(torch.load(PATH_model, map_location=device))
model.to(device)  # Ensure the model is on the correct device
model.eval()

# Initialize YOLOv8 model
yolo_model = YOLO('D:/local-intelligent-cameras/backend/apis/effcc_crowd/best.pt')  # Adjust the path to your YOLOv8 model

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN_STD[0], MEAN_STD[1])
])

# Specify the video path
input_video_path = 'D:/local-intelligent-cameras/backend/apis/effcc_crowd/test.mp4'
output_video_path = 'D:/local-intelligent-cameras/backend/apis/effcc_crowd/output_video.avi'

# Open the video file
cap = cv2.VideoCapture(input_video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Define the video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


fps_list = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Convert frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize the image as needed
    if img.size[0] < corp_size * 2:
        img = img.resize((corp_size * 2, int(corp_size * 2 * img.size[1] / img.size[0])))
    if img.size[1] < corp_size * 2:
        img = img.resize((int(corp_size * 2 * img.size[0] / img.size[1]), corp_size * 2))

    kk = 14  # Adjust as per your model's requirement
    if img.size[0] > corp_size * kk:
        img = img.resize((corp_size * kk, int(corp_size * kk * img.size[1] / img.size[0])))
    if img.size[1] > corp_size * kk:
        img = img.resize((int(corp_size * kk * img.size[0] / img.size[1]), corp_size * kk))

    input_tensor = img_transform(img)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)  # Send to the appropriate device

    with torch.no_grad():
        output, features = model(input_batch)

    pred_count = torch.sum(output).item()

    # Object detection with YOLOv8
    results = yolo_model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_list.append(fps)

    # Display the frame with predicted count
    cv2.putText(annotated_frame, f'Predicted count: {int(pred_count)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
    
    # Write the annotated frame to the output video
    cv2.imshow("test", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Print average FPS
# average_fps = sum(fps_list) / len(fps_list)
# print(f'Average FPS: {average_fps:.2f}')
