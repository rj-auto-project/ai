import torch
import cv2
import time
import torchvision.transforms as transforms
from PIL import Image
from timmML_025.models.factory import create_model

corp_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

import warnings
warnings.filterwarnings("ignore")

# Initialize the model
model = create_model('efficientnet_lite0')  # This is modified lite0.
PATH_model = "D:/local-intelligent-cameras/backend/yolov5/ccapi/student_025.pt"

# Load the state dictionary to the appropriate device
model.load_state_dict(torch.load(PATH_model, map_location=device))
model.to(device)  # Ensure the model is on the correct device
model.eval()

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN_STD[0], MEAN_STD[1])
])

# Specify the video path (you can change this to your desired video)
video_path = 'D:/local-intelligent-cameras/backend/apis/anpr/test.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

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

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_list.append(fps)

    # Display the frame with predicted count
    cv2.putText(frame, f'Predicted count: {int(pred_count)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print average FPS
average_fps = sum(fps_list) / len(fps_list)
print(f'Average FPS: {average_fps:.2f}')
import torch

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")
