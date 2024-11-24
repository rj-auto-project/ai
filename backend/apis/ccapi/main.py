from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
from timmML_025.models.factory import create_model
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

corp_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Initialize the model
model = create_model('efficientnet_lite0')  # Adjust according to your model
PATH_model = "student_025.pt"

# Load the state dictionary to the appropriate device
model.load_state_dict(torch.load(PATH_model, map_location=device))
model.to(device)  # Ensure the model is on the correct device
model.eval()

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN_STD[0], MEAN_STD[1])
])

# Function to resize image
def resize_image(img):
    if img.size[0] < corp_size * 2:
        img = img.resize((corp_size * 2, int(corp_size * 2 * img.size[1] / img.size[0])))
    if img.size[1] < corp_size * 2:
        img = img.resize((int(corp_size * 2 * img.size[0] / img.size[1]), corp_size * 2))

    kk = 14  # Adjust as per your model's requirement
    if img.size[0] > corp_size * kk:
        img = img.resize((corp_size * kk, int(corp_size * kk * img.size[1] / img.size[0])))
    if img.size[1] > corp_size * kk:
        img = img.resize((int(corp_size * kk * img.size[0] / img.size[1]), corp_size * kk))

    return img

# Function to predict count for an image
def predict_count(image, model):
    input_tensor = img_transform(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    with torch.no_grad():
        output, features = model(input_batch)

    pred_count = torch.sum(output).item()
    return int(pred_count)

# Function to process the video and send predicted counts over WebSocket
async def process_video(websocket: WebSocket, video_path: str):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        img = Image.fromarray(rgb_frame)
        img = resize_image(img)
        pred_count = predict_count(img, model)

        # Send predicted count over WebSocket
        await websocket.send_text(str(pred_count))

    cap.release()

# WebSocket endpoint to process video and send predicted counts
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            video_path = data  # The client sends the video path

            # Process the video and send the counts
            await process_video(websocket, video_path)

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
