from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
from timmML_025.models.factory import create_model
import asyncio
from sqlalchemy import create_engine, Column, Integer, String, Float, TIMESTAMP, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import pytz

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_current_time_ist():
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.datetime.now(ist)

corp_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Database setup
DATABASE_URL = "mysql+pymysql://root@localhost:3306/logs"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CrowdCount(Base):
    __tablename__ = 'crowd_count_logs'
    id = Column(Integer, primary_key=True, index=True)
    camera_ip = Column(String, index=True)
    timestamp = Column(TIMESTAMP, default=get_current_time_ist)
    crowd_count = Column(Integer)
    crowd_count_confidence_score = Column(Float)
    created_at = Column(TIMESTAMP, default=get_current_time_ist)
    modified_at = Column(TIMESTAMP, default=get_current_time_ist, onupdate=get_current_time_ist)
    deleted_at = Column(TIMESTAMP)

Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize the model
model = create_model('efficientnet_lite0')  # Adjust according to your model
PATH_model = "D:/local-intelligent-cameras/backend/apis/ccapi/student_025.pt"

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
        output_probs = torch.softmax(output, dim=1)
    
    pred_count = torch.sum(output).item()
    confidence_score = output_probs.max().item()  # Using the maximum softmax probability as confidence score
    return int(pred_count), confidence_score

# Function to process the video and send predicted counts over WebSocket
async def process_video(websocket: WebSocket, video_path: str, camera_ip: str, db):
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
        pred_count, confidence_score = predict_count(img, model)
        current_time_ist = get_current_time_ist()
        # Save predicted count to database
        crowd_count_entry = CrowdCount(
            camera_ip=camera_ip,
            timestamp=current_time_ist,
            crowd_count=pred_count,
            crowd_count_confidence_score=confidence_score,  # Placeholder for confidence score
            created_at=current_time_ist,
            modified_at=current_time_ist,
            deleted_at=None
        )
        db.add(crowd_count_entry)
        db.commit()

        # Send predicted count over WebSocket
        await websocket.send_text(str(pred_count))

    cap.release()

# WebSocket endpoint to process video and send predicted counts
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: SessionLocal = Depends(get_db)):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            video_path = data.get("video_path")
            camera_ip = data.get("camera_ip")

            # Process the video and send the counts
            await process_video(websocket, video_path, camera_ip, db)

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
