from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/home/annone/ai/models/yolov8n.pt")

# Export the model to TorchScript format
model.export(format="torchscript")  # creates 'yolov8n.torchscript'

# Load the exported TorchScript model
torchscript_model = YOLO("/home/annone/ai/models/yolov8n.torchscript")

# Run inference
results = torchscript_model("/home/annone/ai/bus.jpg")
print(results)