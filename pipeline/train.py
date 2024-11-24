from ultralytics import YOLO

pretrained_weights = 'yolov8s.pt'
previous_weights = 'last.pt'

# pretrained weights
model = YOLO(pretrained_weights)
model = YOLO(previous_weights)

data_yaml = 'data.yaml'  # Path to your data YAML file containing training/validation dataset info
epochs = 100             # Number of training epochs
batch_size = 16          # Batch size for training
img_size = 640           # Input image size for the model

# Train the model
model.train(
    data=data_yaml,      # Data config file
    epochs=epochs,       # Number of epochs
    batch=batch_size,    # Batch size
    imgsz=img_size,      # Image size for training
    weights=previous_weights
)