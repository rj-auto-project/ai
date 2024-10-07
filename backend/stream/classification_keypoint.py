import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size=24, hidden_size=256, num_classes=2):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

class KeypointClassification:
    def __init__(self, path_model):
        self.path_model = path_model
        self.classes = ['pee', 'spit']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        self.model = NeuralNet().to(self.device)  # Move the model to GPU
        self.model.load_state_dict(
            torch.load(self.path_model, map_location=self.device)
        )
        self.model.eval()  # Set the model to evaluation mode

    def __call__(self, input_keypoints):
        # Ensure input_keypoints is a tensor and move to the device
        if not isinstance(input_keypoints, torch.Tensor):
            input_keypoints = torch.tensor(input_keypoints, dtype=torch.float32)
        
        input_keypoints = input_keypoints.to(self.device)  # Move to GPU
        
        with torch.no_grad():  # Disable gradient calculation
            out = self.model(input_keypoints)  # Forward pass
            _, predictions = torch.max(out, dim=1)  # Get predicted class indices
            label_predictions = [self.classes[pred.item()] for pred in predictions]  # Convert to class labels
        return label_predictions

if __name__ == '__main__':
    keypoint_classification = KeypointClassification(
        path_model='C:/Users/Laptop/Desktop/YoloV8-Pose-Keypoint-Classification-master/model/spit_pose_classification.pth'
    )
    # Generate a batch of dummy inputs (e.g., batch size of 5)
    batch_size = 5
    dummy_input = torch.randn(batch_size, 24)  # Shape: (batch_size, input_size)
    classification = keypoint_classification(dummy_input)
    print(classification)
