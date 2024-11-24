import sys
import cv2
import numpy as np
from pydantic import BaseModel
import torchvision.transforms as transforms
import ultralytics
from ultralytics.engine.results import Results
from PIL import Image 
import torch
import time
# Define keypoint
class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

class DetectKeypoint:
    def __init__(self, yolov8_model='/home/annone/ai/models/yolov8n-pose.pt', image_size=(320, 192)):
        self.yolov8_model = yolov8_model
        self.image_size = image_size
        self.get_keypoint = GetKeypoint()
        self.__load_model()

    def __load_model(self):
        # if not self.yolov8_model.split('-')[-1] == 'pose':
        #     sys.exit('Model not yolov8 pose')
        self.model = ultralytics.YOLO(model=self.yolov8_model)

        # extract function keypoint
    def extract_keypoint(self, keypoint: np.ndarray) -> list:
        # nose
        nose_x, nose_y = keypoint[self.get_keypoint.NOSE]
        # eye
        left_eye_x, left_eye_y = keypoint[self.get_keypoint.LEFT_EYE]
        right_eye_x, right_eye_y = keypoint[self.get_keypoint.RIGHT_EYE]
        # ear
        left_ear_x, left_ear_y = keypoint[self.get_keypoint.LEFT_EAR]
        right_ear_x, right_ear_y = keypoint[self.get_keypoint.RIGHT_EAR]
        # shoulder
        left_shoulder_x, left_shoulder_y = keypoint[self.get_keypoint.LEFT_SHOULDER]
        right_shoulder_x, right_shoulder_y = keypoint[self.get_keypoint.RIGHT_SHOULDER]
        # elbow
        left_elbow_x, left_elbow_y = keypoint[self.get_keypoint.LEFT_ELBOW]
        right_elbow_x, right_elbow_y = keypoint[self.get_keypoint.RIGHT_ELBOW]
        # wrist
        left_wrist_x, left_wrist_y = keypoint[self.get_keypoint.LEFT_WRIST]
        right_wrist_x, right_wrist_y = keypoint[self.get_keypoint.RIGHT_WRIST]
        # hip
        left_hip_x, left_hip_y = keypoint[self.get_keypoint.LEFT_HIP]
        right_hip_x, right_hip_y = keypoint[self.get_keypoint.RIGHT_HIP]
        # knee
        left_knee_x, left_knee_y = keypoint[self.get_keypoint.LEFT_KNEE]
        right_knee_x, right_knee_y = keypoint[self.get_keypoint.RIGHT_KNEE]
        # ankle
        left_ankle_x, left_ankle_y = keypoint[self.get_keypoint.LEFT_ANKLE]
        right_ankle_x, right_ankle_y = keypoint[self.get_keypoint.RIGHT_ANKLE]
        
        return [
            nose_x, nose_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y,
            left_ear_x, left_ear_y, right_ear_x, right_ear_y, left_shoulder_x, left_shoulder_y,
            right_shoulder_x, right_shoulder_y, left_elbow_x, left_elbow_y, right_elbow_x, right_elbow_y,
            left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y, left_hip_x, left_hip_y,
            right_hip_x, right_hip_y, left_knee_x, left_knee_y, right_knee_x, right_knee_y,        
            left_ankle_x, left_ankle_y,right_ankle_x, right_ankle_y
        ]
    
    def get_xy_keypoint(self, results: Results) -> list:
        if hasattr(results, 'names') and results.keypoints is not None and results.keypoints.xyn is not None:
            result_keypoint = results.keypoints.xyn.cpu().numpy()[0]
            if len(result_keypoint) > 0:
                keypoint_data = self.extract_keypoint(result_keypoint)
                return keypoint_data
            else:
                return [0]*34
        else:
            return [0]*34
    
    def __call__(self, images: list) -> Results:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        resized_frames = [cv2.resize(image, (192 // 32 * 32, 320 // 32 * 32)) for image in images]
        images_tensor = torch.from_numpy(np.stack(resized_frames)).permute(0, 3, 1, 2).float().to(device) / 255.0
        # images_tensor = images_tensor.to(self.model.device)

        # Perform inference with no gradient tracking
        with torch.no_grad():
            results = self.model.predict(images_tensor, save=False)
            # print(results)
        return [self.get_xy_keypoint(result) for result in results]

# Example usage
if __name__ == '__main__':
    detector = DetectKeypoint('/home/annone/ai/models/yolov8n-pose.pt')
    # Load or create a batch of images
    batch_images = [cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'),cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'),cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'),cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png'), cv2.imread('/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png')]  # Add paths to your images
    # batch_images = cv2.imread("/home/annone/ai/images/urination/3_person_0a20f5e2-355d-4999-86d9-6fd5c299bf85.png")
    t1 = time.time()
    keypoints = detector(batch_images)
    t2 = time.time()
    print(t2-t1)
    print(keypoints)  # This will print keypoints for each image in the batch