import signal
import sys
from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import tensorflow as tf
import torch

# Define a global variable to store results
results = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = YOLO("D:/local-intelligent-cameras/backend/stream/best.pt")
coco_model = YOLO("/home/devendra/ai-camera/backend/apis/anpr/yolov8n.pt")
license_plate_detector = YOLO("/home/devendra/LP_model/runs/detect/train/weights/best.pt")
coco_model.to(device)
license_plate_detector.to(device)
# print(device)

def save_results_and_exit(signum, frame):
    """ Save results to a CSV file and exit """
    print("Ctrl+C pressed. Saving results to CSV and exiting...")
    write_csv(results, "test.csv")
    sys.exit(0)

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, save_results_and_exit)

def process_video(coco_model, license_plate_detector, video_path, camera_id):

    global results
    mot_tracker = Sort()

    # coco_model = coco_model
    # license_plate_detector = license_plate_detector

    cap = cv2.VideoCapture(video_path)
    vehicles = [2, 3, 5, 7]  # Vehicle classes (car, motorcycle, bus, truck)
    frame_nmr = -1
    ret = True
    with tf.device("gpu:0"):
        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if ret:
                results[frame_nmr] = {}
                print(device)
                detections = coco_model(frame,device=device)[0]
                detections_ = []
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score, int(class_id)])
                        # Uncomment these lines if you want to visualize detected vehicles
                        # color = (0, 255, 0)
                        # label = coco_model.names[int(class_id)]
                        # label_text = f"{label}: {score:.2f}"
                        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        # cv2.putText(frame, label_text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if detections_:
                    track_ids = mot_tracker.update(np.asarray(detections_))
                else:
                    track_ids = mot_tracker.update(np.empty((0, 5)))

                license_plates = license_plate_detector(frame,device=device)[0]
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                    if car_id != -1:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (250, 0, 0), 2)
                        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(
                            license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                        )

                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh,video_path,camera_id,frame_nmr)

                        if license_plate_text is not None:
                            print(f'Recognized License Plate: {license_plate_text}, Confidence Score: {license_plate_text_score}')
                            results[frame_nmr][car_id] = {
                                "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                                "license_plate": {
                                    "bbox": [x1, y1, x2, y2],
                                    "text": license_plate_text,
                                    "bbox_score": score,
                                    "text_score": license_plate_text_score,
                                },
                            }

                # cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

    write_csv(results, "/home/devendra/ai-camera/backend/apis/anpr/test.csv")

# yolo_coco_model_path = "D:/local-intelligent-cameras/backend/apis/anpr/yolov8n.pt"
# yolo_license_plate_model_path = "D:/local-intelligent-cameras/backend/apis/anpr/indian_license_plate_detector_0.2.pt"
video_path = "/home/devendra/data/171.31.4.26.MKV"
camera_id = "1"
process_video(coco_model, license_plate_detector, video_path, camera_id)