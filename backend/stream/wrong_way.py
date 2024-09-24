import cv2
import os
import time
import torch
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('I:/RJ/models/seg.pt')
model.to(device)

class_list = [ 'auto','bike-rider','bolero','bus','car','hatchback','jcb','motorbike-rider','omni','pickup',
               'scooty-rider','scorpio','sedan','suv','swift','thar','tractor','truck','van' ]

count = 0

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

video_path = "I:/RJ/test_videos/wrongway.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error opening video file: {video_path}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('I:/RJ/output/test.mp4', fourcc, 20.0, (1020, 500))

down = {}
up = {}
counter_down = []
counter_up = []
wrong_way_count = 0

red_line_y = 166
blue_line_y = 234

text_color = (0, 0, 0)
yellow_color = (0, 255, 255)
red_color = (0, 0, 255)
blue_color = (255, 0, 0)

offset = 10

# Create a folder to save frames
if not os.path.exists('test_frames'):
    os.makedirs('test_frames')

while cap.isOpened():
    ret, ori_frame = cap.read()
    if not ret:
        break
    count += 1
    frame = cv2.resize(ori_frame, (1020, 500))
    
    results = model.track(frame, persist=True, device=device)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        track_ids = (
            result.boxes.id.int().cpu().tolist()
            if result.boxes.id is not None
            else []
        )

        for box, score, cls, track_id in zip(boxes, scores, classes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            class_confidence = score
            track_id = int(track_id)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Check for going down
            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                down[track_id] = time.time()
            if track_id in down:
                if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                    elapsed_time = time.time() - down[track_id]
                    if counter_down.count(track_id) == 0:
                        if cx < 650:
                            wrong_way_count += 1
                            cv2.putText(frame, "Wrong Way ", (x2, y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        counter_down.append(track_id)
                        distance = 20  # meters
                        a_speed_ms = distance / elapsed_time
                        a_speed_kh = a_speed_ms * 3.6
                        cv2.putText(frame, f"{int(a_speed_kh)} Km/h", (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            # Check for going up
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                up[track_id] = time.time()
            if track_id in up:
                if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                    elapsed1_time = time.time() - up[track_id]
                    if counter_up.count(track_id) == 0:
                        if cx > 650:
                            wrong_way_count += 1
                            cv2.putText(frame, "Wrong Way", (x2, y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        counter_up.append(track_id)
                        distance1 = 20  # meters
                        a_speed_ms1 = distance1 / elapsed1_time
                        a_speed_kh1 = a_speed_ms1 * 3.6
                        cv2.putText(frame, f"{int(a_speed_kh1)} Km/h", (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)


    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)
    cv2.line(frame, (284, 166), (614, 166), red_color, 2)
    cv2.putText(frame, ('Red Line'), (168, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(frame, (254, 234), (680, 234), blue_color, 2)
    cv2.putText(frame, ('Blue Line'), (236, 234), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, ('Wrong Way - ' + str(wrong_way_count)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Save frame
    frame_filename = f'test_frames/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)

    out.write(frame)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()