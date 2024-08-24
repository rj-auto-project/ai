import cv2
import os
import torch
from ultralytics import YOLO

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('I:/RJ/models/seg.pt')
model.to(device)

# Define global variables
class_list = ['auto', 'bike-rider', 'bolero', 'bus', 'car', 'hatchback', 'jcb', 'motorbike-rider', 'omni', 'pickup',
              'scooty-rider', 'scorpio', 'sedan', 'suv', 'swift', 'thar', 'tractor', 'truck', 'van']
count = 0
lines = []
text_positions = []
directions = []
traffic_violation_count = 0
wrong_way_count = 0
crossed_objects = {}  # To keep track of objects that have crossed lines

# 




# 
# Offset for line detection
offset = 10

def setup_model(model_path):
    model = YOLO(model_path)
    model.to(device)
    return model

def get_traffic_light_color():
    return input("Enter traffic light color (red/yellow/green): ").strip().lower()

def draw_line(event, x, y, flags, param):
    global lines, directions, text_positions

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a line
        lines.append([(x, y), (x, y)])
        text_positions.append((x, y))  # Record position for text

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        # Update the current line
        lines[-1][1] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish the current line
        lines[-1][1] = (x, y)
        # Once a pair of lines is drawn, ask for the direction
        if len(lines) % 2 == 0:
            direction = input(f"Enter direction for line pair {len(lines) // 2} (L-R/R-L): ").strip().upper()
            directions.append(direction)

def process_frame(frame, light_color):
    global traffic_violation_count, wrong_way_count, crossed_objects

    results = model.track(frame, persist=True, device=device)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        track_ids = (
            result.boxes.id.int().cpu().tolist()
            if result.boxes.id is not None
            else []
        )

        for box, cls, track_id in zip(boxes, classes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            detect_traffic_violation(cx, cy, track_id, frame, x2, y2, light_color)
            detect_wrong_way(cx, cy, track_id, frame, x2, y2)
    
    return frame

def over_detection_speed():
    pass

def detect_traffic_violation(cx, cy, track_id, frame, x2, y2, light_color):
    global traffic_violation_count

    if light_color == 'red' and len(lines) >= 2:  # Ensure there are at least two lines drawn
        (x_start1, y_start1), (x_end1, y_end1) = lines[0]
        (x_start2, y_start2), (x_end2, y_end2) = lines[1]

        # Check if object crosses the first line
        if min(y_start1, y_end1) - offset <= cy <= max(y_start1, y_end1) + offset:
            if cx > min(x_start1, x_end1) and cx < max(x_start1, x_end1):
                crossed_objects[track_id] = 'crossed_first_line'

        # Check if object crosses the second line after crossing the first
        if (track_id in crossed_objects and crossed_objects[track_id] == 'crossed_first_line' and
            min(y_start2, y_end2) - offset <= cy <= max(y_start2, y_end2) + offset and
            cx > min(x_start2, x_end2) and cx < max(x_start2, x_end2)):
            cv2.putText(frame, "Traffic Violation", (x2, y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            traffic_violation_count += 1
            crossed_objects[track_id] = 'crossed_both_lines'

def detect_wrong_way(cx, cy, track_id, frame, x2, y2):
    global wrong_way_count

    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines) or i // 2 >= len(directions):
            continue

        (x_start1, y_start1), (x_end1, y_end1) = lines[i]
        (x_start2, y_start2), (x_end2, y_end2) = lines[i + 1]
        direction = directions[i // 2]

        if direction == 'L-R':
            if min(x_start1, x_end1) - offset <= cx <= max(x_start1, x_end1) + offset and cy > min(y_start1, y_end1) and cy < max(y_start1, y_end1):
                crossed_objects[track_id] = 'crossed_first_line_L-R'
            if track_id in crossed_objects and crossed_objects[track_id] == 'crossed_first_line_L-R' and min(x_start2, x_end2) - offset <= cx <= max(x_start2, x_end2) + offset and cy > min(y_start2, y_end2) and cy < max(y_start2, y_end2):
                cv2.putText(frame, "Wrong Way!", (x2, y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                wrong_way_count += 1
                crossed_objects[track_id] = 'crossed_both_lines'

        elif direction == 'R-L':
            if min(x_start2, x_end2) - offset <= cx <= max(x_start2, x_end2) + offset and cy > min(y_start2, y_end2) and cy < max(y_start2, y_end2):
                crossed_objects[track_id] = 'crossed_first_line_R-L'
            if track_id in crossed_objects and crossed_objects[track_id] == 'crossed_first_line_R-L' and min(x_start1, x_end1) - offset <= cx <= max(x_start1, x_end1) + offset and cy > min(y_start1, y_end1) and cy < max(y_start1, y_end1):
                cv2.putText(frame, "Wrong Way!", (x2, y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                wrong_way_count += 1
                crossed_objects[track_id] = 'crossed_both_lines'

def draw_annotations(frame, light_color):
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(lines):
        line_color = (0, 0, 255) if light_color == 'red' else (0, 255, 255)
        cv2.line(frame, (x_start, y_start), (x_end, y_end), line_color, 2)
        cv2.putText(frame, f"Line {i+1}", text_positions[i], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.rectangle(frame, (0, 0), (250, 120), (0, 255, 255), -1)
    cv2.putText(frame, f'Traffic Violations - {traffic_violation_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Wrong Way Violations - {wrong_way_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return frame

def main(video_path, output_path, light_color):
    global count

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (1020, 500))

    if not os.path.exists('test_frames'):
        os.makedirs('test_frames')

    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', draw_line)

    while cap.isOpened():
        ret, ori_frame = cap.read()
        if not ret:
            break
        count += 1
        frame = cv2.resize(ori_frame, (1020, 500))

        frame = process_frame(frame, light_color)
        frame = draw_annotations(frame, light_color)

        frame_filename = f'test_frames/frame_{count}.jpg'
        cv2.imwrite(frame_filename, frame)

        out.write(frame)
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "I:/RJ/test_videos/wrongway.mp4"
    output_path = 'I:/RJ/output/test.mp4'
    light_color = get_traffic_light_color()  # Get the initial traffic light color
    main(video_path, output_path, light_color)