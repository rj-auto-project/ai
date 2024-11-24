import cv2
import numpy as np

offset=10
count = 0
total_parking_violations = 0 
wrong_way_violation_count = 0
traffic_violation_count = 0  
crossed_objects_wrong = {}
violated_objects_wrong = set()
crossed_objects = {}  # Track objects that have crossed lines
violated_objects = set()  # Track objects that have already violated
static_objects = {}  # To track objects that are stationary
stationary_frame_threshold = 200
roi_points = np.array([[00, 00], [640, 00], [640, 480], [00, 480]], dtype=np.int32)  # Replace with your specific points

ww_red_line = [
    [(133, 251), (438, 251)],  # First red line
    # [(42, 200), (368, 200)],   # Second red line
    # [(417, 182), (640, 182)]   # Third red line
]

ww_green_line = [
    [(44, 390), (525, 390)],   # First green line (paired with first red line)
    # [(213, 110), (368, 110)],  # Second green line (paired with second red line)
    # [(404, 118), (561, 118)]    # Third green line (paired with third red line)
]

# Function for check illegal parking
def is_point_in_polygon(point, polygon):
    """Check if a point is inside a polygon defined by a list of points."""
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def check_illegal_parking(track_id, cx, cy):
    """Check if an object is illegally parked based on stationary duration within a specific area."""
    global static_objects, total_parking_violations, offset
    print("Checking")
    # Check if the center of the object is inside the defined ROI
    if not is_point_in_polygon((cx, cy), roi_points):
        return  # Exit if the object is not in the monitored area

    # Initialize tracking information if not already present
    if track_id not in static_objects:
        static_objects[track_id] = {"position": (cx, cy), "frames": 0, "violated": False}
    else:
        last_position = static_objects[track_id]["position"]

        # Check if the object has remained stationary (within a certain offset)
        if abs(cx - last_position[0]) <= offset and abs(cy - last_position[1]) <= offset:
            static_objects[track_id]["frames"] += 1
            # If the object is stationary for more than the threshold and not already marked as a violation
            if static_objects[track_id]["frames"] > stationary_frame_threshold and not static_objects[track_id]["violated"]:
                static_objects[track_id]["violated"] = True  # Mark the object as a violation
                total_parking_violations += 1  # Increase the parking violation count
                print(f"Object {track_id} marked as parking violation. Total Violations: {total_parking_violations}")
        else:
            # If the object has moved, reset its tracking information
            static_objects[track_id]["position"] = (cx, cy)
            static_objects[track_id]["frames"] = 0
            static_objects[track_id]["violated"] = False

            
# Function for check wrong way driving
def detect_wrong_way_violation(track_id, cx, cy):
    global wrong_way_violation_count, violated_objects_wrong, offset
    print("Checking wrong way")
    # Initialize tracking for the object if not already done
    if track_id not in crossed_objects_wrong:
        crossed_objects_wrong[track_id] = {'red': set(), 'green': set()}

    # Check if the object crosses any red line
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_red_line):
        if min(y_start, y_end) - offset <= cy <= max(y_start, y_end) + offset:
            if min(x_start, x_end) <= cx <= max(x_start, x_end):
                crossed_objects_wrong[track_id]['red'].add(i)

    # Check if the object crosses any green line
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_green_line):
        if min(y_start, y_end) - offset <= cy <= max(y_start, y_end) + offset:
            if min(x_start, x_end) <= cx <= max(x_start, x_end):
                crossed_objects_wrong[track_id]['green'].add(i)

    # Detect wrong-way violation (crossing green line after red line)
    if any(
        i in crossed_objects_wrong[track_id]['green'] and
        i in crossed_objects_wrong[track_id]['red'] and
        track_id not in violated_objects_wrong
        for i in crossed_objects_wrong[track_id]['green']
    ):
        # cv2.putText(frame, "Wrong Way Violation", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        wrong_way_violation_count += 1
        violated_objects_wrong.add(track_id)
        
# Function for ckeckred light violation
def detect_traffic_violation(track_id, cx, cy):
    """Check if an object violates traffic rules by crossing lines."""
    global traffic_violation_count, crossed_objects, violated_objects, offset
    print("Checking traffic")
    
    # Initialize tracking for the object if not already done
    if track_id not in crossed_objects:
        crossed_objects[track_id] = {'red': set(), 'green': set()}

    # Check if the object crosses any red line
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_red_line):
        if min(y_start, y_end) - offset <= cy <= max(y_start, y_end) + offset:
            if min(x_start, x_end) <= cx <= max(x_start, x_end):
                crossed_objects[track_id]['red'].add(f"red_{i}")
                

    # Check if the object crosses any green line after crossing a red line (indicating a violation)
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_green_line):
        if min(y_start, y_end) - offset <= cy <= max(y_start, y_end) + offset:
            if min(x_start, x_end) <= cx <= max(x_start, x_end):
                if any(f"red_{j}" in crossed_objects[track_id]['red'] for j in range(len(ww_red_line))) and track_id not in violated_objects:
                    traffic_violation_count += 1
                    violated_objects.add(track_id)

def drawnlinesfortrafficviolation(frame,total_violations):
    """Draw the red and green lines on the frame and display the traffic violation count."""
    cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(frame, "Illegal Parking Area", (roi_points[0][0], roi_points[0][1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_red_line):
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.putText(frame, f"Red {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_green_line):
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(frame, f"Green {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the traffic violation count
    cv2.rectangle(frame, (0, 0), (350, 60), (0, 255, 255), -1)  # Yellow background
    cv2.putText(frame, f'Traffic Violations - {traffic_violation_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Parking Violations - {total_violations}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Wrong Way Violations - {wrong_way_violation_count}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)