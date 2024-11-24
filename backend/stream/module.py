import cv2
import torch
import tensorflow as tf
from timmML_025.models.factory import create_model
import torchvision.transforms as transforms
from PIL import Image
import signal
import sys
import time
import random
import string
from db import Database
import numpy as np
from sklearn.cluster import KMeans
import redis
from psycopg2.extras import execute_values
from collections import Counter, defaultdict
import redis
import psycopg2
import ast  # To convert string representation of lists/dicts from Redis into Python objects

### GLOBAL VARIABLE AND MODELS
r = redis.Redis(host='localhost', port=6379, db=0)
PARENT_DIR = "/home/annone/ai"
global video_writer
# device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loading model on device
# model = YOLO("/home/annone/ai/models/crowd_count.pt")
# model.to(device)
# creating crowd-count model
crowd_count_model = create_model("efficientnet_lite0")
PATH_model = "/home/annone/ai/models/cc_count.pt"
MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
crowd_count_model.load_state_dict(torch.load(PATH_model, map_location=device))
# loading model on device
crowd_count_model.to(device)
crowd_count_model.eval()
img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN_STD[0], MEAN_STD[1])]
)
corp_size = 256


print("start")

### SIGNAL HANDLING

def signal_handler(sig, frame):
    print("Termination signal received. Releasing resources...")
    if "video_writer" in globals() and video_writer is not None:
        video_writer.release()
    sys.exit(0)
# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


### GENERATE CUSTOM UNIQUE-ID

def generate_custom_string(cam_ip, track_id):
    cam_ip = cam_ip.replace(".", "_")
    current_timestamp = int(time.time())
    # Generate a random string of 4 characters and 4 digits mixed
    random_mix = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    custom_string = f"{random_mix}_{track_id}_{current_timestamp}_{cam_ip}"

    return custom_string


### CROWD COUNT PER FRAME


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

def crowd_count_on_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    img = resize_image(img)
    crowd_count, crowd_confidence = predict_count(img, crowd_count_model)
    return crowd_count, crowd_confidence


# Demo Color detection with K-mean
def k_mean_color_detection(image, k=3):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors = colors[sorted_indices]
    # return dominant_colors.astype(int)
    return '{"top":"red","bottom":"yellow"}'

def color_detection():
    matching_keys = r.keys(f"*:d_log")
    for key in matching_keys:
        value = r.get(key)

# '1|127.0.0.1|[412,204,77,121]|motorbike-rider|qcYXs5PM_39_1724913805_127_0_0_1|0.8612010478973389|817_qcYXs5PM_39_1724913805_127_0_0_1_976'

def process_raw_d_logs(auto_loop = True):
    print("cpu")
    while True:
        matching_keys = r.keys(f"*:raw_d_log")
        for key in matching_keys:
            value = r.get(key)
            decode_key = key.decode()
            decode_value = value.decode().split("|")
            print(decode_value)
            uuid = decode_value[6]
            try:
                image_data = r.get(f"{uuid}:image")
                np_arr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                image = image.reshape((-1, image.shape[2]))
                aa = k_mean_color_detection(image)
                r.delete(f"{uuid}:image")
                decode_value[6] = aa
                decode_value.append("red")
                decode_value.append("blue")
                r.set(f"{uuid}:d_log","|".join(decode_value))
                r.delete(decode_key)
            except:
                print("deleted")
                r.delete(decode_key)
        if auto_loop == False:
            break
        time.sleep(2)

def process_d_logs(auto_loop = True):
    conn = Database.get_connection()
    cursor = conn.cursor()
    batch = []
    while True:
        matching_keys = r.keys(f"*:d_log")
        if len(matching_keys) > 0:
            for key in matching_keys:
                value = r.get(key)
                decode_key = key.decode()
                decode_value = value.decode().split("|")
                batch.append(tuple(decode_value))
                r.delete(decode_key)
            if len(batch) > 0:
                try:
                    query = 'INSERT INTO "DetectionLog" ("cameraId", "camera_ip", "boxCoords", "detectionClass", "trackId", "classConfidence", "metadata", "topColor", "bottomColor") VALUES %s;'
                    execute_values(cursor, query, batch)
                    conn.commit()
                    batch = []
                except:
                    print("unable to save to DB")
        if auto_loop == False:
            break
        time.sleep(2)

def process_raw_cc_logs(auto_loop = True):
    while True:
        matching_keys = r.keys(f"*:raw_cc_log")
        # print(matching_keys)
        for key in matching_keys:
            value = r.get(key)
            decode_key = key.decode()
            decode_value = value.decode().split('|')
            uuid = decode_value[1]
            try:
                image_data = r.get(f"{uuid}:image")
                np_arr = np.frombuffer(image_data,np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                image = image.reshape(-1,image.shape[2])
                # image = cv2.imread(f"{PARENT_DIR}/backend/stream/cc_temp/{decode_value[2]}.jpg")
                crowd_count, crowd_confidence = crowd_count_on_frame(image)
                print(crowd_confidence)
                r.set(f"{decode_key}:cc_log",f"{decode_value[0]}|{crowd_count}|{crowd_confidence}")
                r.delete(decode_key)
            except:
                print("deleted")
                r.delete(decode_key)
        if auto_loop == False:
            break
        time.sleep(2)

def process_cc_logs(auto_loop = True):
    conn = Database.get_connection()
    cursor = conn.cursor()
    batch = []
    while True:
        matching_keys = r.keys(f"*:cc_log")
        if len(matching_keys) > 0:
            for key in matching_keys:
                value = r.get(key)
                decode_key = key.decode()
                decode_value = value.decode().split("|")
                batch.append(tuple(decode_value))
                print(decode_value)
                r.delete(decode_key)
            if len(batch) > 0:
                try:
                    query = 'INSERT INTO "CrowdCount" ("camera_ip", "count", "confidence") VALUES %s;'
                    execute_values(cursor, query, batch)
                    conn.commit()
                    batch = []
                except:
                    r.delete(decode_key)
            print("one done")
        if auto_loop == False:
            break
        time.sleep(2)

# async def async_pool():
#     await asyncio.gather(process_raw_d_logs(), process_d_logs(), process_raw_cc_logs(), process_cc_logs())

# def open_async_pool():
#     asyncio.run(async_pool())


def is_within_offset(last_position, current_loc, offset):
    """Check if the current location is within the offset range of the last position."""
    if last_position is None:
        return False
    distance = np.linalg.norm(np.array(current_loc) - np.array(last_position))
    return distance <= offset

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def draw_lines_and_text(frame,red_lines,green_lines, traffic_violation_count):
    """Draw the red and green lines on the frame and display the traffic violation count."""
    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(red_lines):
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)
        cv2.putText(frame, f"Red {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)

    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(green_lines):
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
        cv2.putText(frame, f"Green {i + 1}", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

    cv2.rectangle(frame, (0, 0), (350, 50), (0, 255, 255), 1)
    cv2.putText(frame, f'Traffic Violations - {traffic_violation_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def calculate_speed(last_position, current_loc, fps):
    """ Calculate the speed of the object based on the distance covered between frames """
    if last_position is None:
        return 0
    distance = np.linalg.norm(np.array(current_loc) - np.array(last_position))
    speed = distance * fps
    return speed

def is_within_range(x, y, y_min, y_max, x_min, x_max):
    return y_min <= y <= y_max and x_min <= x <= x_max

def check_traffic_violation(cx, cy, track_id, rlv_crossed_objects, rlv_violated_objects, frame, red_line_ranges_np,green_line_ranges_np):
    for i, (y_min, y_max, x_min, x_max) in enumerate(red_line_ranges_np):
        if is_within_range(cx, cy, y_min, y_max, x_min, x_max):
            rlv_crossed_objects.setdefault(track_id, set()).add(f"red_{i}")
            cv2.putText(frame, "crossed red", (cx + 10, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
            print("crossed red")
            break

    for i, (y_min, y_max, x_min, x_max) in enumerate(green_line_ranges_np):
        if is_within_range(cx, cy, y_min, y_max, x_min, x_max):
            if f"red_{i}" in rlv_crossed_objects.get(track_id, set()) and track_id not in rlv_violated_objects:
                cv2.putText(frame, "crossed green", (cx + 10, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
                print("crossed green")
                return True

    return False

def check_wrong_way_violation(cx, cy, track_id, ww_crossed_objects, ww_violated_objects, ww_red_lines_np, ww_green_lines_np, ww_offset, frame):
    crossed_red = False
    crossed_green = False

    ww_crossed_objects.setdefault(track_id, {'red': set(), 'green': set()})

    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_red_lines_np):
        if min(y_start, y_end) - ww_offset <= cy <= max(y_start, y_end) + ww_offset and min(x_start, x_end) <= cx <= max(x_start, x_end):
            ww_crossed_objects[track_id]['red'].add(i)
            crossed_red = True
            break

    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(ww_green_lines_np):
        if min(y_start, y_end) - ww_offset <= cy <= max(y_start, y_end) + ww_offset and min(x_start, x_end) <= cx <= max(x_start, x_end):
            ww_crossed_objects[track_id]['green'].add(i)
            crossed_green = True
            break

    if crossed_red and crossed_green and track_id not in ww_violated_objects:
        return True
    return False

def process_and_store_detections():
    # Fetch all keys from Redis
    keys = r.keys('*')
    
    # PostgreSQL connection setup
    conn = Database.get_connection()
    cursor = conn.cursor()

    # Iterate over all Redis keys
    for key in keys:
        data = r.get(key)
        if data:
            # Convert Redis data (string) back to a dictionary
            data_dict = ast.literal_eval(data.decode("utf-8"))

            # Extract necessary fields
            custom_track_id = data_dict.get('custom_track_id')
            camera_id = data_dict.get('camera_id')
            camera_ip = data_dict.get('camera_ip')
            first_appearance = data_dict.get('first_appearance')
            last_appearance = data_dict.get('last_appearance')
            bboxes = data_dict.get('dbbox', [])
            dlabels = data_dict.get('dlabel', [])
            dconfs = data_dict.get('dconf', [])

            # If there are multiple labels, average them by selecting the most frequent one
            if dlabels:
                label_counter = Counter(dlabels)
                most_frequent_label = label_counter.most_common(1)[0][0]  # Get most frequent label
                corresponding_confs = [dconfs[i] for i, label in enumerate(dlabels) if label == most_frequent_label]
                corresponding_bboxes = [bboxes[i] for i, label in enumerate(dlabels) if label == most_frequent_label]
                avg_conf = sum(corresponding_confs) / len(corresponding_confs) if corresponding_confs else None
            else:
                most_frequent_label = None
                avg_conf = None
                corresponding_bboxes = []

            # Insert data into PostgreSQL
            try:
                cursor.execute("""
                    INSERT INTO detectionLogs (customTrackID, cameraID, cameraIP, firstAppearance, lastAppearance, dlabel, dconf, dbbox)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    custom_track_id, 
                    camera_id, 
                    camera_ip, 
                    first_appearance, 
                    last_appearance, 
                    most_frequent_label, 
                    avg_conf, 
                    corresponding_bboxes
                ))
            except Exception as e:
                print(f"Error inserting into PostgreSQL: {e}")
                conn.rollback()
            else:
                conn.commit()

    cursor.close()
    conn.close()

# Call the function
process_and_store_detections()