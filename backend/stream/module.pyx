import cv2
from ultralytics import YOLO
from mysql.connector import Error
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
# from color_detection import format_output, crop_and_classify
import joblib
from db import Database
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import redis
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from psycopg2.extras import execute_values

### GLOBAL VARIABLE AND MODELS
r = redis.Redis(host='localhost', port=6379, db=0)

global video_writer
# device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loading model on device
model = YOLO("/home/annone/ai-camera/backend/stream_test/best.pt")
model.to(device)
# creating crowd-count model
crowd_count_model = create_model("efficientnet_lite0")
PATH_model = "/home/annone/ai-camera/backend/stream_test/student_025.pt"
MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
crowd_count_model.load_state_dict(torch.load(PATH_model, map_location=device))
# loading model on device
crowd_count_model.to(device)
crowd_count_model.eval()
img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN_STD[0], MEAN_STD[1])]
)
corp_size = 256
# Initialize the model
model = create_model('efficientnet_lite0')  # Adjust according to your model
PATH_model = "/home/annone/ai-camera/backend/stream/student_025.pt"

# Load the state dictionary to the appropriate device
model.load_state_dict(torch.load(PATH_model, map_location=device))
model.to(device)  # Ensure the model is on the correct device
model.eval()

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN_STD[0], MEAN_STD[1])
])

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


### DETECTIONS LOGS SAVING


def insert_into_table():
    try:
        connection = Database.get_connection()
        # if connection.is_connected():
        cursor = connection.cursor()
            # insert_query = f"INSERT INTO detection_logs (camera_id, camera_ip, timestamp, box_coords, detection_class, track_id, class_confidence, metadata) VALUES ('{camera_id}' ,'{camera_ip}', '{timestamp}', '{box_coords}', '{detection_class}', '{track_id}', '{class_confidence}', '{metadata}');"
            # cursor.execute(insert_query)
            # connection.commit()
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection:
            Database.return_connection(connection)


### CROWD-COUNT LOGS SAVING

def crowd_insert_into_table(
    # camera_id, camera_ip, timestamp, crowd_count, crowd_count_confidence_score
):
    try:
        connection = Database.get_connection()
        # if connection.is_connected():
        cursor = connection.cursor()
            # insert_query = f"INSERT INTO crowd_count (camera_id, camera_ip, timestamp, crowd_count, crowd_count_confidence_score) VALUES ('{camera_id}', '{camera_ip}',' {timestamp}',' {crowd_count}', '{crowd_count_confidence_score}')"
            # cursor.execute(insert_query)
            # connection.commit()
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection:
            Database.return_connection(connection)


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


def process_raw_d_logs():
    while True:
        matching_keys = r.keys(f"*:raw_d_log")
        for key in matching_keys:
            value = r.get(key)
            decode_key = key.decode()
            decode_value = value.decode().split("|")
            track_id = decode_value[5]
            image_path = decode_value[7]
            try:
                image = cv2.imread(f"/home/annone/ai-camera/backend/stream/temp/{image_path}")
                image = image.reshape((image.shape[0] * image.shape[1], 3))
                aa = k_mean_color_detection(image)
                decode_value[7]  = aa
                r.set(f"{track_id}_{random.randint(0,9999)}:d_log","|".join(decode_value))
                os.remove(f"/home/annone/ai-camera/backend/stream/temp/{image_path}")
                r.delete(decode_key)
            except:
                # print("excepted")
                continue
            # show_rgb_colors(aa,image)
            # print(decode_value)

def process_d_logs():
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
            if len(batch) > 0:
                query = 'INSERT INTO "DetectionLog" ("cameraId", "camera_ip", "timestamp", "boxCoords", "detectionClass", "trackId", "classConfidence", "metadata") VALUES %s;'
                execute_values(cursor, query, batch)
                conn.commit()
                cursor.close()
                conn.close()
                batch = []

def process_raw_cc_logs():
    while True:
        matching_keys = r.keys(f"*:raw_cc_log")
        for key in matching_keys:
            value = r.get(key)
            decode_key = key.decode()
            decode_value = value.decode().split('|')
            try:
                image = cv2.imread(f"/home/annone/ai-camera/backend/stream/cc_temp/{decode_value[2]}.jpg")
                crowd_count, crowd_confidence = crowd_count_on_frame(image)
                r.set(f"{decode_value[2]}:cc_log",f"{decode_value[0]}|{decode_value[1]}|{crowd_count}|{crowd_confidence}")
                os.remove(f"/home/annone/ai-camera/backend/stream/cc_temp/{decode_value[2]}.jpg")
                r.delete(decode_key)
            except:
                pass

def process_cc_logs():
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
            if len(batch) > 0:
                query = 'INSERT INTO "CrowdCount" ("camera_ip", "timestamp", "count", "confidence") VALUES %s;'
                execute_values(cursor, query, batch)
                conn.commit()
                batch = []

def do_threading():
    print("start threading")
    frame = cv2.imread("/home/annone/ai-camera/backend/stream_test/4.png")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(generate_custom_string,("23.45.67.89","ABGC25563"))
        executor.map(insert_into_table)
        executor.map(crowd_insert_into_table)
        executor.map(crowd_count_on_frame,frame)

    # t1 = threading.Thread(target=generate_custom_string,args=("23.45.67.89","ABGC25563"))
    # t2 = threading.Thread(target=insert_into_table)
    # t3 = threading.Thread(target=crowd_insert_into_table)
    # t4 = threading.Thread(target=crowd_count_on_frame,args=(frame))

    # t1.start()
    # t2.start()
    # t3.start()
    # t4.start()

    # t1.join()
    # t2.join()
    # t3.join()
    # t4.join()
    return "success"