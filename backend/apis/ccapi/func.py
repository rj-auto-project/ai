import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
from timmML_025.models.factory import create_model
import datetime
import mysql.connector
from mysql.connector import Error
from db_sql import predict_count,resize_image
# frame_count=0
# def crowd_insert_into_table(camera_id, camera_ip, timestamp, crowd_count, crowd_count_confidence_score):
#     try:
#         # Connect to the database
#         # insert_into_table(camera_ip = video_path,timestamp=timestamp,metadata = metadata, box_coords = [x1, y1, x2, y2], detection_class = label)
#         connection = mysql.connector.connect(
#             host="localhost", user="root", password="", database="logs"
#         )
#         if connection.is_connected():
#             cursor = connection.cursor()
#             # Create the insert query
#             insert_query = f"INSERT INTO crowd_count_logs (camera_id, camera_ip, timestamp, crowd_count, crowd_count_confidence_score) VALUES ('{camera_id}', '{camera_ip}',' {timestamp}',' {crowd_count}', '{crowd_count_confidence_score}')"
#             # Execute the insert query
#             cursor.execute(insert_query)
#             # Commit the transaction
#             connection.commit()
#     except Error as e:
#         print(f"Error: {e}")
#     finally:
#         if connection.is_connected():
#             cursor.close()
#             connection.close()


# model = create_model('efficientnet_lite0')  # Adjust according to your model
# PATH_model = "D:/local-intelligent-cameras/backend/apis/ccapi/student_025.pt"
# MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# corp_size = 256
# # Load the state dictionary to the appropriate device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.load_state_dict(torch.load(PATH_model, map_location=device))
# model.to(device)  # Ensure the model is on the correct device
# model.eval()

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(MEAN_STD[0], MEAN_STD[1])
# ])

# # Function to resize image



# video_path='D:/local-intelligent-cameras/backend/apis/anpr/test.mp4'
# camera_ip='171.2.3.4'
# camera_id=1
# cap = cv2.VideoCapture(video_path)
# frame_count = 0

# while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert BGR frame to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Process the frame
#         img = Image.fromarray(rgb_frame)
#         img = resize_image(img)
#         pred_count, confidence_score = predict_count(img, model)
#         current_time_ist = datetime.datetime.now()
#         # Save predicted count to database
#         crowd_count_entry = crowd_insert_into_table(
#             camera_id=camera_id,
#             camera_ip=camera_ip,
#             timestamp=current_time_ist,
#             crowd_count=pred_count,
#             crowd_count_confidence_score=confidence_score,  # Placeholder for confidence score
           
#         )

#         # Send predicted count over WebSocket

# cap.release()

def crowd_insert_into_table(camera_id, camera_ip, timestamp, crowd_count, crowd_count_confidence_score):
    try:
        # Connect to the database
        # insert_into_table(camera_ip = video_path,timestamp=timestamp,metadata = metadata, box_coords = [x1, y1, x2, y2], detection_class = label)
        connection = mysql.connector.connect(
            host="localhost", user="root", password="", database="logs"
        )
        if connection.is_connected():
            cursor = connection.cursor()
            # Create the insert query
            insert_query = f"INSERT INTO crowd_count_logs (camera_id, camera_ip, timestamp, crowd_count, crowd_count_confidence_score) VALUES ('{camera_id}', '{camera_ip}',' {timestamp}',' {crowd_count}', '{crowd_count_confidence_score}')"
            # Execute the insert query
            cursor.execute(insert_query)
            # Commit the transaction
            connection.commit()
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            
            
def crowd_count(video_path,camera_ip, camera_id):
    model = create_model('efficientnet_lite0')  # Adjust according to your model
    PATH_model = "D:/local-intelligent-cameras/backend/apis/ccapi/student_025.pt"
    MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # Load the state dictionary to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(PATH_model, map_location=device))
    model.to(device)  # Ensure the model is on the correct device
    model.eval()

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_STD[0], MEAN_STD[1])
    ])
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count == 1 or frame_count % 100 == 0:  # Process the first frame and then every 100 frames
            # Convert BGR frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            img = Image.fromarray(rgb_frame)
            img = resize_image(img)
            pred_count, confidence_score = predict_count(img, model)
            current_time_ist = datetime.datetime.now()
            # Save predicted count to database
            crowd_insert_into_table(
                camera_id=camera_id,
                camera_ip=camera_ip,
                timestamp=current_time_ist,
                crowd_count=pred_count,
                crowd_count_confidence_score=confidence_score
            )
            print("Saved data for frame", frame_count)

    cap.release()

if __name__=='__main__':
    video_path='D:/local-intelligent-cameras/backend/apis/ccapi/171.31.4.36.mp4'
    camera_ip='171.31.4.36'
    camera_id=2
    crowd_count(video_path,camera_ip,camera_id)