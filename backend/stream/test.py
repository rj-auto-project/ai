# import pandas as pd
# import psycopg2
# from datetime import datetime, timedelta
# import time
# from db import Database
# import random
# import uuid
# # Function to insert a row into the PostgreSQL table
# conn = Database.get_connection()
# cursor = conn.cursor()
# def insert_row(row):
#     global cursor, conn
#     try:

#         data = (row["camera_ip"], row["camera_id"],row["detection_id"],  row['license_number'],row['prediction_confidence'], row['track_id'],row['boxCoords'], row['detectionClass'], row["ownerName"])
#         print(data)
#         insert_query = '''
#         INSERT INTO "AnprLogs" (camera_ip,camera_id,detection_id, license_number, prediction_confidence, "trackId", "boxCoords", "detectionClass", "ownerName")
#         VALUES (%s, %s, %s, %s,%s,%s,%s,%s,%s);
#         '''
#         cursor.execute(insert_query, data)
#         conn.commit()

#         print(f"Inserted row: {row.to_dict()} at {datetime.now().strftime('%H:%M:%S')}")
#     except (Exception, psycopg2.Error) as error:
#         print("Error while connecting to PostgreSQL", error)
#     # finally:
#     #     if conn:
#     #         cursor.close()
#     #         conn.close()

# import os

# # Step 1: Read the CSV file
# csv_file = '/home/annone/ai/backend/stream/test.csv'
# df = pd.read_csv(csv_file)
# # Step 2: Start a countdown timer from 5:00 (HH:MM)
# start_time = datetime.strptime("05:00", "%M:%S")
# current_time = start_time

# # Convert the 'Time' column in the DataFrame to datetime objects
# df['Time'] = pd.to_datetime(df['Time'], format='%M:%S').dt.time

# # Step 3: Countdown loop
# while current_time.time() <= df['Time'].max():
#     current_formatted_time = current_time.strftime("%M:%S")
    
#     # Check if the current time matches any row in the CSV
#     matching_rows = df[df['Time'] == current_time.time()]
    
#     for _, row in matching_rows.iterrows():
#         row = row[:-1]
#         row["camera_id"] = "2"
#         row["camera_ip"] = "128.9.9.12"
#         row["detection_id"] = random.randint(178,999)
#         row['prediction_confidence'] = random.random()
#         row["track_id"] = str(uuid.uuid4())
#         row["boxCoords"] = [f"{random.randint(0,1080)}" f"{random.randint(0,1080)}" f"{random.randint(0,1080)}" f"{random.randint(0,1080)}"]
#         row["ownerName"] = "ramu kaka"
#         for i in row:
#             print(type(i))
#         print(row)
#         insert_row(row)
    
#     # Wait for one second
#     time.sleep(1)
    
#     # Increment the timer by one second
#     current_time += timedelta(seconds=1)
    
#     # Print the current timer (optional)
#     print(f"Timer: {current_formatted_time}")
    

# print("Finished processing all rows.")
# if conn:
#     cursor.close()
#     conn.close()
















from test_main import stream_process

video_path = '/home/annone/ai/data/T-pole wrong way.mp4'
cam_ip = '127.0.0.1'
cam_id = "1"
stream_process(cam_id, cam_ip,video_path)

