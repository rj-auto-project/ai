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

















import redis
import pickle
import numpy as np

# Initialize Redis client
r = redis.Redis(host='localhost', port=6379, db=0)

def store_string_and_images_in_redis(redis_key, my_string, images):
    # Create a dictionary with the string and the numpy array
    data = {
        'string': my_string,
        'images': images
    }
    data['string'] = "new string"
    # Serialize the dictionary using pickle
    serialized_data = pickle.dumps(data)
    
    # Store the serialized data in Redis with the given key
    r.set(redis_key, serialized_data)

def retrieve_data_from_redis(redis_key):
    # Retrieve the serialized data from Redis
    serialized_data = r.get(redis_key)
    
    if serialized_data:
        # Deserialize the data using pickle
        data = pickle.loads(serialized_data)
        return data
    else:
        return None

# Example usage
if __name__ == "__main__":
    # Example string
    my_string = "This is a test string"
    
    # Example array of images (as numpy arrays)
    img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Random image 1
    img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Random image 2
    images = [img1, img2]  # List of images
    
    # Store the string and images in Redis under a single key
    store_string_and_images_in_redis('my_data_key', my_string, images)
    
    # Retrieve the data from Redis
    data = retrieve_data_from_redis('my_data_key')
    
    if data:
        print("Retrieved String:", data['string'])
        print("Retrieved Images:", len(data['images']), "images")


