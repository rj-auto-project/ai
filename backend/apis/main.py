from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import mysql.connector
import multiprocessing
import tensorflow as tf
import os
from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import os

# from face_searching.fr import live_face_srching_api as f_srch
from fastapi.responses import FileResponse
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import sessionmaker
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    TIMESTAMP,
    MetaData,
)
from fastapi.encoders import jsonable_encoder
import asyncio
import json
from datetime import datetime
from plyer import notification
from db.frame_extract import extract_frame
import cv2
from io import BytesIO
from PIL import Image
import base64

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:5667",
    "http://34.47.148.81:5667",
    "http://34.47.148.81",
    "https://34.47.148.81",
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DATABASE_URL = "mysql+pymysql://root@localhost:3306/logs"


@app.get("/")
async def test():
    return "test"


# Fetch Camera's rtmp link
@app.get("/cam/url")
async def get_cam_url():
    from db.cam import fetch_cam_data

    data = fetch_cam_data()
    return data


# License plate searching api
@app.get("/search_license_plate/")
async def srch_license_plate(license_plate: str):
    from number_plate_searching.np_db_searching import license_plate_srch

    response = license_plate_srch(license_plate)
    return response


# # crowd count api
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     from ccapi.main import process_video

#     await websocket.accept()

#     try:
#         while True:
#             data = await websocket.receive_text()
#             video_path = data  # The client sends the video path

#             # Process the video and send the counts
#             await process_video(websocket, video_path)

#     except WebSocketDisconnect:
#         print("Client disconnected")


# Restrict any type of vehicle from an area
# Database connection details
db_config = {"host": "localhost", "user": "root", "password": "", "database": "logs"}


def get_db_connection():
    return mysql.connector.connect(**db_config)


def restrict_vehicle_sql_search(vehicle, ids, initial_timestamp, final_timestamp):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = f"""SELECT track_id,camera_ip,timestamp,camera_id,detection_class,id 
    FROM (
    SELECT track_id, camera_ip, timestamp, camera_id, detection_class, id,
        ROW_NUMBER() OVER (PARTITION BY track_id ORDER BY timestamp) AS row_num
        FROM detection_logs 
    WHERE detection_class IN {vehicle}
    AND camera_id IN {ids}
    AND timestamp BETWEEN '{initial_timestamp}' AND '{final_timestamp}') AS dt
    WHERE row_num = 1;"""
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result


def operation_db_save(
    ips_list,
    ids_list,
    area_list,
    detection,
    detection_data,
    initial_timestamp,
    final_timestamp,
):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = "INSERT INTO operation_logs (camera_id, camera_ip, area, detection, detection_data, initial_timestamp, final_timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s);"
    values = (
        f"{ids_list}",
        f"{ips_list}",
        f"{area_list}",
        f"{detection}",
        f"{detection_data}",
        f"{initial_timestamp}",
        f"{final_timestamp}",
    )
    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    conn.close()
    return "success"


def find_area_by_id(obj, search_id):
    for camera in obj["addedCameras"]:
        if camera["id"] == search_id:
            return camera["area"]
    return None


@app.websocket("/restrict_vehicle/{usage}")
async def websocket_search(websocket: WebSocket, usage: str):
    await websocket.accept()
    try:
        query_data = await websocket.receive_text()
        query_params = json.loads(query_data)
        vehicle_param = query_params.get("objects")
        camera_details_param = query_params.get("addedCameras")
        ids_param = [e.get("id") for e in camera_details_param]
        ips_param = [e.get("cameraIp") for e in camera_details_param]
        area_param = [e.get("area") for e in camera_details_param]
        vehicle_list = f"({', '.join(repr(cls) for cls in vehicle_param)})"
        ids_list = f"({', '.join(repr(cls) for cls in ids_param)})"
        vehicle_list_i = f"[{', '.join(repr(cls) for cls in vehicle_param)}]"
        ids_list_i = f"[{', '.join(repr(cls) for cls in ids_param)}]"
        ips_list = f"[{', '.join(repr(cls) for cls in ips_param)}]"
        area_list = f"[{', '.join(repr(cls) for cls in area_param)}]"
        initial_timeframe = query_params.get("start_time")
        final_timestamp = query_params.get("end_time")
        if usage == "map":
            operation_db_save(
                ips_list,
                ids_list_i,
                area_list,
                "vehicle",
                vehicle_list_i,
                initial_timeframe,
                final_timestamp,
            )
        if len(vehicle_list) == 0:
            await websocket.send_text(
                json.dumps({"error": "vehicle query parameter is required"})
            )
            await websocket.close()
            return
        while True:
            result = restrict_vehicle_sql_search(
                vehicle_list, ids_list, initial_timeframe, final_timestamp
            )
            if result:
                result = jsonable_encoder(result)
                for item in result:
                    item["area"] = (
                        f"{find_area_by_id(query_params, str(item['camera_id']))}"
                    )
                try:
                    await websocket.send_text(json.dumps(result))
                except Exception as e:
                    notification.notify(
                        title="ALERT⚠️",
                        message="Some Restricted Vehicle Enter in the Zone",
                        app_name="Notification App",
                        timeout=10,
                    )
            # initial_timeframe = datetime.fromisoformat(initial_timeframe)
            initial_timeframe = (datetime.now()).replace(microsecond=0)
            final_timestamp_dt = datetime.fromisoformat(final_timestamp)
            if initial_timeframe >= final_timestamp_dt:
                await websocket.send_text(json.dumps([{"end": "end"}]))
                await websocket.close()
                break
            else:
                # await websocket.send_text(json.dumps({"status": "not found"}))
                await asyncio.sleep(1)  # Check every 5 seconds
            initial_timeframe = initial_timeframe.strftime("%Y-%m-%dT%H:%M:%S")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        # await websocket.close()


# SUSPECT SEARCHING
# class srch_req(BaseModel):
#     # number_plate: str
#     vehicleType: str
#     vehicleColor: str
#     clothingColor: str
#     startDate: str
#     endDate: str


def suspect_search_operation_db_save(
    ips_list,
    ids_list,
    area_list,
    detection,
    class_name,
    lower_color,
    upper_color,
    initial_timestamp,
    final_timestamp,
):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    detection_data = {
        "class_name": {class_name},
        "lower": {lower_color},
        "upper": {upper_color},
    }
    query = "INSERT INTO operation_logs (camera_id, camera_ip, area, detection, detection_data, initial_timestamp, final_timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s);"
    values = (
        f"{ids_list}",
        f"{ips_list}",
        f"{area_list}",
        f"{detection}",
        f"{detection_data}",
        f"{initial_timestamp}",
        f"{final_timestamp}",
    )
    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    conn.close()
    return "success"


def suspect_search_sql(
    class_name, upper_color, lower_color, ids, initial_timestamp, final_timestamp
):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = f"""SELECT camera_ip,timestamp,camera_id,detection_class,id,metadata, track_id
    FROM (
    SELECT *,
         ROW_NUMBER() OVER (PARTITION BY track_id ORDER BY timestamp) AS row_num
    FROM detection_logs
    WHERE detection_class = '{class_name}'"""
    if upper_color != "":
        query += f"""AND JSON_EXTRACT(metadata, '$.upper.{upper_color}') IS NOT NULL
      AND JSON_EXTRACT(metadata, '$.lower.{lower_color}') IS NOT NULL
      AND (
        JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.upper')) LIKE '%"{upper_color}":%'
        AND JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.lower')) LIKE '%"{lower_color}":%'
      ) """
    else:
        query += f""" AND JSON_EXTRACT(metadata, '$.lower.{lower_color}') IS NOT NULL
      AND (
        JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.lower')) LIKE '%"{lower_color}":%'
      ) """
    query += f""" AND timestamp BETWEEN '{initial_timestamp}' AND '{final_timestamp}' ) AS dt
WHERE row_num = 1 ;"""
    # query = f"""SELECT camera_ip,timestamp,camera_id,detection_class,id FROM detection_logs WHERE detection_class IN {vehicle} AND camera_id IN {ids} ;"""
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    print(result)
    conn.close()
    return result


def find_area_by_id_from_db(cam_id: str):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = f"""SELECT area FROM cameras WHERE camera_id = {cam_id};"""
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result


@app.websocket("/suspect_search/{usage}")
async def websocket_search(websocket: WebSocket, usage: str):
    await websocket.accept()
    try:
        query_data = await websocket.receive_text()
        query_params = json.loads(query_data)
        class_name = query_params.get("vehicleType")
        lower_color = query_params.get("vehicleColor")
        upper_color = query_params.get("clothingColor")
        # camera_details_param = query_params.get("addedCameras")
        # ids_param = [e.get("id") for e in camera_details_param]
        # ips_param = [e.get("cameraIp") for e in camera_details_param]
        # area_param = [e.get("area") for e in camera_details_param]
        # vehicle_list = f"({', '.join(repr(cls) for cls in vehicle_param)})"
        # ids_list = f"({', '.join(repr(cls) for cls in ids_param)})"
        # vehicle_list_i = f"[{', '.join(repr(cls) for cls in vehicle_param)}]"
        # ids_list_i = f"[{', '.join(repr(cls) for cls in ids_param)}]"
        # ips_list = f"[{', '.join(repr(cls) for cls in ips_param)}]"
        # area_list = f"[{', '.join(repr(cls) for cls in area_param)}]"
        initial_timeframe = query_params.get("startDate")
        final_timestamp = query_params.get("endDate")
        if usage == "search":
            suspect_search_operation_db_save(
                "all",
                "all",
                "all",
                "class_color",
                class_name,
                lower_color,
                upper_color,
                initial_timeframe,
                final_timestamp,
            )
        if class_name == "":
            await websocket.send_text(
                json.dumps({"error": "vehicle query parameter is required"})
            )
            await websocket.close()
            return
        while True:
            result = suspect_search_sql(
                class_name,
                upper_color,
                lower_color,
                "*",
                initial_timeframe,
                final_timestamp,
            )
            print(result)
            if result:
                result = jsonable_encoder(result)
                for item in result:
                    item["area"] = (
                        f"{find_area_by_id_from_db(str(item['camera_id']))[0]['area']}"
                    )
                try:
                    print(result)
                    await websocket.send_text(json.dumps(result))
                except Exception as e:
                    notification.notify(
                        title="ALERT⚠️",
                        message="Some Restricted Vehicle Enter in the Zone",
                        app_name="Notification App",
                        timeout=10,
                    )
            # initial_timeframe = datetime.fromisoformat(initial_timeframe)
            initial_timeframe = (datetime.now()).replace(microsecond=0)
            final_timestamp_dt = datetime.fromisoformat(final_timestamp)
            if initial_timeframe >= final_timestamp_dt:
                await websocket.send_text(json.dumps([{"end": "end"}]))
                await websocket.close()
                break
            else:
                # await websocket.send_text(json.dumps({"status": "not found"}))
                await asyncio.sleep(1)  # Check every 5 seconds
            initial_timeframe = initial_timeframe.strftime("%Y-%m-%dT%H:%M:%S")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        # await websocket.close()




def convert_to_image_format(image_array):
    pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

class srch_req(BaseModel):
    number_plate: str
    vehicleColor: str
    vehicleType: str
    clothingColor:str
    startDate: str
    endDate: str
# {'number_plate': '', 'vehicleColor': 'white', 'vehicleType': 'bus', 'clothingColor': '', 'startDate': '2024-07-07T12:00:00', 'endDate': '2024-07-07T12:02:00'}

@app.post("/search_detection")
def fetch_db(data: srch_req):
    data = {**data.dict()}
    response = []
    con = mysql.connector.connect(
        host="localhost", user="root", password="", database="logs"
    )
    cursor = con.cursor()
    if data["number_plate"] != "":
        query = f"SELECT camera_ip,timestamp,number_plate FROM anpr_logs WHERE LEVENSHTEIN(number_plate, '{data['number_plate']}') <= 2;"
        cursor.execute(query)
        table = cursor.fetchall()
        for row in table:
            r = {
                "camera_ip": row[0],
                "timestamp": row[1],
            }
            if data["number_plate"] != "":
                r["data"] = {row[2]}
            else:
                r["data"] = f"{data['vehicleColor']} {data['vehicleType']}"
            response.append(r)
        cursor.close()

        con.close()
    else:
        result = suspect_search_sql(
            data['vehicleType'],
            data['clothingColor'],
            data['vehicleColor'],
            "*",
            data['startDate'],
            data['endDate'],
        )
        print(result)
        result = jsonable_encoder(result)
        for item in result:
            # print(item)
            # video_path = f"/home/devendra/data/{item['camera_ip']}.MKV"
            # timestamp_str = item['timestamp']
            # datetime_obj = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
            # formatted_string = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
            # video_start_time = datetime(2024, 7, 7, 12, 0, 0)
            # frame = extract_frame(video_path, formatted_string, video_start_time)
            # roi_image_buffer = convert_to_image_format(frame)
            # encoded_image = base64.b64encode(roi_image_buffer.getvalue()).decode("utf-8")
            r = {
                "camera_ip": item['camera_ip'],
                "timestamp": item['timestamp'],
                "data": f"{item['detection_class']}:{item['metadata']}",
                # "img":encoded_image
            }
            response.append(r)
        # response.get("get_that_db")
    return response


# WebSocket endpoint to process video and send predicted counts
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket, db: SessionLocal = Depends(get_db)):
#     await websocket.accept()

#     try:
#         while True:
#             data = await websocket.receive_json()
#             video_path = data.get("video_path")
#             camera_ip = data.get("camera_ip")

#             # Process the video and send the counts
#             await process_video(websocket, video_path, camera_ip, db)

#     except WebSocketDisconnect:
#         print("Client disconnected")


# @app.websocket("/cclogs")
# async def websocket_endpoint(websocket: WebSocket, db: SessionLocal = Depends(get_db)):
#     from ccapi.db_sql import process_video
#     await websocket.accept()

#     try:
#         while True:
#             data = await websocket.receive_json()
#             video_path = data.get("video_path")
#             camera_ip = data.get("camera_ip")

#             # Process the video and send the counts
#             await process_video(websocket, video_path, camera_ip, db)

#     except WebSocketDisconnect:
#         print("Client disconnected")


# crowd count logs search
@app.websocket("/ccsearch/{usage}")
async def websocket_search(websocket: WebSocket, usage: str):
    from ccapi.test import crowd_count_get, find_area_by_ip

    await websocket.accept()
    try:
        query_data = await websocket.receive_text()
        query_params = json.loads(query_data)
        camera_details_param = query_params.get("addedCameras", [])
        threshold_value = int(query_params.get("thresholdInput", 0))
        ids_param = [e.get("id") for e in camera_details_param]
        # ips_param = [e.get("cameraIp") for e in camera_details_param]
        area_param = [e.get("area") for e in camera_details_param]
        ips_param = [e.get("cameraIp") for e in camera_details_param]
        ids_list = f"({', '.join(repr(cls) for cls in ids_param)})"
        vehicle_list_i = f"[{threshold_value}]"
        ids_list_i = f"[{', '.join(repr(cls) for cls in ids_param)}]"
        ips_list = f"[{', '.join(repr(cls) for cls in ips_param)}]"
        area_list = f"[{', '.join(repr(cls) for cls in area_param)}]"
        initial_timeframe = query_params.get("start_time")
        final_timestamp = query_params.get("end_time")
        if usage == "map":
            operation_db_save(
                ips_list,
                ids_list_i,
                area_list,
                "crowd",
                vehicle_list_i,
                initial_timeframe,
                final_timestamp,
            )
        # print(ips_list, ids_list_i,area_list,"crowd",vehicle_list_i,initial_timeframe,final_timestamp)
        while True:
            result = crowd_count_get(ips_param,threshold_value, initial_timeframe, final_timestamp)
            if result:
                result = jsonable_encoder(result)
                for item in result:
                    item["area"] = find_area_by_ip(
                        camera_details_param, item["camera_ip"]
                    )
                    if item.get("crowd_count", 0) > threshold_value:
                        try:
                            await websocket.send_text(json.dumps(item))
                        except Exception as e:
                            notification.notify(
                                title="ALERT⚠️",
                                message="OVER CROWDING",
                                app_name="Notification App",
                                timeout=10,
                            )

            initial_timeframe = (datetime.now()).replace(microsecond=0)
            final_timestamp_dt = datetime.fromisoformat(final_timestamp)

            if (initial_timeframe) >= final_timestamp_dt:
                await websocket.send_text(json.dumps([{"end": "end"}]))
                await websocket.close()
                break
            else:
                await asyncio.sleep(1)
            initial_timeframe = initial_timeframe.strftime("%Y-%m-%dT%H:%M:%S")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")


# VEHICLE COUNT
@app.websocket("/vehicle_count/{usage}")
async def websocket_search(websocket: WebSocket, usage: str):
    from db.vehicle_count import vehicle_count_get, find_area_by_ip

    await websocket.accept()
    try:
        query_data = await websocket.receive_text()
        query_params = json.loads(query_data)
        camera_details_param = query_params.get("addedCameras", [])
        threshold_value = int(query_params.get("thresholdInput", 0))
        ids_param = [e.get("id") for e in camera_details_param]
        # ips_param = [e.get("cameraIp") for e in camera_details_param]
        area_param = [e.get("area") for e in camera_details_param]
        ips_param = [e.get("cameraIp") for e in camera_details_param]
        ids_list = f"({', '.join(repr(cls) for cls in ids_param)})"
        vehicle_list_i = f"[{threshold_value}]"
        ids_list_i = f"[{', '.join(repr(cls) for cls in ids_param)}]"
        ips_list = f"[{', '.join(repr(cls) for cls in ips_param)}]"
        area_list = f"[{', '.join(repr(cls) for cls in area_param)}]"
        initial_timeframe = query_params.get("start_time")
        final_timestamp = query_params.get("end_time")
        if usage == "map":
            operation_db_save(
                ips_list,
                ids_list_i,
                area_list,
                "vehicle_count",
                vehicle_list_i,
                initial_timeframe,
                final_timestamp,
            )
        # print(ips_list, ids_list_i,area_list,"crowd",vehicle_list_i,initial_timeframe,final_timestamp)
        while True:
            result = vehicle_count_get(
                ips_param, threshold_value, initial_timeframe, final_timestamp
            )
            if result:
                result = jsonable_encoder(result)
                for item in result:
                    item["area"] = find_area_by_ip(
                        camera_details_param, item["camera_ip"]
                    )
                    if item.get("vehicle_count", 0) > threshold_value:
                        try:
                            await websocket.send_text(json.dumps(item))
                        except Exception as e:
                            notification.notify(
                                title="ALERT⚠️",
                                message="OVER-CROWDING",
                                app_name="Notification App",
                                timeout=10,
                            )

            initial_timeframe = (datetime.now()).replace(microsecond=0)
            final_timestamp_dt = datetime.fromisoformat(final_timestamp)

            if (initial_timeframe) >= final_timestamp_dt:
                await websocket.send_text(json.dumps([{"end": "end"}]))
                await websocket.close()
                break
            else:
                await asyncio.sleep(1)
            initial_timeframe = initial_timeframe.strftime("%Y-%m-%dT%H:%M:%S")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")


# get operations list
@app.post("/operations")
def operations():
    from db.observation_logs import fetch_operation_logs

    return fetch_operation_logs()


# vehicle count


# gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5444
# cloudflared tunnel --url http://localhost:5444
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5889)
    print("CICD tested")




