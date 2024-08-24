# import mysql.connector
# from fastapi import FastAPI
# from pydantic import BaseModel
# import uvicorn

# app = FastAPI()

# class srch_req(BaseModel):
#     number_plate: str
#     vehicleColor: str
#     vehicleType: str
#     startDate: str
#     endDate: str

# @app.post("/srch_detection")
# def fetch_db(data: srch_req):
#     data = {**data.dict()}
#     response = []
#     con = mysql.connector.connect(
#         host="localhost", user="root", password="", database="logs"
#     )
#     print(data)
#     cursor = con.cursor()
#     if data["number_plate"]!="":
#         query = f"SELECT camera_ip,timestamp,number_plate FROM anpr_logs WHERE LEVENSHTEIN(number_plate, '{data['number_plate']}') <= 2;"
#     else:
#         query = f"SELECT camera_ip,timestamp FROM detection_logs WHERE detection_class = '{data['vehicleType']}' AND JSON_EXTRACT(metadata, '$.color') = '{data['vehicleColor']}';"
#     # "detection_class = 'tractor' AND JSON_EXTRACT(metadata, '$.color') = 'red';"
#     cursor.execute(query)
#     table = cursor.fetchall()
#     for row in table:
#         r = {
#             "camera_ip":row[0],
#             "timestamp":row[1],
#         }
#         if data["number_plate"]!="":
#             r["data"] = {row[2]}
#         else:
#             r["data"] = f"{data['vehicleColor']} {data['vehicleType']}"
#         response.append(r)
#     cursor.close()

#     con.close()
#     return response

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=5667)


import json
import mysql.connector
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from datetime import datetime, timedelta
from fastapi.encoders import jsonable_encoder
import asyncio

app = FastAPI()

# Database connection details
db_config = {"host": "localhost", "user": "root", "password": "", "database": "logs"}


def get_db_connection():
    return mysql.connector.connect(**db_config)


# Function to search for a specific type of vehicle


def search_vehicle(vehicle, color, initial_timestamp, final_timestamp):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = f"""SELECT camera_ip,timestamp,id FROM detection_logs WHERE detection_class = '{vehicle}' AND JSON_EXTRACT(metadata, '$.color') = '{color}' AND timestamp BETWEEN '{initial_timestamp}' AND '{final_timestamp}';"""
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    print(result)
    return result


@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    await websocket.accept()
    try:
        query_data = await websocket.receive_text()
        query_params = json.loads(query_data)
        vehicle = query_params.get("vehicle")
        color = query_params.get("color")
        initial_timeframe = query_params.get("initial_timestamp")
        final_timestamp = query_params.get("final_timestamp")
        if not vehicle:
            await websocket.send_text(
                json.dumps({"error": "vehicle query parameter is required"})
            )
            await websocket.close()
            return
        while True:
            result = search_vehicle(vehicle, color, initial_timeframe, final_timestamp)
            if result:
                result = jsonable_encoder(result)
                await websocket.send_text(json.dumps(result))
            final_timestamp_dt = datetime.fromisoformat(final_timestamp)
            if datetime.now() > final_timestamp_dt:
                await websocket.send_text(json.dumps([{"end": "end"}]))
                await websocket.close()
                break
            else:
                # await websocket.send_text(json.dumps({"status": "not found"}))
                await asyncio.sleep(5)  # Check every 5 seconds
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        # await websocket.close()









# Restrict any type of vehicle from an area


def restrict_vehicle_sql_search(vehicle, ids, initial_timestamp, final_timestamp):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = f"""SELECT camera_ip,timestamp,camera_id FROM detection_logs WHERE detection_class IN {vehicle} AND camera_id IN {ids} AND timestamp BETWEEN '{initial_timestamp}' AND '{final_timestamp}';"""
    print(query)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

def operation_db_save(ips_list,ids_list,area_list,detection,detection_data,initial_timestamp,final_timestamp):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = "INSERT INTO operation_logs (camera_id, camera_ip, area, detection, detection_data, initial_timestamp, final_timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s);"
    values = (f'{ids_list}', f'{ips_list}', f'{area_list}', f'{detection}', f'{detection_data}', f'{initial_timestamp}', f'{final_timestamp}')
    cursor.execute(query,values)
    conn.commit()
    cursor.close()
    conn.close()
    return "success"
def find_area_by_id(obj, search_id):
    for camera in obj["addedCameras"]:
        if camera["id"] == search_id:
            return camera["area"]
    return None

from plyer import notification

@app.websocket("/ws/restrict_vehicle")
async def websocket_search(websocket: WebSocket):
    await websocket.accept()
    try:
        query_data = await websocket.receive_text()
        print(query_data)
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
        operation_db_save(ips_list, ids_list_i,area_list,"vehicle",vehicle_list_i,initial_timeframe,final_timestamp)
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
            print(result)
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

import mysql.connector
import ast

@app.post("/operations")
def fetch_operation_logs():
    response = []
    con = mysql.connector.connect(
        host="localhost", user="root", password="", database="logs"
    )
    cursor = con.cursor()
    # if filter_by == "area":
    query2 = f"select * from operation_logs"
    cursor.execute(query2)
    table = cursor.fetchall()
    for row in table:
        data = {
           "camera_id":eval(row[1]),
           "camera_ip":eval(row[2]),
           "area":eval(row[3]),
           "detection":row[4],
           "detection_data":eval(row[5]),
           "initial_timestamp":row[6],
           "final_timestamp":row[7] 
        }
        response.append(data)
    cursor.close()
    con.close()
    print(response)
    return response


@app.post("/t/{a}")
async def t(a):
    return a

# To run the server: `uvicorn your_script_name:app --reload`


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5777)
