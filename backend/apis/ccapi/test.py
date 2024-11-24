from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import mysql.connector
import json
from fastapi.encoders import jsonable_encoder
import asyncio
from datetime import datetime, timedelta

app = FastAPI()

db_config = {"host": "localhost", "user": "root", "password": "", "database": "logs"}

def get_db_connection():
    return mysql.connector.connect(**db_config)

def crowd_count_get(ips,threshold, initial_timestamp, final_timestamp):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    formatted_ips = ', '.join(['%s'] * len(ips))
    query = f"""SELECT crowd_count, timestamp, camera_ip 
                FROM crowd_count
                WHERE timestamp BETWEEN %s AND %s 
                AND camera_ip IN ({formatted_ips}) AND crowd_count > {threshold};"""
    params = [initial_timestamp, final_timestamp] + ips
    print(f"Executing query: {query} with params: {params}")
    cursor.execute(query, params)
    result = cursor.fetchall()
    print(result)
    cursor.close()
    conn.close()
    return result

def find_area_by_ip(cameras, search_ip):
    for camera in cameras:
        if camera["cameraIp"] == search_ip:
            return camera["area"]
    return None

def operation_db_save(ips_list,ids_list,area_list,detection,detection_data,initial_timestamp,final_timestamp):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = "INSERT INTO operation_logs (camera_id, camera_ip, area, detection, detection_data, initial_timestamp, final_timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s);"
    values = (f'{ids_list}', f'{ips_list}', f'{area_list}', f'{detection}', f'{detection_data}', f'{initial_timestamp}', f'{final_timestamp}')
    print(query,values)
    cursor.execute(query,values)
    conn.commit()
    cursor.close()
    conn.close()
    return "success"

@app.websocket("/ws/crowd_count")
async def websocket_search(websocket: WebSocket):
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
        operation_db_save(ips_list, ids_list_i,area_list,"crowd",vehicle_list_i,initial_timeframe,final_timestamp)
        # print(ips_list, ids_list_i,area_list,"crowd",vehicle_list_i,initial_timeframe,final_timestamp)
        while True:
            result = crowd_count_get(ips_param, initial_timeframe, final_timestamp)
            if result:
                result = jsonable_encoder(result)
                for item in result:
                    item["area"] = find_area_by_ip(camera_details_param, item["camera_ip"])
                    if item.get("crowd_count", 0) > threshold_value:
                        try:
                            await websocket.send_text(json.dumps(item))
                        except Exception as e:
                            print(f"Error sending data: {e}")
            
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
