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
# [
#     "upper":{
#         "charcoal" : 0.23,
#         "red":0.45,
#         "black":0.26
#     },
#     "lower":{
#         "charcoal" : 0.23,
#         "red":0.45,
#         "black":0.26
#     }
# ]

def vehicle_count_get(ips,threshold_value, initial_timestamp, final_timestamp):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    formatted_ips = ", ".join(["%s"] * len(ips))
    query = f"""SELECT vehicle_count, timestamp, camera_ip 
                FROM vehicle_count_logs 
                WHERE timestamp BETWEEN %s AND %s 
                AND camera_ip IN ({formatted_ips})
                AND vehicle_count > {threshold_value};"""
    params = [initial_timestamp, final_timestamp] + ips
    # print(f"Executing query: {query} with params: {params}")
    print(query,params)
    cursor.execute(query, params)
    result = cursor.fetchall()
    print(query, params, result)
    cursor.close()
    conn.close()
    return result


def find_area_by_ip(cameras, search_ip):
    for camera in cameras:
        if camera["cameraIp"] == search_ip:
            return camera["area"]
    return None
