from matplotlib.backend_bases import cursors
from fastapi import FastAPI
from pydantic import BaseModel
import mysql.connector
import os
import uvicorn
import json


def get_cam_coords(auth_id):
    con = mysql.connector.connect(
        host="localhost", user="root", password="", database="xyz"
    )
    cursor = con.cursor()
    # if data["areas"][0] == "*":
    query2 = f"select camera_ip, location, id from dvr where auth_id = '{auth_id}'"
    cursor.execute(query2)
    table = cursor.fetchall()
    response = []
    for row in table:
        data = {}
        dvr_info = {
            "id": f"{row[0]}",
            "status": f"{row[1]}",
            "dvr_coord": json.loads(row[2]),
            "ownershsip": f"{row[3]}",
        }
        cam_ids = row[4].split(",")
        cam_loc = json.loads(row[5])
        cam_status = row[6].split(",")
        cam_data_arr = []
        for i, cam_id in enumerate(cam_ids):
            if cam_id != "":
                cam_info = {
                    "cam_id": cam_id,
                    "cam_coord": cam_loc[i],
                    "cam_status": f"{cam_status[i]}",
                }
                cam_data_arr.append(cam_info)
        dvr_info["cam_data"] = cam_data_arr
        data["dvr_data"] = dvr_info
        response.append(data)
    cursor.close()

    con.close()
    return response


# print(get_dvr_coords("ABC123"))
