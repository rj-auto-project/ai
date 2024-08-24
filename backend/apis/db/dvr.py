from matplotlib.backend_bases import cursors
from fastapi import FastAPI
from pydantic import BaseModel
import mysql.connector
import os
import uvicorn
import json
# [
#     {
#         "DVR_loc":"LOCATION[NAME]",
#         "Associated Cameras":"No. of Cameras[List of all camera Coords]",
#         "Owenership":"Private/Govt.",
#         "owner_name": "name",
#         "DVR_id":"id",
#         "DVR_status":"ON/OFF",                
#     }
# ]

# show dvrs in any permissible areas
def show_dvr(auth_id:str, areas:[str]):
	"""
	auth_id : str => expects user_id of the authority who is requesting to show all the connected DVRs
	areas : [str] => expects array of the areas where user want to see the connected DVRs
	"""
	data = {"auth_id":auth_id,"areas":areas}
	response = []
	con = mysql.connector.connect( 
	host="localhost", user="root", 
	password="", database="xyz") 
	cursor = con.cursor()
	if data["areas"][0] == "*":
		query2 = f"select * from dvr where auth_id = '{data['auth_id']}'"
	else:
		query2 = f"select * from dvr where auth_id = '{data['auth_id']}' and area in {data['areas']}"
	cursor.execute(query2)

	table = cursor.fetchall()

	print('\n Table Data:')
	for row in table: 
		r = {
                "id":f"{row[0]}",
				"auth_id":f"{row[1]}",
                "name":f"{row[2]}",
                "location":f"{row[3]}",
                "status": f"{row[4]}",
                "associated_cam_id":f"{row[5]}",
                "storage":f"{row[6]}",       
				"owner_name":f"{row[7]}",
				"ownershsip":f"{row[8]}",
				"dvr_coord":json.loads(row[9]),
				"cam_coords":json.loads(row[10]),
				# "ownership":f"{row[]}"
            }
		response.append(r)
	cursor.close()

	con.close()
	return response

# delete dvr
def dlt_dvr(data):
	data = {"req":req,**item.dict()}
	con = mysql.connector.connect(
		host="localhost", user = "root",
		password="intelligentCams",database= "xyz"
	)
	cursor = con.cursor()
	query = f"delete rows from dvr where admin_id = {data['admin_id']}"
	if cursor.execute(query):
		cursor.close()
		con.close()
		return 1

def add_dvr(data):
	data = {**data.dict()}
	auth_id = data["auth_id"]
	dvr_name = data["dvr_name"]
	dvr_status = data["dvr_status"]
	dvr_location = data["dvr_location"]
	associated_cam = data["associated_cam"]
	ownership = data["ownership"]
	owner_name = data["owner_name"]
	storage = data["storage"]
	mac = data["mac"]
	con = mysql.connector.connect(
		host = "lcalhost", user = "root",
		password = 'intelligentCams', database= "xyz"
	)
	cursor = con.cursors()
	query = f"INSERT INTO `dvr` (`dvr_id`, `dvr_name`, `dvr_location`, `dvr_status`, `associated_cameras`, `storage`, `owner_name`, `auth_id`) VALUES ('{mac}', '{dvr_name}', '{dvr_location}', '{dvr_status}', '{associated_cam}', '{storage}', '{owner_name}', '00000')"
	if cursor.execute(query2):
		cursor.close()
		con.close()
		return 1

def update_dvr(data:{str}):
	return