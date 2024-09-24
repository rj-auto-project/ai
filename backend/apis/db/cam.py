import mysql.connector


def fetch_cam_data():
    response = []
    con = mysql.connector.connect(
        host="localhost", user="root", password="team123", database="logs"
    )
    cursor = con.cursor()
    # if filter_by == "area":
    query2 = f"select camera_ip,camera_name,area,location,facing_angle,is_live,camera_id from cameras"
    # elif data["filter_by"] == ""
    # else:
    #     query2 = f"select location,status,area,cam_url,cam_id from camera where auth_id = '{auth_id}'"
    cursor.execute(query2)
    table = cursor.fetchall()
    for row in table:
        lat,lng = row[3].split(",")
        r = {
            "id":f"{row[6]}",
            "camera_ip": f"{row[0]}",
            "camera_name": f"{row[1]}",
            "area": f"{row[2]}",
            "location": [float(lat),float(lng)],
            "facing_angle": f"{row[4]}",
            "is_live": f"{row[5]}"
        }
        response.append(r)
    cursor.close()
    con.close()
    return response

