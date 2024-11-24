import mysql.connector,json


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
        if row[1] == "all":
            data = {
           "camera_id":f"{row[1]}",
           "camera_ip":f"{row[2]}",
           "area":f"{row[3]}",
           "detection":row[4],
           "detection_data":eval(row[5]),
           "initial_timestamp":row[6],
           "final_timestamp":row[7] 
            }
        else:
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
    return response