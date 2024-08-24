import mysql.connector

def fetch_db():
    response = []
    con = mysql.connector.connect(host="localhost", user="root", password="intelligentCams", database="xyz")
    cursor = con.cursor()
	# if data["areas"][0] == "*":
	# query2 = f"select * from dvr where auth_id = {data['auth_id']}"
    query2 = f"select * from dvr"
	# else:
	# 	query2 = f"select * from dvr where auth_id = {data['auth_id']} and pincode in {data['areas']}"
    cursor.execute(query2)

    table = cursor.fetchall()
    print(table)
    # print('\n Table Data:')
    # for row in table:
    #     r = {
    #             "DVR_loc":f"{row[0]}",
    #             "Associated Cameras":f"{row[1]}",
    #             "Owenership":f"{row[2]}",
    #             "owner_name": f"{row[4]}",
    #             "DVR_id":f"{row[5]}",
    #             "DVR_status":f"{row[6]}",
    #         }
    #     response.append(r)
    cursor.close()

    con.close()
    return response

fetch_db()