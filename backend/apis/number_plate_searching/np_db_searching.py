
import mysql.connector
from mysql.connector import Error

def license_plate_srch(license_plate):
    try:
        # Connect to the database
        connection = mysql.connector.connect(
            host="localhost", user="root", password="", database="vehicle_db"
        )

        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            # Create the select query
            select_query = f"SELECT cam_id,number_pate,time FROM `1` WHERE number_pate = %s"

            # Execute the select query
            cursor.execute(select_query, (license_plate,))

            # Fetch all the matching rows
            result = cursor.fetchall()

            return result

    except Error as e:
        print(f"Error: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")