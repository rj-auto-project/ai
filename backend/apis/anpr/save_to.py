import pandas as pd
import pymysql
import re
from datetime import datetime
import difflib

# Function to connect to the MySQL database
def create_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='pass',
        database='RIC'
    )


def fetch_rider_details_for_number_plates(conn, number_plates):
    with conn.cursor() as cursor:
        # Construct the SQL query to fetch rider details for specific number plates
        query = "SELECT * FROM riders WHERE number_plate IN (%s)"
        in_clause = ', '.join(['%s'] * len(number_plates))
        query = query % in_clause
        cursor.execute(query, number_plates)
        result = cursor.fetchall()
    return result

# Function to validate license plate format
def validate_license_plate(license_plate):
    pattern = r'^[A-Za-z]{2}\d{2}[A-Za-z]{2}\d{4}$'
    if re.match(pattern, license_plate):
        return True
    else:
        return False

# Main function
def main(csv_file):
    conn = create_connection()
    try:
        if conn.open:
            print("Connected to the database")
            
            # Read license numbers from CSV file and handle parser errors
            try:
                df = pd.read_csv(csv_file)
            except pd.errors.ParserError as e:
                print(f"Error parsing CSV file: {e}")
                return
            
            license_numbers = []
            for idx, row in df.iterrows():
                try:
                    license_number = row['license_number']
                    if validate_license_plate(license_number):
                        license_numbers.append(license_number)
       
                except Exception as e:
                    print(f"Error processing license plate {license_number}: {e}")
                    continue
            
            # Fetch rider details for matching number plates
            if license_numbers:
                rider_details = fetch_rider_details_for_number_plates(conn, license_numbers)
                for detail in rider_details:
                    print(detail)
            else:
                print("No valid license numbers found.")

        else:
            print("Failed to connect to the database")
    finally:
        conn.close()

def check_license_number(file_path, license_number):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        output_file_path = "D:/local-intelligent-cameras/backend/apis/num_plate_searching/final.csv"
        # Check if the license number exists in the DataFrame
        if license_number in df['Number_Plate'].values:
            # Fetch the details of the row with the specified license number
            details = df[df['Number_Plate'] == license_number]
            
            # Add a new column 'Time' with the current time
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            details['Time'] = current_time
            
            # Write the result to another CSV file
            # details.to_csv(output_file_path, index=False)
            details.to_csv(output_file_path, mode='a', index=False, header=not pd.io.common.file_exists(output_file_path))
            print(details)
        else:
            closest_match = difflib.get_close_matches(license_number, df['Number_Plate'].tolist(), n=1)
            if closest_match:
                closest_license_number = closest_match[0]
                closest_details = df[df['Number_Plate'] == closest_license_number].to_dict(orient='records')[0]
                return closest_details
            else:
                return None
    except Exception as e:
        return str(e)

# Example usage


if __name__ == "__main__":
    file_path = 'D:/local-intelligent-cameras/backend/apis/num_plate_searching/recognized.csv'
    license_number = 'RJ11CA8282'
    result = check_license_number(file_path, license_number)
    print(result)