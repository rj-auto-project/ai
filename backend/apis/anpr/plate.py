import mysql.connector
import json
import pandas as pd

def fetch_data_from_db(registration_number):
    # Database connection details
    db_config = {
        'user': 'root',
        'password': '',
        'host': '127.0.0.1',
        'database': 'license_plate'
    }

    # Connect to the database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    # Query to fetch data based on registration number
    query = "SELECT * FROM vehicle WHERE Plate = %s"
    cursor.execute(query, (registration_number,))

    # Fetch the result
    result = cursor.fetchone()

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return result

def main():
    # Specify the filename
    filename = 'test.csv'
    # Specify the column name you want to extract
    column_name = 'license_number'

    try:
        # Read the CSV file using pandas with additional parameters to handle common issues
        df = pd.read_csv(filename, on_bad_lines='skip', quoting=3)

        # Extract the column data and convert it to a list
        column_data = df[column_name].tolist()

        # Remove duplicates by converting the list to a set and then back to a list
        column_data = list(set(column_data))

        # Select only the first four unique entries
        registration_numbers = column_data[:4]
    except pd.errors.ParserError as e:
        print(f'Error parsing the CSV file: {e}')
        return
    except KeyError:
        print(f'Column "{column_name}" does not exist in the CSV file.')
        return
    print(registration_numbers)
    # Dictionary to store results
    results = {}

    # Iterate over each registration number and fetch data
    for reg_number in registration_numbers:
        data = fetch_data_from_db(reg_number)
        if data:
            results[reg_number] = data
        else:
            results[reg_number] = "No data found"

    # Convert the results to JSON
    json_data = json.dumps(results, indent=4)

    # Write JSON data to a file
    with open('output.json', 'w') as json_file:
        json_file.write(json_data)

    print("Data has been written to output.json")

if __name__ == "__main__":
    main()
