import pandas as pd
from datetime import datetime, timedelta
import ast
def load_results_from_excel(input_path='detection_results.xlsx'):
    df = pd.read_excel(input_path)
    return df

def search_detections(start_time, end_time, class_id, input_path='detection_results.xlsx'):
    # Load the detection results from the Excel file
    df = load_results_from_excel(input_path)

    # Convert the 'timestamp' column to datetime objects for comparison
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S').dt.time

    # Convert start_time and end_time to time objects
    start_time = datetime.strptime(start_time, '%H:%M:%S').time()
    end_time = datetime.strptime(end_time, '%H:%M:%S').time()

    # Filter the dataframe based on the specified time range and class_id
    filtered_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

    # Extract detections of the specified class
    detections = []
    for _, row in filtered_df.iterrows():
        tensor_data = ast.literal_eval(row['tensor'])
        tensor_class_id = tensor_data[-1] # Assuming class_id is the last element in the tensor list

        if tensor_class_id == class_id:
            detections.append({
                'timestamp': row['timestamp'].strftime('%H:%M:%S'),
                'class': tensor_class_id
            })

    return detections

# Example usage
if __name__ == '__main__':
    start_time = input('Enter the start time: ')  # Start time in HH:MM:SS format
    end_time = input('Enter the end time: ')    # End time in HH:MM:SS format
    class_id = int(input("Enter the id of item to search: "))           # Class ID to search for

    detections = search_detections(start_time, end_time, class_id)
    for detection in detections:
        print(detection)
