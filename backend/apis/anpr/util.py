import string
import easyocr
import csv
import pandas as pd
from datetime import datetime, timedelta
import cv2
import os
import mysql.connector
from mysql.connector import Error


# Initialize the OCR reader
reader = easyocr.Reader(["en"], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {
    "O": "0",
    "I": "1",
    "J": "3",
    "A": "4",
    "G": "6",
    "S": "5",
    "B": "8",
    "Z": "7",
    "T": "1",
}

dict_int_to_char = {
    "0": "O",
    "1": "I",
    "3": "J",
    "4": "A",
    "6": "G",
    "5": "S",
    "8": "B",
}


def write_csv(results, output_path):
    file_exists = os.path.isfile(output_path)

    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, "a") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [
                "Frame Number",
                "Car ID",
                "Car BBox X1",
                "Car BBox Y1",
                "Car BBox X2",
                "Car BBox Y2",
                "License Plate BBox X1",
                "License Plate BBox Y1",
                "License Plate BBox X2",
                "License Plate BBox Y2",
                "License Plate Text",
                "License Plate BBox Score",
                "License Plate Text Score",
            ]
            writer.writerow(header)
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if (
                    "car" in results[frame_nmr][car_id].keys()
                    and "license_plate" in results[frame_nmr][car_id].keys()
                    and "text" in results[frame_nmr][car_id]["license_plate"].keys()
                ):
                    f.write(
                        "{},{},{},{},{},{},{}\n".format(
                            frame_nmr,
                            car_id,
                            "[{} {} {} {}]".format(
                                results[frame_nmr][car_id]["car"]["bbox"][0],
                                results[frame_nmr][car_id]["car"]["bbox"][1],
                                results[frame_nmr][car_id]["car"]["bbox"][2],
                                results[frame_nmr][car_id]["car"]["bbox"][3],
                            ),
                            "[{} {} {} {}]".format(
                                results[frame_nmr][car_id]["license_plate"]["bbox"][0],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][1],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][2],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][3],
                            ),
                            results[frame_nmr][car_id]["license_plate"]["bbox_score"],
                            results[frame_nmr][car_id]["license_plate"]["text"],
                            results[frame_nmr][car_id]["license_plate"]["text_score"],
                        )
                    )
        f.close()


def insert_into_table(values):
    try:
        # Connect to the database
        connection = mysql.connector.connect(
            host="localhost", user="root", password="", database="logs"
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Create the insert query
            insert_query = f"INSERT INTO anpr_logs (camera_id, camera_ip, number_plate, timestamp ,number_plate_confidence_score) VALUES (%s, %s, %s, %s, %s)"

            # Execute the insert query
            cursor.execute(insert_query, values)

            # Commit the transaction
            connection.commit()

    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def remove_special_chars(s, chars="!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/~` "):
    return s.strip(chars)


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 10:
        # 5 - 5
        return False
    if (
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys())
        and (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys())
        and (
            text[2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[2] in dict_char_to_int.keys()
        )
        and (
            text[3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[3] in dict_char_to_int.keys()
        )
        and (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys())
        and (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys())
        and (
            text[6] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[6] in dict_char_to_int.keys()
        )
        and (
            text[7] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[7] in dict_char_to_int.keys()
        )
        and (
            text[8] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[8] in dict_char_to_int.keys()
        )
        and (
            text[9] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[9] in dict_char_to_int.keys()
        )
    ):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """

    license_plate_ = ""
    mapping = {
        0: dict_int_to_char,
        1: dict_int_to_char,
        2: dict_char_to_int,
        3: dict_char_to_int,
        4: dict_int_to_char,
        5: dict_int_to_char,
        6: dict_char_to_int,
        7: dict_char_to_int,
        8: dict_char_to_int,
        9: dict_char_to_int,
    }
    for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def find_best_match(user_number, csv_path):
    best_matches = []
    min_distance = float("inf")

    with open(csv_path, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            number_plate = row[0]
            distance = levenshtein_distance(user_number, number_plate)
            if distance < min_distance:
                min_distance = distance
                best_matches = [number_plate]
            elif distance == min_distance:
                best_matches.append(number_plate)

    return best_matches if min_distance in [1, 2] else []


def check_license_number(file_path, license_number):
    try:
        # Load the CSV file into a DataFrame
        file_path = (
            "/home/devendra/ai-camera/backend/apis/anpr/vehicle.csv"
        )
        df = pd.read_csv(file_path)
        output_file_path = "/home/devendra/ai-camera/backend/apis/anpr/recognized.csv"
        license_number = find_best_match(license_number, file_path)[0]
        with open(
            "/home/devendra/ai-camera/backend/apis/anpr/license_plate_.txt",
            "a",
        ) as file:
            file.write("\n" + license_plate_)
        # Check if the license number exists in the DataFrame
        if license_number in df["Number_Plate"].values:
            # Fetch the details of the row with the specified license number
            details = df[df["Number_Plate"] == license_number]

            # Add a new column 'Time' with the current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            details["Time"] = current_time

            # Write the result to another CSV file
            # details.to_csv(output_file_path, index=False)
            details.to_csv(
                output_file_path,
                mode="a",
                index=False,
                header=not pd.io.common.file_exists(output_file_path),
            )
        else:
            return f"License number {license_number} does not exist in the file."
    except Exception as e:
        return str(e)


def get_creation_time(file_path):
    # Get the file creation time
    if os.name == "nt":  # Windows
        creation_time = os.path.getctime(file_path)
    else:
        stat = os.stat(file_path)
        try:
            creation_time = stat.st_birthtime
        except AttributeError:
            # For Linux, if birth time is not available, fall back to the last metadata change time
            creation_time = stat.st_mtime

    return datetime.fromtimestamp(creation_time)


def get_video_frame_time(video_path, frame_number):
    # Get the start time from the file creation time
    start_time = get_creation_time(video_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not retrieve frame rate.")
        return None

    # Calculate the elapsed time since the start of the video
    elapsed_time_seconds = frame_number / fps
    elapsed_time = timedelta(seconds=elapsed_time_seconds)

    # Calculate the actual time of the frame
    actual_time = start_time + elapsed_time

    # Release the video capture object
    cap.release()

    return actual_time


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    detections = reader.readtext(license_plate_crop)
    print(detections,"----")
    ip_id_dict = {"":"1","":"2","":"3","":"4","":"5"}
    for i, detection in enumerate(detections):
        bbox, text, score = detection

        text = text.upper().replace(" ", "")
        text = remove_special_chars(text)
        if (len(detections) > i + 1) and (
            len(detections[i][1]) + len(detections[i + 1][1])
        ) == 10:
            text = f"{detections[i][1] + detections[i + 1][1]}"
            text = text.replace("\n", "")
            text = text.upper().replace(" ", "")
            print(text)
        # if license_complies_format(text):
        #     formatted_licenses_num = format_license(text)
        #     actual_time = get_video_frame_time(video_path, frame_nmr)
        #     with open(
        #         "/home/devendra/ai-camera/backend/apis/anpr/licese_numbers.csv",
        #         "a",
        #     ) as f:
        #         f.write(
        #             "{},{},{}\n".format(video_path, formatted_licenses_num, actual_time)
        #         )
        #         f.close()
        #     file_name = os.path.basename(video_path)
        #     camera_ip, _ = os.path.splitext(file_name)
        #     values = (f"{camera_id}", f"{camera_ip}", f"{formatted_licenses_num}", f"{actual_time}", f"{score}")
        #     insert_into_table(values)
        #     check_license_number(
        #         file_path="/home/devendra/ai-camera/backend/apis/anpr/vehicle.csv",
        #         license_number=formatted_licenses_num,
        #     )
        #     return formatted_licenses_num, score

    return None, None

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
