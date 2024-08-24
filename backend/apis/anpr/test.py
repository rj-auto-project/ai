# import csv

# def levenshtein_distance(s1, s2):
#     if len(s1) < len(s2):
#         return levenshtein_distance(s2, s1)

#     if len(s2) == 0:
#         return len(s1)

#     previous_row = range(len(s2) + 1)
#     for i, c1 in enumerate(s1):
#         current_row = [i + 1]
#         for j, c2 in enumerate(s2):
#             insertions = previous_row[j + 1] + 1
#             deletions = current_row[j] + 1
#             substitutions = previous_row[j] + (c1 != c2)
#             current_row.append(min(insertions, deletions, substitutions))
#         previous_row = current_row

#     return previous_row[-1]

# def find_best_match(user_number, csv_path):
#     best_matches = []
#     min_distance = float('inf')

#     with open(csv_path, mode='r') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             number_plate = row[0]
#             distance = levenshtein_distance(user_number, number_plate)
#             if distance < min_distance:
#                 min_distance = distance
#                 best_matches = [number_plate]
#             elif distance == min_distance:
#                 best_matches.append(number_plate)

#     return best_matches if min_distance in [1, 2] else []

# # Example usage
# user_provided_number = "RJ71GB6827"
# csv_path = "D:/local-intelligent-cameras/backend/apis/num_plate_searching/vehicle.csv"

# matches = find_best_match(user_provided_number, csv_path)
# print(f"Best matches for '{user_provided_number}': {matches[0]}")


# INSERTION

from fastapi import FastAPI
import uvicorn
app = FastAPI()

@app.get("/srch_license_plate/")
async def print_data(license_plate: str):
    # Print the received data to the console
    print(f"Received license_plate: {license_plate}")
    
    # Return a response to the frontend
    return {"message": f"Received license_plate: {license_plate}"}

# To run the server, use the command: uvicorn filename:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5667)