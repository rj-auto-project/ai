from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import os
import requests

app = Flask(__name__)

# Configure upload folder and JSON file path
UPLOAD_FOLDER = './uploads'
JSON_FILE = "/home/annone/ai/survey/temp_database.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'txt', 'mkv'}

# Available models (dropdown options)
AVAILABLE_MODELS = {
    "model1": "road model",
    "model2": "municipal model"
}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to read JSON data
def read_json():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []  # Return an empty list if the file is corrupted
    return []

# Function to write JSON data
def write_json(data):
    with open(JSON_FILE, "w") as file:
        json.dump(data, file, indent=4)

# Home route with two tabs
@app.route('/')
def home():
    return render_template('home.html')

# Route for file upload
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    # if request.method == 'POST':
    #     if 'video' not in request.files or 'text' not in request.files or 'model' not in request.form:
    #         return 'Missing required data', 400

    #     video_file = request.files['video']
    #     text_file = request.files['text']
    #     selected_model = request.form['model']

    #     if not selected_model or selected_model not in AVAILABLE_MODELS:
    #         return 'Invalid model selected', 400

    #     if video_file and allowed_file(video_file.filename):
    #         video_filename = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    #         video_file.save(video_filename)
    #     else:
    #         return 'Invalid video file', 400

    #     if text_file and allowed_file(text_file.filename):
    #         text_filename = os.path.join(app.config['UPLOAD_FOLDER'], text_file.filename)
    #         text_file.save(text_filename)
    #     else:
    #         return 'Invalid text file', 400

        # # Call the /process-video/ API
        # try:
        #     api_url = 'http://0.0.0.0:5346/process-videowewe/'  # Replace with actual API endpoint
        #     with open(video_filename, 'rb') as video, open(text_filename, 'rb') as text:
        #         response = requests.post(api_url, files={'video': video, 'text': text}, data={'model': selected_model})
            
        #     if response.status_code == 200:
        #         return jsonify({'message': 'Processing completed successfully!', 'details': response.json()})
        #     else:
        #         return f'Error during processing: {response.text}', response.status_code
        # except Exception as e:
        #     return f'Error: {str(e)}', 500
    return render_template('upload.html', models=AVAILABLE_MODELS)

# Route to display logs
@app.route('/logs', methods=['GET', 'POST'])
def logs():
    data = read_json()
    columns = list(data[0].keys()) if data else []  # Dynamically get the column headers
    return render_template('logs.html', data=data, columns=columns)

# Route to update a log entry
@app.route('/update/<detection_id>', methods=['GET', 'POST'])
def update(detection_id):
    data = read_json()
    row = next((item for item in data if item["detection_id"] == detection_id), None)
    if request.method == 'POST':
        if row:
            for key in row.keys():
                if key == "location":  # Handle location as a list
                    row["location"] = [float(request.form["lat"]), float(request.form["lng"])]
                else:
                    row[key] = request.form.get(key, row[key])  # Update other fields
            write_json(data)
        return redirect(url_for('logs'))
    return render_template("update.html", row=row)

# Route to delete a log entry
@app.route('/delete/<detection_id>')
def delete(detection_id):
    data = read_json()
    data = [item for item in data if item["detection_id"] != detection_id]
    write_json(data)
    return redirect(url_for('logs'))

if __name__ == '__main__':
    app.run(debug=True, port=5678)
