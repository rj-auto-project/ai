from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import requests

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'txt', 'mkv'}

# Available models (dropdown options)
AVAILABLE_MODELS = {
    "model1": "First Model",
    "model2": "Second Model",
    "model3": "Third Model"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html', models=AVAILABLE_MODELS)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files or 'text' not in request.files or 'model' not in request.form:
        return 'Missing required data', 400

    video_file = request.files['video']
    text_file = request.files['text']
    selected_model = request.form['model']

    if not selected_model or selected_model not in AVAILABLE_MODELS:
        return 'Invalid model selected', 400

    if video_file and allowed_file(video_file.filename):
        video_filename = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_filename)
    else:
        return 'Invalid video file', 400

    if text_file and allowed_file(text_file.filename):
        text_filename = os.path.join(app.config['UPLOAD_FOLDER'], text_file.filename)
        text_file.save(text_filename)
    else:
        return 'Invalid text file', 400

    # Call the /process-video/ API
    try:
        api_url = 'http://127.0.0.1:8000/process-video/'  # Replace with actual API endpoint
        with open(video_filename, 'rb') as video, open(text_filename, 'rb') as text:
            response = requests.post(api_url, files={'video': video, 'text': text}, data={'model': selected_model})
        
        if response.status_code == 200:
            return jsonify({'message': 'Processing completed successfully!', 'details': response.json()})
        else:
            return f'Error during processing: {response.text}', response.status_code
    except Exception as e:
        return f'Error: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True, port=5678)
