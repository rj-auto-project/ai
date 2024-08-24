import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import os
import json
import base64
import dash_bootstrap_components as dbc
from datetime import datetime

# Set of predefined colors
COLOR_OPTIONS = ['black', 'blue', 'charcoal', 'green', 'red', 'silver', 'white', 'yellow', 'MIXED']
LABELED_DIRECTORIES_FILE = 'labeled_directories.json'

# Load the list of labeled directories with timestamps
def load_labeled_directories():
    if os.path.exists(LABELED_DIRECTORIES_FILE):
        with open(LABELED_DIRECTORIES_FILE, 'r') as f:
            return json.load(f)
    return []

# Save the list of labeled directories with timestamps
def save_labeled_directories(labeled_directories):
    with open(LABELED_DIRECTORIES_FILE, 'w') as f:
        json.dump(labeled_directories, f)

# Function to encode image to base64 for displaying in Dash
def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Function to load color centers from JSON
def load_color_centers(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Function to save color centers to JSON
def save_color_centers(json_path, color_centers):
    with open(json_path, 'w') as f:
        json.dump(color_centers, f)

# Function to create directory dropdown options
def get_directory_options(base_path='labeling'):
    options = []
    labeled_directories = set([item['path'] for item in load_labeled_directories()])
    for root, dirs, files in os.walk(base_path):
        if 'color_centers.json' in files and root not in labeled_directories:
            options.append({'label': os.path.relpath(root, base_path), 'value': root})
    return options

# Function to create review directory dropdown options
def get_review_directory_options(base_path='ColoursRJ/labeling'):
    labeled_directories = load_labeled_directories()
    options = sorted(labeled_directories, key=lambda x: x['time'], reverse=True)
    return [{'label': os.path.relpath(item['path'], base_path), 'value': item['path']} for item in options]

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1('Image Labeling Dashboard'),
    html.Label('Select Folder to Label:'),
    dcc.Dropdown(id='folder-dropdown', options=get_directory_options(), placeholder='Select a folder...'),
    html.Div(id='images-div'),
    html.Button('Save Labels', id='save-button', n_clicks=0),
    html.Hr(),
    html.H2('Review Labeled Folders'),
    html.Label('Select Folder to Review:'),
    dcc.Dropdown(id='review-folder-dropdown', options=get_review_directory_options(), placeholder='Select a folder...'),
    html.Div(id='review-images-div'),
    html.Button('Save Review Labels', id='save-review-button', n_clicks=0)
])

@app.callback(
    Output('images-div', 'children'),
    Input('folder-dropdown', 'value')
)
def display_images(folder_path):
    if folder_path is None:
        return []

    images = []
    # Display the original image at the top
    original_image_path = os.path.join(folder_path, 'original.png')
    encoded_original_image = encode_image(original_image_path)
    images.append(html.Div([
        html.H3('Original Image'),
        html.Img(src='data:image/png;base64,{}'.format(encoded_original_image), style={'width': '400px'}),
    ], style={'text-align': 'center', 'margin-bottom': '20px'}))

    color_centers_path = os.path.join(folder_path, 'color_centers.json')
    color_centers = load_color_centers(color_centers_path)

    # Display the cluster center images in a grid
    grid_elements = []
    for i, center in enumerate(color_centers):
        img_path = os.path.join(folder_path, 'color_center_{}.png'.format(i + 1))
        encoded_image = encode_image(img_path)
        img_element = html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '100%'}),
            dcc.Dropdown(
                id={'type': 'color-dropdown', 'index': i},
                options=[{'label': color, 'value': color} for color in COLOR_OPTIONS],
                value=center.get('label', None),
                placeholder='Select color...',
                style={'margin-top': '10px'}
            )
        ], style={'width': '200px', 'display': 'inline-block', 'margin': '10px'})
        grid_elements.append(img_element)

    images.append(html.Div(grid_elements, style={'text-align': 'center'}))

    return images

@app.callback(
    [Output('save-button', 'children'),
     Output('folder-dropdown', 'options'),
     Output('review-folder-dropdown', 'options'),
     Output('save-review-button', 'children')],
    [Input('save-button', 'n_clicks'),
     Input('save-review-button', 'n_clicks')],
    [State('folder-dropdown', 'value'),
     State({'type': 'color-dropdown', 'index': dash.dependencies.ALL}, 'value'),
     State('review-folder-dropdown', 'value'),
     State({'type': 'review-color-dropdown', 'index': dash.dependencies.ALL}, 'value')]
)
def save_labels_and_review_labels(save_n_clicks, review_n_clicks, folder_path, labels, review_folder_path, review_labels):
    ctx = dash.callback_context

    if not ctx.triggered:
        return 'Save Labels', get_directory_options(), get_review_directory_options(), 'Save Review Labels'

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'save-button' and save_n_clicks > 0 and folder_path:
        color_centers_path = os.path.join(folder_path, 'color_centers.json')
        color_centers = load_color_centers(color_centers_path)

        for i, label in enumerate(labels):
            if i < len(color_centers):
                color_centers[i]['label'] = label

        save_color_centers(color_centers_path, color_centers)

        # Update the list of labeled directories with timestamp
        labeled_directories = load_labeled_directories()
        labeled_directories.append({'path': folder_path, 'time': datetime.now().isoformat()})
        save_labeled_directories(labeled_directories)

        # Update the dropdown options
        return 'Labels Saved!', get_directory_options(), get_review_directory_options(), 'Save Review Labels'

    if button_id == 'save-review-button' and review_n_clicks > 0 and review_folder_path:
        color_centers_path = os.path.join(review_folder_path, 'color_centers.json')
        color_centers = load_color_centers(color_centers_path)

        for i, label in enumerate(review_labels):
            if i < len(color_centers):
                color_centers[i]['label'] = label

        save_color_centers(color_centers_path, color_centers)

        # Update the list of labeled directories with timestamp
        labeled_directories = load_labeled_directories()
        # Find and update the timestamp for the current folder
        for item in labeled_directories:
            if item['path'] == review_folder_path:
                item['time'] = datetime.now().isoformat()
                break
        save_labeled_directories(labeled_directories)

        # Update the dropdown options
        return 'Save Labels', get_directory_options(), get_review_directory_options(), 'Review Labels Saved!'

    return 'Save Labels', get_directory_options(), get_review_directory_options(), 'Save Review Labels'

@app.callback(
    Output('review-images-div', 'children'),
    Input('review-folder-dropdown', 'value')
)
def display_review_images(folder_path):
    if folder_path is None:
        return []

    images = []
    # Display the original image at the top
    original_image_path = os.path.join(folder_path, 'original.png')
    encoded_original_image = encode_image(original_image_path)
    images.append(html.Div([
        html.H3('Original Image'),
        html.Img(src='data:image/png;base64,{}'.format(encoded_original_image), style={'width': '400px'}),
    ], style={'text-align': 'center', 'margin-bottom': '20px'}))

    color_centers_path = os.path.join(folder_path, 'color_centers.json')
    color_centers = load_color_centers(color_centers_path)

    # Display the cluster center images in a grid with labels
    grid_elements = []
    for i, center in enumerate(color_centers):
        img_path = os.path.join(folder_path, 'color_center_{}.png'.format(i + 1))
        encoded_image = encode_image(img_path)
        img_element = html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '100%'}),
            dcc.Dropdown(
                id={'type': 'review-color-dropdown', 'index': i},
                options=[{'label': color, 'value': color} for color in COLOR_OPTIONS],
                value=center.get('label', None),
                placeholder='Select color...',
                style={'margin-top': '10px'}
            )
        ], style={'width': '200px', 'display': 'inline-block', 'margin': '10px'})
        grid_elements.append(img_element)

    images.append(html.Div(grid_elements, style={'text-align': 'center'}))

    return images

if __name__ == '__main__':
    app.run_server(debug=True)
