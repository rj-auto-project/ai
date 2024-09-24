import cv2
import numpy as np
import joblib
import time
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO

# Load the saved classifier and label encoder
classifier = joblib.load('/home/annone/ai-camera/backend/stream/pixel_classifier.joblib')
label_encoder = joblib.load('/home/annone/ai-camera/backend/stream/label_encoder.joblib')

# Function to resize image such that the max dimension is 400 pixels
def resize_image(image, crop_fraction=0.8, max_total_pixels=400):
    image = image
    # pimg = np.array(pimg)
    h, w, _ = image.shape
    crop_h, crop_w = int(h * crop_fraction), int(w * crop_fraction)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    cropped_image = image[start_h:start_h + crop_h, start_w:start_w + crop_w]

    # Check if resizing is needed
    total_pixels = cropped_image.shape[0] * cropped_image.shape[1]
    if total_pixels > max_total_pixels:
        scale_factor = (max_total_pixels / total_pixels) ** 0.5
        new_h, new_w = int(cropped_image.shape[0] * scale_factor), int(cropped_image.shape[1] * scale_factor)
        cropped_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return cropped_image

# Function to classify a new image and get top 3 colors
def classify_new_image(image, classifier, label_encoder):
    image = resize_image(image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    non_zero_pixels = image_hsv[np.any(image_hsv != [0, 0, 0], axis=-1)]
    pixel_predictions = classifier.predict(non_zero_pixels)
    
    # Count the occurrences of each predicted label
    label_counts = np.bincount(pixel_predictions, minlength=len(label_encoder.classes_))
    
    # Get the top 3 colors
    top_3_indices = np.argsort(label_counts)[-3:][::-1]
    top_3_labels = label_encoder.inverse_transform(top_3_indices)
    top_3_scores = label_counts[top_3_indices] / np.sum(label_counts)
    
    return dict(zip(top_3_labels, top_3_scores)), image

def convert_to_image_format(image_array):
    pil_image = Image.fromarray(image_array)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# Function to crop the image based on class name
def crop_and_classify(image_path, class_name, classifier, label_encoder):
    print("Cython function")
    # roi_image_buffer = convert_to_image_format(image_path)
    image = Image.open(image_path)
    width, height = image.size
    
    results = {}
    
    if class_name == "bike-rider" or class_name == "person" or class_name == "scooty-rider":
        # Crop into two halves: upper part (person) and lower part (bike)
        upper_half = image.crop((0, 0, width, height // 2))
        lower_half = image.crop((0, height // 2, width, height))
        
        # Save the cropped images
        # upper_half_path = "/home/annone/ai-camera/backend/stream/rider.png"
        # lower_half_path = "/home/annone/ai-camera/backend/stream/bike.png"
        # upper_half.save(upper_half_path)
        # lower_half.save(lower_half_path)
        
        # Classify the upper half (person)
        top_3_colors_upper, _ = classify_new_image(np.array(upper_half), classifier, label_encoder)
        # Classify the lower half (bike)
        top_3_colors_lower, _ = classify_new_image(np.array(lower_half), classifier, label_encoder)
        
        results["upper"] = top_3_colors_upper
        results["lower"] = top_3_colors_lower
        
    # elif class_name == "person":
    #     # Crop into two halves: upper body and lower body
    #     upper_half = image.crop((0, 0, width, height // 2))
    #     lower_half = image.crop((0, height // 2, width, height))
        
    #     # # Save the cropped images
    #     # upper_half_path = "upper_body.png"
    #     # lower_half_path = "lower_body.png"
    #     # upper_half.save(upper_half_path)
    #     # lower_half.save(lower_half_path)
        
    #     # Classify the upper half (upper body)
    #     top_3_colors_upper, _ = classify_new_image(np.array(upper_half), classifier, label_encoder)
    #     # Classify the lower half (lower body)
    #     top_3_colors_lower, _ = classify_new_image(np.array(lower_half), classifier, label_encoder)
        
    #     results["upper"] = top_3_colors_upper
    #     results["lower"] = top_3_colors_lower
        
    else:
        # Directly classify the whole image for other classes
        top_3_colors, _ = classify_new_image(image, classifier, label_encoder)
        results["lower"] = top_3_colors

    return results

# Function to format the output as desired
def format_output(result_dict):
    formatted_result = "{\n"
    for key, value in result_dict.items():
        formatted_result += f'    "{key}": {{\n'
        for color, score in value.items():
            formatted_result += f'        "{color}": {score:.2f},\n'
        formatted_result = formatted_result.rstrip(",\n") + "\n    },\n"
    formatted_result = formatted_result.rstrip(",\n") + "\n}"
    return formatted_result

# Example usage
image_path = '/home/annone/ai-camera/backend/stream/4.png'
class_name = "bike rider"  # Change this to "person" or other class names for testing

# output = crop_and_classify(image_path, class_name, classifier, label_encoder)
# formatted_output = format_output(output)
# print(formatted_output)
