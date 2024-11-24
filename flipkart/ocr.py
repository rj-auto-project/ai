import cv2
import pytesseract
import numpy as np

# Set the tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Resize the image
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)  # Double size for clarity

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Binarization using Otsu's method
    _, thresh_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, thresh_img

def extract_text_from_image(image, psm_mode):
    # Test different page segmentation modes
    custom_config = f'--oem 3 --psm {psm_mode} --dpi 300'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def main(image_path):
    # Step 1: Preprocess the image
    original_img, processed_img = preprocess_image(image_path)

    # Test different PSM modes
    for psm_mode in [3, 6, 11, 12]:  # Test modes 3, 6, 11, 12
        print(f"\nTesting with PSM mode {psm_mode}:")
        detected_text = extract_text_from_image(processed_img, psm_mode)
        print(f"Detected Text (PSM {psm_mode}):\n{detected_text}")

    # Optionally, save the processed image for review
    cv2.imwrite('processed_image_otsu.jpg', processed_img)

if _name_ == "_main_":
    image_path = 'image.png'  # Change this to your image path
    main(image_path)