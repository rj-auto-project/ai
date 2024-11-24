from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import supervision as sv
import subprocess
import numpy as np
import easyocr

# Initialize annotators
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoundingBoxAnnotator()
class_list = [
    "Aashirvaad-Atta",
    "Amul-Butter",
    "Apple",
    "Banana",
    "Bitter Gourd",
    "Bourn-Vita",
    "Brinjal",
    "Carrot",
    "Cauliflower",
    "Chilli",
    "Coca-Cola",
    "Coconut",
    "Colgate-Toothpaste",
    "Custard-Apple",
    "Dabur Chyawanprash",
    "Dabur-Honey",
    "Dairy-Milk-Chocolate",
    "Dettol-Hand-Wash",
    "Dragon-Fruit",
    "Ginger",
    "Grapes",
    "Harpic",
    "Himalaya-Face-Wash",
    "Kissan-Tomato-Ketchup",
    "Kurkure",
    "Lady's-Finger",
    "Lakme-Peach-Moisture-Riser",
    "Lays",
    "Loreal-Shampoo",
    "Lotus-sunscreen",
    "Maggi",
    "Mango",
    "Nescafe-Coffee",
    "Nivea-Body-Lotion",
    "Onion",
    "Orange",
    "Oreo",
    "Parachute Coconut Oil",
    "Pears-Soap",
    "Pilgrim-Hair-serum",
    "Pine Apple",
    "Pond's-Powder",
    "Potato",
    "Pro-ease",
    "Strawberry",
    "Surf-excel",
    "Tomato",
    "Ujala",
    "Vaseline",
    "Vim Bar"
]
# Initialize EasyOCR reader (set language as needed)
reader = easyocr.Reader(['en'])  # You can add more languages if needed

# Start FFmpeg to stream over RTSP
ffmpeg_process = subprocess.Popen(
    [
        "ffmpeg",
        "-f", "rawvideo",  # Input format as raw video
        "-pix_fmt", "bgr24",  # Pixel format
        "-s", "640x480",  # Frame size, update based on your input video dimensions
        "-r", "30",  # Frame rate
        "-i", "-",  # Input from stdin
        "-c:v", "libx264",  # Video codec
        "-preset", "ultrafast",  # Encoding preset
        '-tune', 'zerolatency',
        "-f", "rtsp",  # Output format RTSP
        "rtsp://localhost:8554/stream"  # Output RTSP stream URL
    ],
    stdin=subprocess.PIPE
)

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # Prepare annotations
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_inference(predictions)
    new_labels = []
    
    # Create the annotated image
    image = video_frame.image.copy()  # Make a copy of the frame for annotation
    
    # Loop through detections to perform OCR on detected objects
    for detection in predictions["predictions"]:
        x, y, w, h, class_name = int(detection['x']),int(detection["y"]),int(detection["width"]), int(detection["height"]), class_list[detection["class_id"]]
        # Crop the detected object from the image for OCR processing
        cropped_image = image[y:y+h, x:x+w]
        # Perform OCR on the cropped image using EasyOCR
        ocr_result = reader.readtext(cropped_image)
        print(ocr_result)
        # Extract the detected text from the OCR result (if any)
        detected_text = " ".join([text for _, text, _ in ocr_result]) if ocr_result else "N/A"
        
        # Add the detected text to the label for this object
        label = f"class : {class_name}\n predicted text : {detected_text}"
        new_labels.append(label)
    
    # Annotate the frame with labels and bounding boxes
    print(len(detections), len(labels))
    annotated_image = label_annotator.annotate(
        scene=image, detections=detections, labels=new_labels
    )
    annotated_image = box_annotator.annotate(annotated_image, detections=detections)
    # Display locally in a window (optional)
    cv2.imshow("Predictions with OCR", annotated_image)
    cv2.waitKey(1)
    
    # Resize image if necessary (ensure it's the same size as the FFmpeg input settings)
    image_resized = cv2.resize(annotated_image, (640, 480))  # Resize to match FFmpeg settings

    # Write the frame to the FFmpeg stdin for streaming
    try:
        ffmpeg_process.stdin.write(image_resized.tobytes())
    except BrokenPipeError:
        print("FFmpeg pipe closed.")

# Initialize inference pipeline
pipeline = InferencePipeline.init(
    model_id="flipkartgrid6.0/1",
    video_reference="rtsp://192.168.29.204:8080/h264_ulaw.sdp",
    on_prediction=my_custom_sink,
    api_key="2w3eGijzYuAAmKQzzlhk", 
)

pipeline.start()
pipeline.join()

# Make sure to gracefully terminate the FFmpeg process
ffmpeg_process.stdin.close()
ffmpeg_process.wait()