from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import supervision as sv
import subprocess
import numpy as np

# Initialize annotators
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoundingBoxAnnotator()

# Start FFmpeg to stream over RTSP
ffmpeg_process = subprocess.Popen(
    [
        "ffmpeg", 
        "-re",  # Read input at native frame rate
        "-f", "rawvideo",  # Input format as raw video
        "-pix_fmt", "bgr24",  # Pixel format
        "-s", "640x480",  # Frame size, update based on your input video dimensions
        "-r", "30",  # Frame rate
        "-i", "-",  # Input from stdin
        "-c:v", "libx264",  # Video codec
        "-preset", "ultrafast",  # Encoding preset
        "-f", "rtsp",  # Output format RTSP
        "rtsp://localhost:8554/stream1"  # Output RTSP stream URL
    ],
    stdin=subprocess.PIPE
)

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # Prepare annotations
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_inference(predictions)
    
    # Create the annotated image
    image = label_annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    image = box_annotator.annotate(image, detections=detections)
    
    # Display locally in a window (optional)
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)
    
    # Resize image if necessary (ensure it's the same size as the FFmpeg input settings)
    image_resized = cv2.resize(image, (640, 480))  # Resize to match FFmpeg settings

    # Write the frame to the FFmpeg stdin for streaming
    try:
        ffmpeg_process.stdin.write(image_resized.tobytes())
    except BrokenPipeError:
        print("FFmpeg pipe closed.")

# Initialize inference pipeline
pipeline = InferencePipeline.init(
    model_id="detection-xt8ag/2",
    video_reference=0,
    on_prediction=my_custom_sink,
    api_key="xlSCYXy7QQXARjVhQJmn", 
)

pipeline.start()
pipeline.join()

# Make sure to gracefully terminate the FFmpeg process
ffmpeg_process.stdin.close()
ffmpeg_process.wait()