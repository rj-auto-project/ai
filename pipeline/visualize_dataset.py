import os
import random
import matplotlib.pyplot as plt
import cv2

# Constants

TRAIN_FOLDER = '/home/annone/ai/pipeline/train/image'
LABEL_FOLDER = '/home/annone/ai/pipeline/train/labels'
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']  # Adjust based on your dataset format
IMG_WIDTH,IMG_HEIGHT = 640 , 480

def load_yolo_labels(label_file, img_width, img_height):
    boxes = []
    print("pppp")
    with open(label_file, 'r') as file:
        for line in file.readlines():
            label = line.strip().split()
            class_id = int(label[0])
            # YOLO format: class_id, x_center, y_center, width, height (all normalized 0-1)
            x_center, y_center, width, height = map(float, label[1:])
            
            # Convert normalized coordinates to pixel values
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            print(x_center)
            # Calculate the top-left and bottom-right corners
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            boxes.append([class_id, x_min, y_min, x_max, y_max])
    return boxes

# Helper function to plot bounding boxes on an image
def plot_bounding_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        class_id, x_min, y_min, x_max, y_max = box
        cv2.putText(img,f"{class_id}",(x_min+10,y_min+10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img

# Select 10 random images from the train folder
def select_random_images(folder, num_images=10):
    all_files = [f for f in os.listdir(folder) if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS]
    return random.sample(all_files, min(num_images, len(all_files)))

# Visualize selected images in a collage with bounding boxes
def visualize_collage(images_folder, labels_folder, num_images=10):
    selected_images = select_random_images(images_folder, num_images)
    
    # Setup Matplotlib figure for the collage (2x5 grid)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, img_file in enumerate(selected_images):
        # Load image
        img_path = os.path.join(images_folder, img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib
        
        # Load corresponding YOLO labels
        label_file = os.path.join(labels_folder, os.path.splitext(img_file)[0] + '.txt')
        print(label_file)
        if os.path.exists(label_file):
            print("-----------------------------------------")
            boxes = load_yolo_labels(label_file, IMG_WIDTH, IMG_HEIGHT)
            img_rgb = plot_bounding_boxes(img_rgb, boxes)
        
        # Display image in the subplot
        axes[i].imshow(img_rgb)
        # axes[i].set_title(img_file)
        axes[i].axis('off')  # Hide axis

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Visualize 10 random images from the training folder
    visualize_collage(TRAIN_FOLDER, LABEL_FOLDER, num_images=10)
