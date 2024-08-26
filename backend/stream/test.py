import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Load the image
image = cv2.imread('/home/annone/ai/data/eee.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Detect edges using Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# Step 4: Create a mask from the edges
mask = np.zeros_like(gray)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# Step 5: Extract colors inside the edges using the mask
masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Step 6: Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Colors inside edges")
plt.imshow(masked_image)
plt.axis('off')

plt.show()
