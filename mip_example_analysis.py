from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load the image
image_path = "mip_example_image.png"
original_img = Image.open(image_path).convert("RGB")

# Convert to numpy array for OpenCV processing
img_np = np.array(original_img)

# Step 1: Apply edge detection to highlight patterns
edges = cv2.Canny(img_np, threshold1=30, threshold2=100)

# Step 2: Apply high-pass filtering to emphasize fine details (possible patch patterns)
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
high_pass = cv2.filter2D(img_np, -1, kernel)

# Step 3: Increase contrast to make hidden elements more visible
enhancer = ImageEnhance.Contrast(original_img)
high_contrast_img = enhancer.enhance(3.0)

# Display all processed outputs
fig, axs = plt.subplots(2, 2, figsize=(9, 9))

axs[0][0].imshow(original_img)
axs[0][0].set_title("Original Image")
axs[0][0].axis("off")

axs[0][1].imshow(edges, cmap='gray')
axs[0][1].set_title("Edge Detection")
axs[0][1].axis("off")

axs[1][0].imshow(high_pass)
axs[1][0].set_title("High-Pass Filtered")
axs[1][0].axis("off")

axs[1][1].imshow(high_contrast_img)
axs[1][1].set_title("High Contrast")
axs[1][1].axis("off")

plt.tight_layout()
plt.savefig('mip_example_analysis.png', dpi=300)
