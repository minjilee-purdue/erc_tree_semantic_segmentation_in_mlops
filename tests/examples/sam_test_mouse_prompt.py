import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

# Set matplotlib backend (choose the appropriate backend for your system)
import matplotlib
matplotlib.use('TkAgg')

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.random.rand(3)
    else:
        color = np.array([30/255, 144/255, 255/255])  # Ensure color values are in [0, 1]
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor='none', lw=2))

# Load and display the image
image = cv2.imread('/home/minjilee/erc_tree_semantic_segmentation_in_mlops/tests/src/test_image_02.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create the figure and axis for image display
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
ax.set_title("Click on the image to select points. Kill the image by clicking X button when finished.")
ax.set_axis_on()  # Ensure the axis is on for clarity

# Function to handle mouse clicks for point selection
def on_click(event):
    if event.button == 1:  # Left mouse button clicked
        x, y = int(event.xdata), int(event.ydata)
        ax.scatter(x, y, color='yellow', marker='o', s=100, edgecolor='black', linewidth=1.5)
        plt.draw()

# Connect the mouse click event to the figure
fig.canvas.mpl_connect('button_press_event', on_click)

# Display the figure
plt.show()

# After points are selected (close the figure or continue with another action)

# Assuming the following parts are correctly set up as in your original code
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/home/minjilee/erc_tree_semantic_segmentation_in_mlops/weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if CUDA is available

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

predictor.set_image(image)

# Get clicked points from the scatter plot
clicked_points = np.asarray(ax.collections[0].get_offsets())

# Display selected points on the image
if len(clicked_points) > 0:
    # Perform prediction with the selected points
    input_labels = np.ones(len(clicked_points), dtype=int)  # Assuming all points are positive for simplicity

    masks, scores, logits = predictor.predict(
        point_coords=clicked_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    # Create a figure with subplots for each mask
    num_masks = len(masks)
    fig, axs = plt.subplots(1, num_masks + 1, figsize=(20, 5))  # +1 for the original image

    # Plot the original image with selected points
    axs[0].imshow(image)
    axs[0].scatter(clicked_points[:, 0], clicked_points[:, 1], color='yellow', marker='o', s=100, edgecolor='black', linewidth=1.5)
    axs[0].set_title("Original Image with Selected Points")
    axs[0].set_axis_on()

    # Display masks and points
    for i, (mask, score) in enumerate(zip(masks, scores), start=1):
        axs[i].imshow(image)
        show_mask(mask, axs[i])
        show_points(clicked_points, input_labels, axs[i])
        axs[i].set_title(f"Mask {i}, Score: {score:.3f}")
        axs[i].set_axis_off()  # Turn off axis for clean visualization

    plt.tight_layout()
    plt.show()
else:
    print("No points selected.")