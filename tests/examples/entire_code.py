'''
SAM (Segment Anything Model), developed by Meta, Revised by Minji Lee
Contact: lee3450@purdue.edu or LinkedIn profile https://www.linkedin.com/in/minji-lee-purdue

Copyright (c) 2024 Minji Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import shutil
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry

# Check if CUDA is available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Define paths and names for checkpoint file
CHECKPOINT_PATH = os.path.join("checkpoint")
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "/home/minjilee/Downloads/sam_vit_h_4b8939.pth"

# Create the checkpoint directory if it doesn't exist
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Check if the checkpoint file exists, otherwise copy it from URL
checkpoint = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
if not os.path.exists(checkpoint):
    shutil.copyfile(CHECKPOINT_URL, checkpoint)

# Load the SAM model
sam = sam_model_registry["default"](checkpoint=checkpoint).to(DEVICE)
predictor = SamPredictor(sam)

# Load the image
image = cv2.imread('/home/minjilee/Downloads/sunflower.jpeg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to perform segmentation
def segment(image: np.ndarray, point_x: int, point_y: int) -> np.ndarray:
    points_coords = np.array([[point_x, point_y], [0, 0]])
    points_label = np.array([1, -1])

    # Inference SAM Decoder model with point information
    masks, scores, _ = predictor.predict(points_coords, points_label)

    # Select the best mask based on the score
    mask, _ = select_masks(masks, scores, points_coords.shape[0])
    mask = (mask > 0).astype(np.uint8) * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return mask

# Function to select masks based on scores
def select_masks(masks: np.ndarray, iou_preds: np.ndarray, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    score_reweight = np.array([1000] + [0] * 2)
    score = iou_preds + (num_points - 2.5) * score_reweight
    best_idx = np.argmax(score)
    mask = np.expand_dims(masks[best_idx, :, :], axis=-1)
    iou_pred = np.expand_dims(iou_preds[best_idx], axis=0)
    return mask, iou_pred

# Function to set image for segmentation
def set_image(image: np.ndarray):
    # Preprocess image and get image embedding with SAM Encoder
    predictor.set_image(image)

# Function to get coordinates of click event
def get_coords(evt: gr.SelectData):
    return evt.index[0], evt.index[1]

# Function to handle segmentation on click event
def segment_by_click(image: np.ndarray, evt: gr.SelectData):
    click_w, click_h = evt.index
    print(f"Clicked coordinates: {click_w}, {click_h}")

    # Perform segmentation
    segmented_image = segment(image, click_w, click_h)

    # Debugging: Print information about the segmented image
    print(f"Segmented image shape: {segmented_image.shape}")

    return segmented_image

# Create the Gradio application
with gr.Blocks() as app:
    # Set up UI components
    gr.Markdown("# Example of SAM with 1 click")
    with gr.Row():
        coord_w = gr.Number(label="Mouse coords w")
        coord_h = gr.Number(label="Mouse coords h")

    with gr.Row():
        input_img = gr.Image(label="Input image")
        output_img = gr.Image(label="Output image")
    
    # Set up event handlers
    input_img.upload(set_image, [input_img], None)
    input_img.select(get_coords, None, [coord_w, coord_h])
    input_img.select(segment_by_click, [input_img], output_img)

# Launch the Gradio application
app.launch(inline=False, share=True)
