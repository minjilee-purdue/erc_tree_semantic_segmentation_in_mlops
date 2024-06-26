import gradio as gr
import numpy as np
import cv2
import numpy as np
import gradio as gr
from typing import Tuple

from segmentation import segment_by_click, set_image, get_coords


def run_gradio_app():

    # Function to run the Gradio application
    with gr.Blocks() as app:
        # Define Gradio interface
        gr.Markdown("# SAM: Eastern Red Cedar Tree | Semantic Segmentation and Bounding Box with User's Click Prompt")
        with gr.Row():
            coord_w = gr.Number(label="Mouse coords w | x-coordinate")  # Input field for x-coordinate
            coord_h = gr.Number(label="Mouse coords h | y-coordinate")  # Input field for y-coordinate

        with gr.Row():  # Create a row layout
            input_img = gr.Image(label="Input image")  # Display input image
            output_img = gr.Image(label="Output image with segmented mask (can be multiple)")  # Display segmented output image
            bounding_box_img = gr.Image(label="Output image with bounding box")  # Display bounding box image
        
        # Define event handlers
        input_img.upload(set_image, [input_img], None)  # Set input image
        input_img.select(get_coords, None, [coord_w, coord_h])  # Get click coordinates
        input_img.select(segment_by_click, [input_img], [output_img, bounding_box_img])  # Perform segmentation on click event

    # Launch the Gradio application
    app.launch(inline=False, share=True)
    # app.launch(inline=False, share=False)