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

'''
nvidia-smi # Use the 'nvidia-smi' command to check GPU access
Tue May  7 16:29:25 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off |   00000000:01:00.0  On |                  Off |
|  0%   29C    P8              6W /  500W |     678MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2367      G   /usr/lib/xorg/Xorg                            146MiB |
|    0   N/A  N/A      3060      G   /usr/bin/gnome-shell                          154MiB |
|    0   N/A  N/A     23983      G   ...sion,SpareRendererForSitePerProcess        246MiB |
|    0   N/A  N/A     28012      G   /usr/lib/firefox/firefox-bin                  112MiB |
+-----------------------------------------------------------------------------------------+

'''

import cv2
import numpy as np
import gradio as gr
from typing import Tuple
from segment_anything import SamPredictor, sam_model_registry

from app import run_gradio_app
from segmentation import segment, set_image, get_coords

'''
import os
import shutil
import torch
'''

'''
def segment_by_click(image: np.ndarray, evt: gr.SelectData):
    # Function to perform segmentation when a click event occurs
    click_w, click_h = evt.index
    print(f"Clicked coordinates: {click_w}, {click_h}")

    # Perform segmentation
    segmented_image = segment(image, click_w, click_h)

    # Debugging: Print information about the segmented image
    print(f"Segmented image shape: {segmented_image.shape}")

    return segmented_image
'''

if __name__ == "__main__":
    run_gradio_app()  # Run the Gradio application when the script is executed directly