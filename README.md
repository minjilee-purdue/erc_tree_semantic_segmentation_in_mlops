## Semantic Segmentation for Detecting Eastern Red Cedar Trees

- **Overview:**
  The management and control of tree types, growth, and structure in real-world settings remains a challenge due to scale and cost. Often, such approaches are seen as impractical and overly theoretical, particularly for large-scale projects, as the labor and equipment required can be prohibitively expensive. One prominent example of this challenge is the spread of invasive plant species, such as Eastern Red Cedar (ECR) or Juniperus virginiana.

- **Research Objective:**
  This research aims to develop and explore the application of MLOps principles using a machine learning pipeline with SAM (Segment Anything Model) for the semantic segmentation of ERC trees. The pipeline encompasses stages such as data collection, preprocessing, model building, and deployment in production.

- **Approach:**
  To examine the performance of the SAM model, the research also explains multiple approaches, including edge-based detection methods like Sobel filters to understand the complexities present in the image data.


### Workflow Overview
This research represents a comprehensive workflow from data collection to the development of a SAM model for the detection of ERC trees. The initial phase involves collecting a diverse dataset, capturing ERC trees in various conditions such as different seasons, lighting, and backgrounds.

### Directory Structure 
```
seasonaldataset/
│
├───feb/
│   ├───training/
│   │   ├───raw/
│   │   │   ├───tree/
│   │   │   │   ├───image_001.jpg
│   │   │   │   └───image_001_01.jpg  # multiple tree objects within the image
│   │   │   └───background/
│   │   │       └───image_002.jpg
│   │   ├───processed/
│   │   │   ├───annotations_masks/
│   │   │   │   └───image_001.png
│   │   │   ├───annotations_boundingbox/
│   │   │   │   └───image_001.png
│   │   │   └───annotations_boundingbox_coords/
│   │   │       └───image_001.txt
│   ├───testing/
│   │   ├───raw/
│   │   │   ├───tree/
│   │   │   │   └───image_003.jpg
│   │   │   └───background/
│   │   │       └───image_004.jpg
│   │   ├───processed/
│   │   │   ├───annotations_masks/
│   │   │   │   └───mask_003.png
│   │   │   ├───annotations_boundingbox/
│   │   │   │   └───mask_003.png
│   │   │   └───annotations_boundingbox_coords/
│   │   │       └───mask_003.txt
│   ├───evaluation/
│   │   ├───raw/
│   │   │   ├───tree/
│   │   │   │   └───image_005.jpg
│   │   │   └───background/
│   │   │       └───image_006.jpg
│   │   ├───processed/
│   │   │   ├───annotations_masks/
│   │   │   │   └───mask_005.png
│   │   │   ├───annotations_boundingbox/
│   │   │   │   └───mask_005.png
│   │   │   └───annotations_boundingbox_coords/
│   │   │       └───mask_005.txt
│
├───mar/
│   ├───training/
│   │   ├───raw/
│   │   │   ├───tree/
│   │   │   └───background/
│   │   ├───processed/
│   │   │   ├───annotations_masks/
│   │   │   ├───annotations_boundingbox/
│   │   │   └───annotations_boundingbox_coords/
│   ├───testing/
│   │   ├───raw/
│   │   │   ├───tree/
│   │   │   └───background/
│   │   ├───processed/
│   │   │   ├───annotations_masks/
│   │   │   ├───annotations_boundingbox/
│   │   │   └───annotations_boundingbox_coords/
│   ├───evaluation/
│   │   ├───raw/
│   │   │   ├───tree/
│   │   │   └───background/
│   │   ├───processed/
│   │   │   ├───annotations_masks/
│   │   │   ├───annotations_boundingbox/
│   │   │   └───annotations_boundingbox_coords/
│
├───metadata/
│   ├───dataset_statistics.json
│   ├───preprocessing_log.txt
│   └───training_log.txt
│
├───README.md  # Overall description of the dataset and structure
```


### SAM Model

SAM ([Source](https://segment-anything.com/)), developed by Meta AI, is an image segmentation model released in April 2023. It accurately identifies the location of specific objects or every object in an image and is open source under the Apache 2.0 license. SAM utilizes various input prompts, specifying what to segment in an image, allowing for a wide range of segmentation tasks without the need for additional training.

### Semantic Segmentation with SAM

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
```
#### Function to display a mask overlay on an image
```python
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
```
#### Function to display points on an image/to display a bounding box on an image
```python
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
```
#### Load and convert the image from BGR to RGB format then create subplots and display the original image
```python
# Load and display the image
image = cv2.imread('/home/minjilee/erc_tree_semantic_segmentation_in_mlops/tests/src/test_image_01.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create subplots for displaying images and masks in one row
fig, axs = plt.subplots(1, 5, figsize=(25, 10))  # Adjusted figsize to make room for 5 subplots

# Plot the original image with title
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('on')

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

```
#### Load the SAM model and move it to the specified device
```python
sam_checkpoint = "/home/minjilee/erc_tree_semantic_segmentation_in_mlops/weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

predictor.set_image(image)

```
#### Define the coordinates of the input point and its label
```python
input_point = np.array([[1000, 1030]])
input_label = np.array([1])

# Plot with input point and title
axs[1].imshow(image)
show_points(input_point, input_label, axs[1])
axs[1].set_title('Image with Input Point')
axs[1].axis('on')

# Perform prediction
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Plot masks and points with titles
num_masks = len(masks)
for i in range(min(num_masks, 3)):  # Display up to 3 masks
    axs[i+2].imshow(image)
    show_mask(masks[i], axs[i+2])
    show_points(input_point, input_label, axs[i+2])
    axs[i+2].set_title(f'Mask {i+1}, Score: {scores[i]:.3f}')
    axs[i+2].axis('off')

# Adjust layout and display all subplots
plt.tight_layout()
plt.show()
```

![demo_erc_1](/workshop/src/workshop_demo_erc_1.png)
![demo_erc_2](/workshop/src/workshop_demo_erc_2.png)

### Environment Settings
```
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
```

### Operating System and GPU Utilization
- **Operating System:** Pop!_OS ([Source](https://pop.system76.com/))
  - Pop!_OS is a Linux distribution developed by System76, based on Ubuntu and featuring a customized GNOME desktop known as COSMIC.
  - It provides out-of-the-box support for both AMD and Nvidia GPUs.
  - The distribution supports TensorFlow and CUDA without additional configuration.
  - Includes a recovery partition for system refresh.
  - The latest version, Ubuntu 22.04 LTS, was selected for this study.
 
### SAM Demo with HuggingFace/Gradio

![demo_erc_1](/src/assets/web_ui_demo_01.png)
![demo_erc_2](/src/assets/web_ui_demo_02.png)
![demo_erc_3](/src/assets/web_ui_demo_03.png)
![demo_erc_4](/src/assets/web_ui_demo_04.png)

![demo_erc_4](final_demo.gif)

### Development Tools
- **Editor:** Visual Studio Code (VS Code) ([Source](https://code.visualstudio.com/))
  - VS Code was chosen as the editor due to its compatibility with Linux.
 


### Using HuggingFace ([Source](https://huggingface.co/)) and Gradio for Interactive Image Segmentation with SAM

With Hugging Face, it is easy to get access to a vast repository of pre-trained models, including transformers and vision models. This enables us to experiment with various architectures and leverage state-of-the-art techniques for image segmentation tasks. In addition, Gradio offers a user-friendly interface for building interactive applications with machine learning models. Its intuitive design allows users to upload images and interact with the model through simple UI components like sliders and buttons.
