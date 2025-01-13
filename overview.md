### Eastern Red Cedar Detection and Segmentation Pipeline

#### Step 1: Dataset Collection
EVO II Pro 6K drone was used to capture high-resolution aerial images containing ERC trees and background elements like other tree species, ensuring consistent flight parameters such as altitude and angle to minimize variability. I collected data across seasons and weather conditions to increase model robustness and generalizability.

Details:

* GPS Metadata: I recorded GPS coordinates for each image during the drone flight, particularly noting proximity to water bodies or dense vegetation. This is significant because ERC trees tend to struggle near water due to competition from more dominant species better adapted to wet environments, which limits their ability to establish and thrive.
* Systematic Organization: The collected images were stored in directories (e.g., /drone_images/), with metadata, such as GPS coordinates, saved in a structured format like CSV for easy integration during modeling.
* Incorporating Variations: I included scenarios such as overlapping trees, shadowed regions, and mixed species in the dataset. These challenging cases will test the model's ability to generalize to real-world conditions and improve its robustness.

```python
import os
from geopy.distance import geodesic

# Organize dataset paths
DATASET_DIR = "/path/to/dataset"
DRONE_IMAGES_DIR = os.path.join(DATASET_DIR, "drone_images")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
GPS_COORDINATES_FILE = os.path.join(DATASET_DIR, "gps_coordinates.csv")

# Example: Adding GPS coordinates to a file
import pandas as pd

def save_gps_data(image_name, lat, lon):
    data = {"image_name": image_name, "latitude": lat, "longitude": lon}
    if not os.path.exists(GPS_COORDINATES_FILE):
        pd.DataFrame([data]).to_csv(GPS_COORDINATES_FILE, index=False)
    else:
        existing_data = pd.read_csv(GPS_COORDINATES_FILE)
        pd.concat([existing_data, pd.DataFrame([data])]).to_csv(GPS_COORDINATES_FILE, index=False)

# Add example GPS data
save_gps_data("image_001.jpg", 39.7392, -104.9903)
```

#### Step 2: Segmentation and Annotation
I manually segmented ERC trees from the images using the Adobe program pen tool to ensure precise annotations. Then the segmentation masks were used to generate bounding boxes for supervised learning tasks. All annotations were saved in widely compatible formats such as Pascal VOC or COCO to ensure seamless integration with the training pipeline.

Details:

* Manual Precision: Instead of using bounding boxes or automated tools for segmentation, manual steps were employed to ensure accurate delineation of tree boundaries from other tree objects and shadows. This approach provided high-quality masks essential for effective model training.
* Annotation Validation: All annotations were validated by overlaying them on the original images to ensure proper alignment and corrected as needed for consistency.
* Organized Structure: The annotation files were systematically organized in directories (e.g., /annotations/) and aligned with image file names to streamline their integration into the training pipeline.

```python
import os
from PIL import Image
import numpy as np
import json

# Paths to images and manually created masks
IMAGE_DIR = "/path/to/images"
MASK_DIR = "/path/to/manual_masks"
ANNOTATIONS_DIR = "/path/to/annotations"

# Ensure annotation directory exists
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

def generate_bounding_boxes(mask):
    """
    Generate bounding boxes from a binary mask.
    Args:
        mask (numpy array): Binary mask where tree areas are 1, background is 0.
    Returns:
        List of bounding boxes as [x_min, y_min, x_max, y_max].
    """
    boxes = []
    labeled, num_objects = ndimage.label(mask)  # Label connected components
    for obj_id in range(1, num_objects + 1):  # Start from 1 (background is 0)
        coords = np.argwhere(labeled == obj_id)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes

def process_segmentation(image_path, mask_path):
    """
    Process an image and its corresponding mask to validate alignment
    and generate bounding box annotations.
    """
    # Load image and mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # Grayscale mask
    mask = np.array(mask)  # Convert to numpy array
    
    # Ensure mask dimensions match image dimensions
    if mask.shape != (image.height, image.width):
        raise ValueError(f"Mask dimensions {mask.shape} do not match image dimensions {image.size}.")

    # Generate bounding boxes from the mask
    bounding_boxes = generate_bounding_boxes(mask)
    
    # Save bounding boxes as annotations in JSON format
    annotation = {
        "image": os.path.basename(image_path),
        "width": image.width,
        "height": image.height,
        "bounding_boxes": bounding_boxes
    }
    return annotation

# Iterate through all images and masks to create annotations
annotations = []
for mask_file in os.listdir(MASK_DIR):
    image_file = mask_file.replace("_mask.png", ".jpg")  # Assuming consistent naming
    image_path = os.path.join(IMAGE_DIR, image_file)
    mask_path = os.path.join(MASK_DIR, mask_file)

    if os.path.exists(image_path):
        annotation = process_segmentation(image_path, mask_path)
        annotations.append(annotation)
    else:
        print(f"Warning: Image file {image_file} not found for mask {mask_file}")

# Save all annotations to a JSON file
annotations_path = os.path.join(ANNOTATIONS_DIR, "annotations.json")
with open(annotations_path, "w") as f:
    json.dump(annotations, f, indent=4)

print(f"Annotations saved to {annotations_path}")
```

#### Step 3: Model Training
##### Step 3-1: Fine-Tuning SAM
The SAM model with annotated dataset described above was used as pre-trained weights (ViT-H) as a starting point, and transfer learning techniques were applied to adapt the model to the specific task of detecting and segmenting eastern red cedar trees.

Details:

* Data Augmentation: The dataset was augmented using transformations such as flipping, rotation, scaling, cropping, and adding Gaussian noise. This helped increase the dataset size and introduce variability, improving the model’s ability to generalize across different environmental conditions and tree appearances.
Seasonal variability in the dataset (e.g., images from March, July, and November) was leveraged to ensure the model's robustness against changes in tree foliage and surrounding environments.

##### Step 3-2: Comparing Fine-Tuned vs. Non-Fine-Tuned Models
I evaluated the performance improvement of the fine-tuned SAM model over the pre-trained version by comparing them on key metrics.

Details:

* Quantitative Metrics: I used metrics like Intersection over Union (IoU), precision, recall, and F1 score to measure segmentation quality.
* Qualitative Assessment: I visualized segmentation outputs by overlaying the predicted masks on test images, allowing for a detailed comparison of alignment with ground truth.
* Inference Time: I compared the inference times of the fine-tuned and non-fine-tuned models to evaluate their computational efficiency for deployment scenarios.

##### Step 3-3: Comparing Models with and without GPS Integration
I integrated GPS data into the model to assess whether it improved segmentation accuracy, especially in ambiguous cases where environmental context played a role.

Details:

* GPS as Auxiliary Input: I incorporated GPS features (e.g., latitude, longitude, proximity to water) into a multi-modal learning framework. These features were combined with image embeddings from SAM during training.
* Performance Comparison: I evaluated the segmentation accuracy of models with and without GPS data, focusing on scenarios where environmental context significantly impacted predictions.


##### Step 3-4: Building a Multi-Model System
I developed a multi-model system to answer user prompts such as, “Is there an eastern red cedar tree (ERC) in this image?”

Details:

* Two-Stage Pipeline: The system first detects regions of interest (ROIs) in the image and then classifies these ROIs with confidence scores.
* Interactive User Interface: I built a user-friendly interface using Gradio, allowing users to query images and receive results like “89% probability of eastern red cedar presence,” along with annotated outputs.
* Customizable Thresholds: I implemented adjustable confidence thresholds to provide flexibility based on the user’s needs.


###### Key Features of the Code:
* Pre-trained Model Loading: The SAM model is loaded with pre-trained weights for transfer learning.
* Custom Segmentation Head: A 1x1 convolutional layer is added to tailor the model for binary segmentation tasks.
* Layer Freezing: SAM’s pre-trained layers are frozen initially, allowing only the segmentation head to be trained.
* Fine-Tuning: Specific layers of SAM are unfrozen gradually for fine-tuning, adapting pre-trained features to the  cedar segmentation task.
* Layer-Specific Learning Rates: Different learning rates are applied to SAM layers and the new segmentation head to balance stability and task-specific learning.

```python
import torch
import torch.nn as nn
from segment_anything import sam_model_registry

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained SAM model (ViT-H variant)
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Path to the SAM checkpoint
model_type = "vit_h"  # Model type: Vision Transformer - H
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

# Define the modified SAM model with a custom binary segmentation head
class SAMFineTuned(nn.Module):
    def __init__(self, sam_model):
        super(SAMFineTuned, self).__init__()
        self.sam = sam_model  # Pre-trained SAM model
        self.segmentation_head = nn.Conv2d(256, 1, kernel_size=1)  # Binary segmentation head

    def forward(self, x):
        # Extract features using SAM's pre-trained image encoder
        features = self.sam.image_encoder(x)
        # Apply the segmentation head to get the output
        output = self.segmentation_head(features)
        # Use sigmoid activation for binary mask output
        return torch.sigmoid(output)

# Initialize the fine-tuned SAM model
model = SAMFineTuned(sam)
model.to(device)

# Freeze all layers of the pre-trained SAM model for transfer learning
for param in model.sam.parameters():
    param.requires_grad = False

# Unfreeze only the segmentation head for initial training
for param in model.segmentation_head.parameters():
    param.requires_grad = True

# After initial training, unfreeze specific layers of the SAM model for fine-tuning
for name, param in model.sam.named_parameters():
    if "image_encoder.layer4" in name:  # Example: Unfreeze specific encoder layers
        param.requires_grad = True

# Define the optimizer with separate learning rates for different parts of the model
optimizer = torch.optim.Adam([
    {"params": model.sam.parameters(), "lr": 1e-5},  # Lower learning rate for SAM layers
    {"params": model.segmentation_head.parameters(), "lr": 1e-4},  # Higher learning rate for the segmentation head
])

# Print a summary of the model layers and their trainability
print("Model summary:")
for name, param in model.named_parameters():
    print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

print("Model ready for transfer learning and fine-tuning.")
```
