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

#### Step 3: Segmentation and Annotation
