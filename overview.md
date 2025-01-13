### Eastern Red Cedar Detection and Segmentation Pipeline

#### Step 1: Dataset Collection
EVO II Pro 6K drone was used to capture high-resolution aerial images containing ERC trees and background elements like other tree species, ensuring consistent flight parameters such as altitude and angle to minimize variability. I collected data across seasons and weather conditions to increase model robustness and generalizability.

Details:

* GPS Metadata: I recorded GPS coordinates for each image during the drone flight, particularly noting proximity to water bodies or dense vegetation. This is significant because eastern red cedar trees tend to struggle near water due to competition from more dominant species better adapted to wet environments, which limits their ability to establish and thrive.
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
I segmented ERC trees from the images using the Photoshop pen tool to ensure precise annotations. The segmentation masks were used to generate bounding boxes for supervised learning tasks. All annotations were saved in widely compatible formats such as Pascal VOC or COCO to ensure seamless integration with the training pipeline.

Details:

* Efficient Annotation Workflow: I used the Segment Anything Model (SAM) as a preliminary tool to generate rough segmentation masks. These masks were then refined manually using the Photoshop pen tool to ensure precision, saving significant time during annotation.
* Annotation Validation: I validated all annotations by overlaying them on the original images and correcting any misalignments to maintain accuracy.
* Organized Structure: The annotation files were organized systematically in directories (e.g., /annotations/) and aligned with image file names to facilitate smooth loading during model training.
