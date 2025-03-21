# march19_dataset.py (with logging)

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger("CedarTreeDataset")

class CedarTreeDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, bbox_dir: str, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bbox_dir = bbox_dir
        self.transform = transform

        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.bbox_files = sorted(os.listdir(bbox_dir))

        logger.info(f"Found {len(self.image_files)} images")
        logger.info(f"Found {len(self.mask_files)} masks")
        logger.info(f"Found {len(self.bbox_files)} bbox files")

        # Match files by filename
        self.valid_samples = self._match_files()
        logger.info(f"Found {len(self.valid_samples)} valid matching samples")

        # Visualize first sample (optional)
        if self.valid_samples:
            self._visualize_sample(self.valid_samples[0])

    def _match_files(self) -> List[Dict[str, str]]:
        matched = []
        for img_file in self.image_files:
            name, _ = os.path.splitext(img_file)
            mask_file = name + ".png"
            bbox_file = name + "_bboxes.txt"
            if mask_file in self.mask_files and bbox_file in self.bbox_files:
                matched.append({
                    "image": img_file,
                    "mask": mask_file,
                    "bbox": bbox_file
                })
        return matched

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx: int):
        sample = self.valid_samples[idx]
        img_path = os.path.join(self.image_dir, sample["image"])
        mask_path = os.path.join(self.mask_dir, sample["mask"])
        bbox_path = os.path.join(self.bbox_dir, sample["bbox"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  # Normalize to [0, 1]

        # Resize to match SAM input if needed (done in transforms otherwise)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        bboxes = self._read_bbox(bbox_path)

        if self.transform:
            image = self.transform(image)

        return image, mask, bboxes, sample['image']

    def _read_bbox(self, bbox_path: str) -> List[List[int]]:
        bboxes = []
        with open(bbox_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    bboxes.append([int(p) for p in parts])
        return bboxes

    def scale_bbox(self, bbox: List[int], original_size: Tuple[int, int], target_size: Tuple[int, int]) -> List[int]:
        ow, oh = original_size
        tw, th = target_size
        scale_x = tw / ow
        scale_y = th / oh
        x1, y1, x2, y2 = bbox
        return [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]

    def _visualize_sample(self, sample: Dict[str, str]):
        import matplotlib.pyplot as plt
        logger.info("\nVisualizing first sample:")
        logger.info(f"  Image: {sample['image']}")
        logger.info(f"  Mask: {sample['mask']}")
        logger.info(f"  Bbox: {sample['bbox']}")

        img_path = os.path.join(self.image_dir, sample["image"])
        mask_path = os.path.join(self.mask_dir, sample["mask"])
        bbox_path = os.path.join(self.bbox_dir, sample["bbox"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        bboxes = self._read_bbox(bbox_path)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Image with Bboxes")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")

        plt.tight_layout()
        plt.savefig("first_sample_visualization.png")
        plt.close()
        logger.info("  Visualization saved to first_sample_visualization.png")


    def __len__(self):
        return len(self.valid_samples)

    def _read_bbox(self, bbox_path):
        """Read all bounding boxes from file"""
        try:
            # Read all bounding boxes from file
            all_bboxes = []
            with open(bbox_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Parse the line
                    if ': ' in line:
                        parts = line.split(': ')
                        layer_name = parts[0]
                        coords_str = parts[1].strip()
                        
                        # Extract coordinates
                        coords = [int(c) for c in coords_str.split(', ')]
                        
                        # Store layer info and coordinates
                        if len(coords) == 4:
                            # Skip Layer 1 (background)
                            if "Layer 1" not in layer_name:
                                all_bboxes.append(np.array(coords))
            
            return all_bboxes
                
        except Exception as e:
            print(f"Error reading bbox file {bbox_path}: {str(e)}")
            return [np.array([0, 0, 100, 100])]  # Default fallback

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample = self.valid_samples[idx]
        
        # Construct full paths
        img_path = os.path.join(self.image_dir, sample['image'])
        mask_path = os.path.join(self.mask_dir, sample['mask'])
        bbox_path = os.path.join(self.bbox_dir, sample['bbox'])

        # Load image
        image = cv2.imread(img_path)
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read all bounding boxes and scale them
        all_bboxes = self._read_bbox(bbox_path)  # Read all original bounding boxes
        scaled_bboxes = [self.scale_bbox(bbox, original_size, (1024, 1024)) for bbox in all_bboxes]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        mask = torch.tensor(mask, dtype=torch.float32)
        return image, mask, scaled_bboxes, sample['image']