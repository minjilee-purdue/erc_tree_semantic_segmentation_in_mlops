import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Add this code at the beginning of your script to help with debugging
import sys

# Print Python version and directory information
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")



# Define dataset and dataloader
image_dir = "/home/minjilee/Desktop/dataset/images"
mask_dir = "/home/minjilee/Desktop/dataset/masks_grey"
bbox_dir = "/home/minjilee/Desktop/dataset/bbox_txt"


# Add this to check if all paths exist
print(f"Image directory exists: {os.path.exists(image_dir)}")
print(f"Mask directory exists: {os.path.exists(mask_dir)}")
print(f"Bbox directory exists: {os.path.exists(bbox_dir)}")

# Check sample file listings
print("\nSample files in directories:")
if os.path.exists(image_dir):
    print(f"Images: {os.listdir(image_dir)[:3]}")
if os.path.exists(mask_dir):
    print(f"Masks: {os.listdir(mask_dir)[:3]}")
if os.path.exists(bbox_dir):
    print(f"Bboxes: {os.listdir(bbox_dir)[:3]}")



class CedarTreeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, bbox_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bbox_dir = bbox_dir
        self.transform = transform
        
        # Get sorted lists of filenames
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.bbox_filenames = sorted(os.listdir(bbox_dir))
        
        print(f"\nFound {len(self.image_filenames)} images")
        print(f"Found {len(self.mask_filenames)} masks")
        print(f"Found {len(self.bbox_filenames)} bbox files")
        
        # Map image files to corresponding mask and bbox files
        self.valid_samples = []
        
        # Create a mapping from base names to bbox filenames
        bbox_mapping = {}
        for bbox_file in self.bbox_filenames:
            # Remove file extension
            bbox_base_with_suffix = os.path.splitext(bbox_file)[0]
            
            # Handle the "_bboxes" suffix
            if bbox_base_with_suffix.endswith("_bboxes"):
                # Extract the base name without "_bboxes"
                bbox_base = bbox_base_with_suffix[:-7]  # Remove "_bboxes"
                bbox_mapping[bbox_base] = bbox_file
            else:
                # If no suffix, use as is
                bbox_base = bbox_base_with_suffix
                bbox_mapping[bbox_base] = bbox_file
        
        # Now match image files with masks and bboxes
        for img_file in self.image_filenames:
            img_base = os.path.splitext(img_file)[0]
            
            # Find matching mask file
            mask_file = None
            for m_file in self.mask_filenames:
                if os.path.splitext(m_file)[0] == img_base:
                    mask_file = m_file
                    break
            
            # Find matching bbox file using our mapping
            bbox_file = bbox_mapping.get(img_base)
            
            # If we found all files, add to valid samples
            if mask_file and bbox_file:
                self.valid_samples.append({
                    'image': img_file,
                    'mask': mask_file,
                    'bbox': bbox_file
                })
                
        print(f"Found {len(self.valid_samples)} valid matching samples")
        
        if len(self.valid_samples) == 0:
            # Instead of raising an error, print detailed information to help debug
            print("\nWARNING: No valid samples found! Here's some debug information:")
            print("\nFirst few image files:")
            for i in self.image_filenames[:5]:
                print(f"  {i} -> base: {os.path.splitext(i)[0]}")
                
            print("\nFirst few mask files:")
            for m in self.mask_filenames[:5]:
                print(f"  {m} -> base: {os.path.splitext(m)[0]}")
                
            print("\nFirst few bbox files:")
            for b in self.bbox_filenames[:5]:
                print(f"  {b} -> base: {os.path.splitext(b)[0]}")
                
            print("\nBounding box mapping:")
            for key, value in list(bbox_mapping.items())[:5]:
                print(f"  {key} -> {value}")
                
            raise ValueError("No valid samples found. Check the debug information above.")
            
        # Visualize first sample
        self._visualize_first_sample()



    def scale_bbox(self, bbox, original_size, target_size):
        """Scale bounding box coordinates from original image size to target size"""
        x_scale = target_size[0] / original_size[0]
        y_scale = target_size[1] / original_size[1]
        
        scaled_bbox = np.array([
            int(bbox[0] * x_scale),
            int(bbox[1] * y_scale),
            int(bbox[2] * x_scale),
            int(bbox[3] * y_scale)
        ])
        
        # Ensure coordinates are within valid range
        scaled_bbox[0] = max(0, min(scaled_bbox[0], target_size[0] - 1))
        scaled_bbox[1] = max(0, min(scaled_bbox[1], target_size[1] - 1))
        scaled_bbox[2] = max(0, min(scaled_bbox[2], target_size[0] - 1))
        scaled_bbox[3] = max(0, min(scaled_bbox[3], target_size[1] - 1))
        
        return scaled_bbox

    def _visualize_first_sample(self):
        """Visualize the first sample for debugging purposes"""
        if not self.valid_samples:
            return
            
        sample = self.valid_samples[0]
        img_path = os.path.join(self.image_dir, sample['image'])
        mask_path = os.path.join(self.mask_dir, sample['mask'])
        bbox_path = os.path.join(self.bbox_dir, sample['bbox'])
        
        print(f"\nVisualizing first sample:")
        print(f"  Image: {sample['image']}")
        print(f"  Mask: {sample['mask']}")
        print(f"  Bbox: {sample['bbox']}")
        
        # Load image and mask
        image = cv2.imread(img_path)
        if image is None:
            print(f"ERROR: Could not read image file {img_path}")
            return
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"ERROR: Could not read mask file {mask_path}")
            return
            
        mask = mask / 255.0  # Normalize
        
        # Read bbox
        try:
            bbox = self._read_bbox(bbox_path)
            print(f"  Bounding box: {bbox}")
        except Exception as e:
            print(f"ERROR reading bbox: {str(e)}")
        
        # Visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")
        plt.axis("off")
        
        # Visualize bbox on image
        plt.subplot(1, 3, 3)
        img_with_bbox = image.copy()
        x_min, y_min, x_max, y_max = bbox.astype(int)
        cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        plt.imshow(img_with_bbox)
        plt.title("Image with Bounding Box")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig("first_sample_visualization.png")
        plt.close()
        print("  Visualization saved to first_sample_visualization.png")

    def __len__(self):
        return len(self.valid_samples)

    def _read_bbox(self, bbox_path):
        """Read bounding box from file"""
        try:
            # Read all bounding boxes from file
            bboxes = []
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
                            bboxes.append((layer_name, coords))
            
            # Find Layer 2 (cedar tree)
            for layer_name, coords in bboxes:
                if "Layer 2" in layer_name:
                    return np.array(coords)
            
            # If Layer 2 isn't found, return the first bbox that's not Layer 1
            for layer_name, coords in bboxes:
                if "Layer 1" not in layer_name:
                    return np.array(coords)
                    
            # If none of the above worked, return the first bbox
            if bboxes:
                _, coords = bboxes[0]
                return np.array(coords)
                
            # Last resort fallback
            return np.array([0, 0, 100, 100])
            
        except Exception as e:
            print(f"Error reading bbox file {bbox_path}: {str(e)}")
            return np.array([0, 0, 100, 100])  # Default fallback

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

        # Read bounding box and scale it
        bbox = self._read_bbox(bbox_path)  # Read the original bounding box
        scaled_bbox = self.scale_bbox(bbox, original_size, (1024, 1024))  # Scale the bounding box

        print(f"Original BBox: {bbox}, Scaled BBox: {scaled_bbox}")



        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        mask = torch.tensor(mask, dtype=torch.float32)
        return image, mask, scaled_bbox, sample['image']
    
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1024, 1024)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# Create the full dataset
full_dataset = CedarTreeDataset(image_dir, mask_dir, bbox_dir, transform=None)  # No transform yet

# Split into train and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                          generator=torch.Generator().manual_seed(42))

# Apply different transformations to train and validation datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Load pre-trained SAM model
sam_checkpoint = "/home/minjilee/Desktop/edit/weights/sam_vit_h_4b8939.pth" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_state = torch.load(sam_checkpoint, map_location=device)
sam = sam_model_registry["vit_h"]()
sam.load_state_dict(model_state)
sam = sam.to(device)

# Improved adaptation layer with batch normalization
class ImprovedSAMAdaptationLayer(nn.Module):
    def __init__(self, input_channels=1):
        super(ImprovedSAMAdaptationLayer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.conv_layers(x)

# Initialize components
adaptation_layer = ImprovedSAMAdaptationLayer().to(device)
predictor = SamPredictor(sam)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(adaptation_layer.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Function to calculate IoU
def calculate_iou(pred, target):
    pred_binary = (pred > 0).float()  # Convert logits to binary mask
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)  # Add small epsilon to avoid division by zero
    print(f"Predicted Mask Sum: {pred.sum().item()}, Target Mask Sum: {target.sum().item()}")

    return iou.item()

# Function to visualize results
def visualize_results(image, true_mask, pred_mask, filename, epoch, save_dir="visualization"):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    image = image.permute(1, 2, 0).cpu().numpy()
    true_mask = true_mask.cpu().numpy()
    pred_mask = torch.sigmoid(pred_mask).squeeze(0).cpu().numpy()  # Apply sigmoid, remove the extra dimension, and convert to numpy
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot image
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")
    
    # Plot true mask
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title("True Mask")
    axes[1].axis("off")
    
    # Plot predicted mask
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")
    
    # Save the figure
    filepath = os.path.join(save_dir, f"epoch_{epoch}_{filename}.png")
    plt.savefig(filepath)
    plt.close(fig)
    print(f"  Saved visualization to {filepath}")

# Training loop
def train_model(adaptation_layer, train_dataloader, val_dataloader, optimizer, criterion, scheduler, num_epochs=15, device="cuda"):
    best_val_loss = float('inf')
    
    print(f"Training on device: {device}")
    print(f"Total images: {len(full_dataset)}")
    print(f"Training images: {len(train_dataset)}, Validation images: {len(val_dataset)}")
    
    for epoch in range(num_epochs):
        adaptation_layer.train()
        train_loss = 0.0
        train_iou = 0.0
        
        loop = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=True)
        for i, (images, masks, bboxes, filenames) in enumerate(loop):
            images = images.to(device)
            masks = masks.to(device)
            
            bboxes = bboxes.to(device)

            optimizer.zero_grad()

            # Prepare prompts and predict masks
            predicted_masks = []
            for image, bbox in zip(images, bboxes):
                # Reshape the bounding box to (1, 4) and convert to float
                bbox = bbox.float().reshape(1, 4)
                
                # Set image for the predictor
                predictor.set_image(image.permute(1, 2, 0).cpu().numpy())
                
                transformed_box = predictor.transform.apply_boxes_torch(bbox, image.shape[1:])
                # Validate and clip coordinates
                transformed_box = torch.clamp(transformed_box, 0, 1024)



                # ********************************************************************
                # ***  ESSENTIAL: VALIDATE BOUNDING BOX COORDINATES HERE  ***
                # ********************************************************************
                print(f"Transformed Bounding Box: {transformed_box}")  # PRINT THIS!
                if torch.any(transformed_box < 0) or torch.any(transformed_box > 1024):
                    print("WARNING: Bounding box coordinates are out of range!")
                # ********************************************************************

                # In train_model (before predictor.predict_torch)
                print(f"Scaled BBox (before transformation): {bbox}")
                print(f"Transformed BBox (after apply_boxes_torch): {transformed_box}")
                
                # Predict masks with logits=True
                mask, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_box,
                    multimask_output=False,
                    return_logits=True,
                )
                predicted_masks.append(mask)
            
            # Stack the predicted masks
            predicted_masks = torch.stack(predicted_masks).squeeze(1).to(device)
            
            # Pass the predicted masks through the adaptation layer
            adapted_masks = adaptation_layer(predicted_masks)

            # Calculate loss
            loss = criterion(adapted_masks, masks.unsqueeze(1))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            train_iou += calculate_iou(adapted_masks.sigmoid(), masks.unsqueeze(1))

            loop.set_postfix({
                'loss': loss.item(),
                'iou': calculate_iou(adapted_masks.sigmoid(), masks.unsqueeze(1))
            })

        # Evaluate on validation set
        adaptation_layer.eval()
        val_loss = 0.0
        val_iou = 0.0

        loop = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", leave=True)
        with torch.no_grad():
            for i, (images, masks, bboxes, filenames) in enumerate(loop):
                images = images.to(device)
                masks = masks.to(device)
                bboxes = bboxes.to(device)
                
                # Prepare prompts and predict masks
                predicted_masks = []
                for image, bbox in zip(images, bboxes):
                    # Reshape the bounding box to (1, 4) and convert to float
                    bbox = bbox.float().reshape(1, 4)
                    
                    # Set image for the predictor
                    predictor.set_image(image.permute(1, 2, 0).cpu().numpy())
                    
                    # Transform the bounding box
                    #transformed_box = predictor.transform.apply_boxes_torch(bbox, image.shape[1:])
                    transformed_box = predictor.transform.apply_boxes_torch(bbox.float().reshape(1, 4), image.shape[1:])

                    # Predict masks with logits=True
                    mask, _, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_box,
                        multimask_output=False,
                        return_logits=True,
                    )
                    predicted_masks.append(mask)
                
                # Stack the predicted masks
                predicted_masks = torch.stack(predicted_masks).squeeze(1).to(device)

                # Pass the predicted masks through the adaptation layer
                adapted_masks = adaptation_layer(predicted_masks)

                # Calculate loss
                loss = criterion(adapted_masks, masks.unsqueeze(1))

                # Update metrics
                val_loss += loss.item()
                val_iou += calculate_iou(adapted_masks.sigmoid(), masks.unsqueeze(1))

                loop.set_postfix({
                    'val_loss': loss.item(),
                    'val_iou': calculate_iou(adapted_masks.sigmoid(), masks.unsqueeze(1))
                })
                
                # Visualize the first batch of validation results
                if i == 0:
                    for j in range(len(images)):
                        visualize_results(images[j], masks[j], adapted_masks[j], filenames[j], epoch)

        # Calculate average loss and IoU
        train_loss /= len(train_dataloader)
        train_iou /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        val_iou /= len(val_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Training: Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
        print(f"  Validation: Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(adaptation_layer.state_dict(), "best_adaptation_layer.pth")
            print("  Saved best model")

# Run training
train_model(adaptation_layer, train_dataloader, val_dataloader, optimizer, criterion, scheduler, num_epochs=2, device=device)
