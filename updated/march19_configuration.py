# configuration.py

"""
Configuration file for SAM fine-tuning experiments
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data paths
DATA_CONFIG = {
    "cedar_trees": {
        "image_dir": "/home/minjilee/Desktop/dataset/images",
        "mask_dir": "/home/minjilee/Desktop/dataset/masks_grey",
        "bbox_dir": "/home/minjilee/Desktop/dataset/bbox_txt",
        "description": "Cedar Tree Segmentation Dataset"
    }
}

# Model configurations
MODEL_CONFIGS = {
    "vit_b": {
        "checkpoint_path": "/home/minjilee/Downloads/sam_vit_b_01ec64.pth",
        "description": "SAM ViT-B Model"
    },
    "vit_l": {
        "checkpoint_path": "/home/minjilee/Downloads/sam_vit_l_0b3195.pth",
        "description": "SAM ViT-L Model"
    },
    "vit_h": {
        "checkpoint_path": "/home/minjilee/Desktop/edit/weights/sam_vit_h_4b8939.pth",
        "description": "SAM ViT-H Model"
    }
}

# Training configurations
TRAINING_CONFIGS = {
    "default": {
        "batch_size": 2,
        "num_epochs": 20,
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "val_split": 0.2,
        "num_workers": 4,
        "seed": 42
    },
    "long_training": {
        "batch_size": 2,
        "num_epochs": 30,
        "learning_rate": 5e-6,
        "weight_decay": 1e-4,
        "val_split": 0.2,
        "num_workers": 4,
        "seed": 42
    },
    "quick_test": {
        "batch_size": 4,
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "val_split": 0.1,
        "num_workers": 4,
        "seed": 42
    }
}

# Augmentation configurations
AUGMENTATION_CONFIGS = {
    "minimal": {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotate": False,
        "scale": False,
        "color_jitter": False
    },
    "standard": {
        "horizontal_flip": True,
        "vertical_flip": True,
        "rotate": True,
        "rotate_degree": 20,
        "scale": True,
        "scale_range": (0.8, 1.2),
        "color_jitter": True,
        "color_jitter_strength": 0.2
    },
    "aggressive": {
        "horizontal_flip": True,
        "vertical_flip": True,
        "rotate": True,
        "rotate_degree": 45,
        "scale": True,
        "scale_range": (0.7, 1.3),
        "color_jitter": True,
        "color_jitter_strength": 0.4,
        "random_crop": True,
        "random_crop_size": (512, 512)
    }
}

# Experiment configurations
EXPERIMENTS = {
    "exp1_baseline": {
        "model": "vit_b",
        "dataset": "cedar_trees",
        "training": "default",
        "augmentation": "minimal",
        "description": "Baseline experiment with minimal augmentation"
    },
    "exp2_long_train": {
        "model": "vit_b",
        "dataset": "cedar_trees",
        "training": "long_training",
        "augmentation": "standard",
        "description": "Extended training with standard augmentation"
    },
    "exp3_large_model": {
        "model": "vit_l",
        "dataset": "cedar_trees",
        "training": "default",
        "augmentation": "standard",
        "description": "Using ViT-L model with standard augmentation"
    },
    "exp4_huge_model": {
        "model": "vit_h",
        "dataset": "cedar_trees",
        "training": "default",
        "augmentation": "aggressive",
        "description": "Using ViT-H model with aggressive augmentation"
    }
}

# Evaluation metrics
EVALUATION_METRICS = [
    "mean_iou",
    "boundary_f1",
    "dice_coefficient",
    "pixel_accuracy",
    "precision",
    "recall"
]

# Logging configuration
LOGGING_CONFIG = {
    "tensorboard": True,
    "checkpoint_interval": 5,
    "log_interval": 50,
    "log_dir": os.path.join(OUTPUT_DIR, "logs"),
    "tensorboard_dir": os.path.join(OUTPUT_DIR, "tensorboard")
}

# Inference configurations
INFERENCE_CONFIG = {
    "box_nms_thresh": 0.7,
    "crop_n_layers": 0,
    "crop_nms_thresh": 0.7,
    "crop_overlap_ratio": 0.3,
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95
}