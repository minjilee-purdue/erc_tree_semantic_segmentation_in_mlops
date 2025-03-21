# sam_finetuning_utility.py

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import datetime
import json

def load_sam_model(model_type, checkpoint_path, device):
    """
    Load a SAM model with specified backbone
    
    Args:
        model_type: Model type ('vit_b', 'vit_l', or 'vit_h')
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded SAM model
    """
    print(f"Loading {model_type} model from {checkpoint_path}...")
    model_state = torch.load(checkpoint_path, map_location=device)
    sam = sam_model_registry[model_type]()
    sam.load_state_dict(model_state)
    sam = sam.to(device)
    print(f"Model loaded successfully and moved to {device}")
    return sam

def prepare_data_loaders(dataset, batch_size=2, val_split=0.2, num_workers=4, seed=42):
    """
    Prepare training and validation data loaders
    
    Args:
        dataset: The dataset to split
        batch_size: Batch size for loading data
        val_split: Proportion of data to use for validation
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader: Data loaders for training and validation
    """
    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Define data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    
    # Apply transforms to datasets
    class TransformedSubset:
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        
        def __len__(self):
            return len(self.subset)
        
        def __getitem__(self, idx):
            image, mask, bboxes, img_name = self.subset[idx]
            if self.transform:
                image = self.transform(image)
            return image, mask, bboxes, img_name
    
    # Create transformed datasets
    train_dataset_tf = TransformedSubset(train_dataset, train_transform)
    val_dataset_tf = TransformedSubset(val_dataset, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset_tf, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset_tf,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader

def custom_collate_fn(batch):
    """
    Custom collate function for handling variable-length bounding boxes
    
    Args:
        batch: Batch of data items
        
    Returns:
        Processed batch with proper formatting
    """
    images = []
    masks = []
    bboxes_list = []
    img_names = []
    
    for image, mask, bboxes, img_name in batch:
        images.append(image)
        masks.append(mask)
        bboxes_list.append(bboxes)
        img_names.append(img_name)
    
    # Stack images and masks
    images = torch.stack(images)
    masks = torch.stack(masks)
    
    return images, masks, bboxes_list, img_names

def run_model_comparison(dataset, model_configs, train_loader, val_loader, num_epochs=3):
    """
    Run comparison between different SAM models
    
    Args:
        dataset: Dataset to use
        model_configs: List of (model_type, checkpoint_path) tuples
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs for training
    
    Returns:
        Dictionary of results for each model
    """
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_type, checkpoint_path in model_configs:
        # Set up experiment name
        experiment_name = f"sam_{model_type}"
        save_dir = f"checkpoints/{experiment_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Training {model_type} model")
        print(f"{'='*50}")
        
        # Load model
        sam = load_sam_model(model_type, checkpoint_path, device)
        
        # Import here to avoid circular imports
        from march19_sam_trainer import SamFineTuner
        
        # Initialize fine-tuner with same parameters
        fine_tuner = SamFineTuner(
            sam_model=sam,
            device=device,
            learning_rate=1e-5,
            weight_decay=1e-4
        )
        
        # Train model
        start_time = time.time()
        history = fine_tuner.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            save_dir=save_dir
        )
        training_time = time.time() - start_time
        
        # Test model
        test_results_dir = f"results/{experiment_name}"
        test_metrics = test_model_and_evaluate(
            model=sam,
            dataset=dataset,
            device=device,
            num_samples=30,
            save_dir=test_results_dir
        )
        
        # Store results
        results[model_type] = {
            'training_time': training_time,
            'history': history,
            'test_metrics': test_metrics
        }
    
    # Compare results
    compare_models(results)
    return results

def compare_models(results):
    """
    Visualize comparison between models
    
    Args:
        results: Dictionary of model results
    """
    models = list(results.keys())
    
    # Plot metrics comparison
    metrics = ['avg_dice', 'avg_iou', 'avg_precision', 'avg_recall']
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        values = [results[model]['test_metrics'][metric] for model in models]
        plt.bar(models, values)
        plt.title(f'Comparison of {metric}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("model_comparison_metrics.png")
    print("Model metrics comparison saved to model_comparison_metrics.png")
    
    # Plot training time comparison
    plt.figure(figsize=(10, 5))
    times = [results[model]['training_time'] / 3600 for model in models]  # Convert to hours
    plt.bar(models, times)
    plt.title('Training Time Comparison (hours)')
    plt.grid(True, alpha=0.3)
    plt.savefig("model_comparison_time.png")
    print("Training time comparison saved to model_comparison_time.png")
    
    # Plot validation loss curves
    plt.figure(figsize=(10, 5))
    for model in models:
        plt.plot(results[model]['history']['val_loss'], label=model)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("model_comparison_loss.png")
    print("Validation loss comparison saved to model_comparison_loss.png")
    
    # Create a summary table and save it
    with open("model_comparison_summary.txt", "w") as f:
        f.write("Model Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Write metrics
        f.write("Evaluation Metrics:\n")
        f.write("-" * 80 + "\n")
        header = "Model".ljust(10)
        for metric in metrics:
            header += f" | {metric}".ljust(15)
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        
        for model in models:
            row = model.ljust(10)
            for metric in metrics:
                row += f" | {results[model]['test_metrics'][metric]:.4f}".ljust(15)
            f.write(row + "\n")
        
        # Write training time
        f.write("\nTraining Time:\n")
        f.write("-" * 80 + "\n")
        for model in models:
            hours = results[model]['training_time'] / 3600
            minutes = (results[model]['training_time'] % 3600) / 60
            f.write(f"{model}: {int(hours)}h {int(minutes)}m\n")
    
    print("Model comparison summary saved to model_comparison_summary.txt")

def test_model_and_evaluate(model, dataset, device, num_samples=30, save_dir="test_results"):
    """
    Test the model and calculate metrics
    
    Args:
        model: SAM model to test
        dataset: Dataset to use for testing
        device: Device to run inference on
        num_samples: Number of samples to test
        save_dir: Directory to save test results
        
    Returns:
        Dictionary of evaluation metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    predictor = SamPredictor(model)
    model.eval()
    metrics = {
        'dice_scores': [],
        'iou_scores': [],
        'precision': [],
        'recall': [],
        'inference_times': []
    }

    # Test transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    
    # Randomly select samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Process each sample
    with torch.no_grad():
        for idx in indices:
            # Get sample data
            sample = dataset.valid_samples[idx]
            
            # Load paths
            img_path = os.path.join(dataset.image_dir, sample['image'])
            mask_path = os.path.join(dataset.mask_dir, sample['mask'])
            bbox_path = os.path.join(dataset.bbox_dir, sample['bbox'])
            
            # Load image and mask
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = (image.shape[1], image.shape[0])
            transformed_image = transform(image)
            
            # Load target mask
            target_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
            target_mask_resized = cv2.resize(target_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            
            # Get bounding boxes
            bbox_list = dataset._read_bbox(bbox_path)
            
            # Measure inference time
            start_time = time.time()
            
            # Prepare for prediction
            predictor.set_image(transformed_image.permute(1, 2, 0).cpu().numpy())
            
            # Create a combined prediction mask
            combined_pred_mask = np.zeros((1024, 1024), dtype=np.float32)
            
            # Process each bounding box
            for bbox in bbox_list:
                scaled_bbox = dataset.scale_bbox(bbox, original_size, (1024, 1024))
                bbox_tensor = torch.tensor(scaled_bbox, dtype=torch.float).reshape(1, 4).to(device)
                transformed_box = predictor.transform.apply_boxes_torch(bbox_tensor, transformed_image.shape[1:])
                
                # Get mask prediction
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_box,
                    multimask_output=False,
                    return_logits=True
                )
                
                # Convert to numpy and apply sigmoid
                pred_mask_logits = masks.cpu().squeeze()
                pred_mask = torch.sigmoid(pred_mask_logits).numpy()
                
                # Combine predictions
                combined_pred_mask = np.maximum(combined_pred_mask, pred_mask)
            
            # Record inference time
            inference_time = time.time() - start_time
            metrics['inference_times'].append(inference_time)
            
            # Convert prediction to binary mask
            binary_pred_mask = (combined_pred_mask > 0.5).astype(float)
            
            # Calculate metrics
            # Dice score
            dice = (2 * (binary_pred_mask * target_mask_resized).sum()) / (
                binary_pred_mask.sum() + target_mask_resized.sum() + 1e-6
            )
            metrics['dice_scores'].append(dice)
            
            # IoU (Intersection over Union)
            intersection = (binary_pred_mask * target_mask_resized).sum()
            union = binary_pred_mask.sum() + target_mask_resized.sum() - intersection
            iou = intersection / (union + 1e-6)
            metrics['iou_scores'].append(iou)
            
            # Precision and Recall
            true_positives = (binary_pred_mask * target_mask_resized).sum()
            false_positives = binary_pred_mask.sum() - true_positives
            false_negatives = target_mask_resized.sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives + 1e-6)
            recall = true_positives / (true_positives + false_negatives + 1e-6)
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            
            # Calculate areas
            # Calculate areas
            ground_truth_area = np.sum(target_mask_resized)
            predicted_area = np.sum(binary_pred_mask)
            area_diff_percent = ((predicted_area - ground_truth_area) / ground_truth_area) * 100

            # Visualize results
            plt.figure(figsize=(15, 5))

            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(target_mask_resized, cmap='gray')
            plt.title(f"Ground Truth: {ground_truth_area:.0f} pixels")
            plt.axis("off")

            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(binary_pred_mask, cmap='gray')
            plt.title(f"Predicted Mask: {predicted_area:.0f} pixels ({area_diff_percent:.1f}%)")
            plt.axis("off")

            # Add the dice and IoU scores as text at the bottom of the figure
            plt.figtext(0.5, 0.01, f"Dice: {dice:.4f}, IoU: {iou:.4f}", 
                        ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

            # Increase bottom margin to make room for the text
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(save_dir, f"test_result_{idx}.png"))
            plt.close()

    # Calculate average metrics
    avg_metrics = {
        'avg_dice': np.mean(metrics['dice_scores']),
        'avg_iou': np.mean(metrics['iou_scores']),
        'avg_precision': np.mean(metrics['precision']),
        'avg_recall': np.mean(metrics['recall']),
        'avg_inference_time': np.mean(metrics['inference_times'])
    }

    # Save metrics as JSON
    metrics_path = os.path.join(save_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'individual_metrics': {
                'dice_scores': metrics['dice_scores'],
                'iou_scores': metrics['iou_scores'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'inference_times': metrics['inference_times']
            },
            'average_metrics': avg_metrics
        }, f, indent=4)
    
    print(f"Test metrics saved to {metrics_path}")
    
    # Create a summary visualization
    create_metrics_visualization(metrics, save_dir)
    
    return avg_metrics

def create_metrics_visualization(metrics, save_dir):
    """
    Create visualizations of the metrics
    
    Args:
        metrics: Dictionary of metrics
        save_dir: Directory to save visualizations
    """
    plt.figure(figsize=(15, 10))
    
    # Dice scores
    plt.subplot(2, 2, 1)
    plt.hist(metrics['dice_scores'], bins=10, alpha=0.7)
    plt.axvline(np.mean(metrics['dice_scores']), color='r', linestyle='dashed', linewidth=1)
    plt.title(f"Dice Scores (Avg: {np.mean(metrics['dice_scores']):.4f})")
    plt.grid(True, alpha=0.3)
    
    # IoU scores
    plt.subplot(2, 2, 2)
    plt.hist(metrics['iou_scores'], bins=10, alpha=0.7)
    plt.axvline(np.mean(metrics['iou_scores']), color='r', linestyle='dashed', linewidth=1)
    plt.title(f"IoU Scores (Avg: {np.mean(metrics['iou_scores']):.4f})")
    plt.grid(True, alpha=0.3)
    
    # Precision
    plt.subplot(2, 2, 3)
    plt.hist(metrics['precision'], bins=10, alpha=0.7)
    plt.axvline(np.mean(metrics['precision']), color='r', linestyle='dashed', linewidth=1)
    plt.title(f"Precision (Avg: {np.mean(metrics['precision']):.4f})")
    plt.grid(True, alpha=0.3)
    
    # Recall
    plt.subplot(2, 2, 4)
    plt.hist(metrics['recall'], bins=10, alpha=0.7)
    plt.axvline(np.mean(metrics['recall']), color='r', linestyle='dashed', linewidth=1)
    plt.title(f"Recall (Avg: {np.mean(metrics['recall']):.4f})")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_distribution.png"))
    plt.close()
    print(f"Metrics distribution visualization saved to {os.path.join(save_dir, 'metrics_distribution.png')}")

def setup_experiment(experiment_name, model_configs, image_dir, mask_dir, bbox_dir):
    """
    Set up a new experiment run
    
    Args:
        experiment_name: Name of the experiment
        model_configs: List of model configurations
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        bbox_dir: Directory containing bounding boxes
        
    Returns:
        Dictionary with experiment configuration
    """
    # Create experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{experiment_name}_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    results_dir = os.path.join(experiment_dir, "results")
    logs_dir = os.path.join(experiment_dir, "logs")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save experiment configuration
    config = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "model_configs": model_configs,
        "data_paths": {
            "image_dir": image_dir,
            "mask_dir": mask_dir,
            "bbox_dir": bbox_dir
        },
        "directories": {
            "experiment_dir": experiment_dir,
            "checkpoints_dir": checkpoints_dir,
            "results_dir": results_dir,
            "logs_dir": logs_dir
        }
    }
    
    # Save config to JSON
    config_path = os.path.join(experiment_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Experiment '{experiment_name}' set up in directory: {experiment_dir}")
    print(f"Configuration saved to: {config_path}")
    
    return config