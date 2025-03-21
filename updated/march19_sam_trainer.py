# sam_trainer.py

import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as npsam_fine
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Import from improved files
from march19_loss import CombinedLoss, DiceLoss, FocalLoss
from march19_evaluation_matrics import calculate_metrics


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("SAM-Trainer")


class MetricTracker:
    """Tracks and computes average metrics during training/validation."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = {}
        self.counts = {}
        
    def update(self, metrics_dict: Dict[str, float], count: int = 1):
        """Update metrics with new values."""
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = 0
                self.counts[k] = 0
            self.metrics[k] += v * count
            self.counts[k] += count
            
    def avg(self) -> Dict[str, float]:
        """Get average metrics."""
        return {k: self.metrics[k] / max(self.counts[k], 1) for k in self.metrics}


class SamFineTuner:
    def __init__(
        self,
        sam_model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        config: Dict[str, Any] = None
    ):

        """
        Initialize the SAM fine-tuning class with improved configuration.
        
        Args:
            sam_model: The SAM model to fine-tune
            device: The device to use for training ('cuda' or 'cpu')
            config: Configuration dictionary with hyperparameters
        """
        self.sam = sam_model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Default config with sensible defaults
        default_config = {
            'learning_rate': 1e-5,
            'weight_decay': 1e-4,
            'mask_size': 256,  # Size to resize masks for training
            'dice_weight':0.3,        # Keep good overall structure
            'focal_weight':0.2,       # Handle class imbalance
            'boundary_weight':0.35,   # Strong focus on boundaries
            'hausdorff_weight':0.1,   # Minimize boundary distance
            'tversky_weight':0.05,    # Handle false negatives more strictly
            'tversky_alpha':0.7,      # Higher penalty for false negatives
            'tversky_beta':0.3,        # Lower penalty for false positives
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'grad_accumulation_steps': 1,
            'use_amp': True,  # Use automatic mixed precision
            'scheduler': 'plateau',  # 'plateau' or 'cosine'
            'patience': 5,  # For plateau scheduler
            'min_lr': 1e-7,
            'cosine_epochs': 30,  # For cosine scheduler
        }
        
        # Override defaults with provided config
        self.config = default_config
        if config:
            self.config.update(config)
            
        # Extract configs for easier access
        self.mask_size = self.config['mask_size']
        
        # Set up predictor for validation
        try:
            from segment_anything import SamPredictor
            self.predictor = SamPredictor(self.sam)
        except ImportError:
            logger.warning("SamPredictor not available; some validation functions may not work")
            self.predictor = None
        
        # Define loss functions
        self.combined_loss = CombinedLoss(
        dice_weight=self.config['dice_weight'],
        focal_weight=self.config['focal_weight'],
        #boundary_weight=self.config['boundary_weight'],
        #hausdorff_weight=self.config['hausdorff_weight'],
        #tversky_weight=self.config['tversky_weight'],
        #tversky_alpha=self.config['tversky_alpha'],
        #tversky_beta=self.config['tversky_beta'],
        focal_alpha=self.config['focal_alpha'],
        focal_gamma=self.config['focal_gamma']
    )
        
        # Set up optimizer - only fine-tune the mask decoder
        self.optimizer = optim.Adam(
            [
                {'params': self.sam.mask_decoder.parameters(), 'lr': self.config['learning_rate']}
            ],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        if self.config['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, 
                patience=self.config['patience'], 
                verbose=True, min_lr=self.config['min_lr']
            )
        elif self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config['cosine_epochs'], 
                eta_min=self.config['min_lr']
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config['scheduler']}")
            
        # Set up AMP scaler for mixed precision training
        self.scaler = GradScaler(enabled=self.config['use_amp'])
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rates': [],
        }
        
        # Track best metrics for model saving
        self.best_val_metrics = {
            'loss': float('inf'),
            'dice': 0,
        }
    
    def process_batch(
            self, 
            images: torch.Tensor, 
            target_masks: torch.Tensor, 
            bboxes_list: List, 
            training: bool = True
        ) -> Tuple[Dict[str, float], int]:
        """
        Process a batch of data through the model.
        
        Args:
            images: Batch of images [B, C, H, W]
            target_masks: Batch of target masks [B, H, W]
            bboxes_list: List of bounding boxes for each image
            training: Whether to train the model or just evaluate
            
        Returns:
            Dictionary of metrics and number of valid samples processed
        """
        batch_metrics = MetricTracker()
        valid_count = 0
        
        # Process each image in the batch
        for i in range(images.shape[0]):
            image = images[i]
            # Add channel dimension first, then resize
            target_mask = target_masks[i].unsqueeze(0)  
            target_mask = F.interpolate(
                target_mask.unsqueeze(0), 
                size=(self.mask_size, self.mask_size), 
                mode='nearest'
            ).squeeze(0)
            
            bboxes = bboxes_list[i]
            
            # Skip if no bounding boxes
            if len(bboxes) == 0:
                continue
                
            # Get image embedding - no need to track gradients for image encoder
            with torch.no_grad():
                image_embedding = self.sam.image_encoder(image.unsqueeze(0))
            
            # Process each bounding box to get predictions
            pred_masks = []
            for bbox in bboxes:
                # Convert bbox to tensor and prepare for SAM
                bbox_tensor = torch.tensor(bbox, dtype=torch.float).reshape(1, 1, 4).to(self.device)
                
                # Get prompt embeddings using the proper API
                with torch.no_grad():  # We're not training the prompt encoder
                    sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                        points=None,
                        boxes=bbox_tensor,
                        masks=None
                    )
                
                # Use context manager for mixed precision only in training mode
                if training and self.config['use_amp']:
                    with autocast():
                        # Run mask decoder (this part will have gradients)
                        mask_predictions, _ = self.sam.mask_decoder(
                            image_embeddings=image_embedding,
                            image_pe=self.sam.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False
                        )
                else:
                    # Run mask decoder without mixed precision
                    mask_predictions, _ = self.sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )
                
                pred_masks.append(mask_predictions)
            
            # Combine predictions (take max if multiple bounding boxes)
            if len(pred_masks) > 1:
                combined_pred = pred_masks[0]
                for mask in pred_masks[1:]:
                    combined_pred = torch.maximum(combined_pred, mask)
                pred_mask_logits = combined_pred
            else:
                pred_mask_logits = pred_masks[0]
                
            # Calculate loss and metrics
            if training and self.config['use_amp']:
                with autocast():
                    loss, loss_components = self.combined_loss(pred_mask_logits, target_mask)
            else:
                loss, loss_components = self.combined_loss(pred_mask_logits, target_mask)
                
            # Calculate additional metrics
            with torch.no_grad():
                pred_mask_probs = torch.sigmoid(pred_mask_logits)
                metrics = calculate_metrics(
                    pred_mask_probs.detach().cpu().numpy(),
                    target_mask.detach().cpu().numpy()
                )
                
            # Update metrics with loss components
            metrics.update(loss_components)
            metrics['loss'] = loss.item()
            
            # If in training mode, backward pass
            if training:
                # Scale loss for gradient accumulation
                scaled_loss = loss / self.config['grad_accumulation_steps']
                
                if self.config['use_amp']:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            
            # Update batch metrics
            batch_metrics.update(metrics)
            valid_count += 1
                
        return batch_metrics.avg(), valid_count
    
    def optimizer_step(self) -> None:
        """Perform optimizer step with gradient scaling if enabled."""
        if self.config['use_amp']:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
    
    def train_epoch(self, dataloader, epoch: int) -> Tuple[Dict[str, float], int]:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of average metrics and number of steps
        """
        self.sam.train()
        epoch_metrics = MetricTracker()
        valid_samples = 0
        steps = 0
        
        progress_bar = tqdm(
            enumerate(dataloader), 
            total=len(dataloader), 
            desc=f"Epoch {epoch+1} Training"
        )
        
        # Track accumulated steps
        accumulated_steps = 0
        
        for batch_idx, (images, target_masks, bboxes_list, _) in progress_bar:
            # Move data to device
            images = images.to(self.device)
            target_masks = target_masks.to(self.device)
            
            # Process batch
            batch_metrics, batch_valid_samples = self.process_batch(
                images, target_masks, bboxes_list, training=True
            )
            
            # Skip if no valid samples
            if batch_valid_samples == 0:
                continue
                
            # Update counters
            epoch_metrics.update(batch_metrics, count=batch_valid_samples)
            valid_samples += batch_valid_samples
            accumulated_steps += 1
            
            # Optimizer step after accumulation or at end of epoch
            if (accumulated_steps == self.config['grad_accumulation_steps'] or 
                batch_idx == len(dataloader) - 1):
                self.optimizer_step()
                accumulated_steps = 0
                steps += 1
            
            # Update progress bar
            avg_metrics = epoch_metrics.avg()
            progress_bar.set_postfix({
                'loss': f"{avg_metrics.get('loss', 0):.4f}",
                'dice': f"{avg_metrics.get('dice', 0):.4f}"
            })
        
        return epoch_metrics.avg(), steps
    
    def validate(self, dataloader) -> Dict[str, float]:
        """
        Validate the model on validation data.
        
        Args:
            dataloader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.sam.eval()
        val_metrics = MetricTracker()
        valid_samples = 0

        with torch.no_grad():
            for images, target_masks, bboxes_list, _ in tqdm(dataloader, desc="Validation"):
                # Move data to device
                images = images.to(self.device)
                target_masks = target_masks.to(self.device)
                
                # Process batch
                batch_metrics, batch_valid_samples = self.process_batch(
                    images, target_masks, bboxes_list, training=False
                )
                
                # Skip if no valid samples
                if batch_valid_samples == 0:
                    continue
                    
                # Update metrics
                val_metrics.update(batch_metrics, count=batch_valid_samples)
                valid_samples += batch_valid_samples
        
        if valid_samples == 0:
            logger.warning("No valid samples in validation set!")
            return {'loss': float('inf'), 'dice': 0.0}
            
        return val_metrics.avg()
    
    def train(
            self, 
            train_loader, 
            val_loader, 
            num_epochs: int,
            save_dir: str = "checkpoints",
            experiment_name: str = "sam_finetuned",
            save_interval: int = 5,
            early_stopping_patience: Optional[int] = None
        ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            experiment_name: Name for experiment files
            save_interval: Epochs between saving regular checkpoints
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            Dictionary containing training history
        """
        # Set up directories
        os.makedirs(save_dir, exist_ok=True)
        base_save_path = os.path.join(save_dir, experiment_name)
        
        # Track early stopping
        early_stop_counter = 0
        
        # Save configuration
        config_path = f"{base_save_path}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logger.info(f"Saved configuration to {config_path}")
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            epoch_start_time = time.time()
            train_metrics, train_steps = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Calculate time taken
            epoch_time = time.time() - epoch_start_time
            
            # Update learning rate based on validation loss
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.config['scheduler'] == 'plateau':
                self.scheduler.step(val_metrics['loss'])
            else:  # cosine
                self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_metrics.get('loss', 0))
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['train_dice'].append(train_metrics.get('dice', 0))
            self.history['val_dice'].append(val_metrics.get('dice', 0))
            self.history['learning_rates'].append(current_lr)
            
            # Print metrics
            logger.info(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s:")
            logger.info(f"Train - Loss: {train_metrics.get('loss', 0):.4f}, Dice: {train_metrics.get('dice', 0):.4f}")
            logger.info(f"Val   - Loss: {val_metrics.get('loss', 0):.4f}, Dice: {val_metrics.get('dice', 0):.4f}")
            logger.info(f"Learning rate: {current_lr:.8f}")
            
            # Save model if it's the best so far (by loss)
            if val_metrics['loss'] < self.best_val_metrics['loss']:
                self.best_val_metrics['loss'] = val_metrics['loss']
                checkpoint_path = f"{base_save_path}_best_loss.pth"
                torch.save(self.sam.state_dict(), checkpoint_path)
                logger.info(f"Saved best loss model checkpoint to {checkpoint_path}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            # Save model if it's the best so far (by dice)
            if val_metrics['dice'] > self.best_val_metrics['dice']:
                self.best_val_metrics['dice'] = val_metrics['dice']
                checkpoint_path = f"{base_save_path}_best_dice.pth"
                torch.save(self.sam.state_dict(), checkpoint_path)
                logger.info(f"Saved best dice model checkpoint to {checkpoint_path}")
                early_stop_counter = 0
            
            # Save regular checkpoint at intervals
            if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
                checkpoint_path = f"{base_save_path}_epoch_{epoch+1}.pth"
                torch.save(self.sam.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Also save training history
                history_path = f"{base_save_path}_history.json"
                with open(history_path, 'w') as f:
                    json.dump(self.history, f, indent=4)
            
            # Early stopping check
            if early_stopping_patience and early_stop_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Plot and save training history at end
        self.plot_training_history(save_dir, experiment_name)
        
        return self.history
    
    def plot_training_history(self, save_dir: str, experiment_name: str) -> None:
        """
        Plot and save training history with improved visualization.
        
        Args:
            save_dir: Directory to save plots
            experiment_name: Prefix for plot filenames
        """
        # Set plot style
        plt.style.use('seaborn-v0_8')
        
        # Loss plot
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Save loss plot
        loss_plot_path = os.path.join(save_dir, f"{experiment_name}_loss.png")
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=200)
        plt.close()
        
        # Dice coefficient plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['train_dice'], 'b-', label='Training Dice')
        plt.plot(epochs, self.history['val_dice'], 'r-', label='Validation Dice')
        
        plt.title('Training and Validation Dice Coefficient', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Dice Coefficient', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Save dice plot
        dice_plot_path = os.path.join(save_dir, f"{experiment_name}_dice.png")
        plt.tight_layout()
        plt.savefig(dice_plot_path, dpi=200)
        plt.close()
        
        # Learning rate plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['learning_rates'], 'g-')
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save learning rate plot
        lr_plot_path = os.path.join(save_dir, f"{experiment_name}_lr.png")
        plt.tight_layout()
        plt.savefig(lr_plot_path, dpi=200)
        plt.close()