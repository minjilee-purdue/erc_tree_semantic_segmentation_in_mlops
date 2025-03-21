# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def standardize_dimensions(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standardizes input dimensions to ensure compatibility.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        
    Returns:
        Tuple of standardized prediction and target tensors
    """
    # Handle channel dimension differences
    if pred.dim() == 4 and target.dim() == 3:
        # If pred is [B, C, H, W] and target is [B, H, W]
        if pred.size(1) == 1:
            pred = pred.squeeze(1)  # Convert to [B, H, W]
        else:
            # Multi-class case, would need special handling
            raise ValueError("Multi-channel predictions not supported with single-channel targets")
    elif pred.dim() == 3 and target.dim() == 4:
        # If pred is [B, H, W] and target is [B, C, H, W]
        if target.size(1) == 1:
            target = target.squeeze(1)  # Convert to [B, H, W]
        else:
            # Multi-class case, would need special handling
            raise ValueError("Single-channel predictions not supported with multi-channel targets")
            
    # Ensure shapes match
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch after standardization: {pred.shape} vs {target.shape}")
        
    return pred, target









class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Focal Loss for binary segmentation.
        
        Args:
            alpha: Weighting factor for the rare class (usually foreground)
            gamma: Focusing parameter - higher values reduce the relative loss for well-classified examples
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            logits: Raw logits from model
            targets: Target binary masks
            
        Returns:
            Focal loss value
        """
        # Standardize dimensions
        logits, targets = standardize_dimensions(logits, targets)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate final loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, from_logits: bool = False):
        """
        Dice Loss for binary segmentation.
        
        Args:
            smooth: Small constant to avoid division by zero
            from_logits: Whether input is logits (requiring sigmoid) or probabilities
        """
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits
    
    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss
        
        Args:
            pred: Predicted values (logits or probabilities)
            targets: Target binary masks
            
        Returns:
            Dice loss value
        """
        # Apply sigmoid if inputs are logits
        if self.from_logits:
            pred = torch.sigmoid(pred)
            
        # Standardize dimensions
        pred, targets = standardize_dimensions(pred, targets)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * targets_flat).sum()
        
        # Calculate Dice coefficient and loss
        dice_coef = (2. * intersection + self.smooth) / (
            pred_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        # Return 1 - dice coefficient as the loss
        return 1 - dice_coef


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5, 
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """
        Combined Dice and Focal loss with configurable weights.
        
        Args:
            dice_weight: Weight for Dice loss component
            focal_weight: Weight for Focal loss component
            focal_alpha: Alpha parameter for Focal loss
            focal_gamma: Gamma parameter for Focal loss
        """
        super().__init__()
        self.dice_loss = DiceLoss(from_logits=True)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss
        
        Args:
            logits: Raw logits from model
            targets: Target binary masks
            
        Returns:
            Combined loss value and dictionary with individual loss components
        """
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        
        # Combine losses
        combined = self.focal_weight * focal + self.dice_weight * dice
        
        # Return combined loss and individual components for logging
        return combined, {
            'focal_loss': focal.item(),
            'dice_loss': dice.item()
        }
