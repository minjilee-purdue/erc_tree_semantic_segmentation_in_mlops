import numpy as np
from typing import Dict, Any, Union, Tuple


def calculate_metrics(pred_mask: np.ndarray, target_mask: np.ndarray, 
                      smooth: float = 1e-5) -> Dict[str, float]:
    """
    Calculate evaluation metrics for segmentation.
    
    Args:
        pred_mask: Predicted binary mask
        target_mask: Ground truth binary mask
        smooth: Small constant to avoid division by zero
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Ensure binary masks
    pred_mask_bin = pred_mask > 0.5
    target_mask_bin = target_mask > 0.5
    
    # Calculate true positives, false positives, false negatives
    true_pos = np.sum(pred_mask_bin & target_mask_bin)
    false_pos = np.sum(pred_mask_bin & ~target_mask_bin)
    false_neg = np.sum(~pred_mask_bin & target_mask_bin)
    
    # Calculate metrics
    precision = true_pos / (true_pos + false_pos + smooth)
    recall = true_pos / (true_pos + false_neg + smooth)
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall + smooth)
    
    # IoU (Jaccard Index)
    iou = true_pos / (true_pos + false_pos + false_neg + smooth)
    
    # Dice coefficient (F1 and Dice are equivalent for binary case)
    dice = 2 * true_pos / (2 * true_pos + false_pos + false_neg + smooth)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'iou': float(iou),
        'dice': float(dice)
    }
