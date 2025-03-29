# metric.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculates segmentation metrics.  Assumes binary segmentation.

    Args:
        predictions (torch.Tensor): Model output (probabilities or logits).
        targets (torch.Tensor): Ground truth labels (0 or 1).
        threshold (float): Threshold for converting probabilities to binary predictions.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    predictions = torch.sigmoid(predictions) # Apply sigmoid if logits

    predictions = (predictions > threshold).int()
    targets = targets.int()

    TP = torch.sum((predictions == 1) & (targets == 1)).item()
    TN = torch.sum((predictions == 0) & (targets == 0)).item()
    FP = torch.sum((predictions == 1) & (targets == 0)).item()
    FN = torch.sum((predictions == 0) & (targets == 1)).item()

    epsilon = 1e-7  # Small value to prevent division by zero

    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)

    dice_coefficient = (2 * TP) / (2 * TP + FP + FN + epsilon)
    iou = TP / (TP + FP + FN + epsilon)
    boundary_f1_score = (2 * precision * recall) / (precision + recall + epsilon)

    # Hausdorff distance (using approximation since it's computationally expensive)
    # Replace with a library like SciPy's `scipy.spatial.distance.directed_hausdorff` if you want true Hausdorff
    mean_hausdorff = 0 # Placeholder, implement approximation here
    max_hausdorff = 0  # Placeholder, implement approximation here

    false_positive_rate = FP / (FP + TN + epsilon)
    false_negative_rate = FN / (FN + TP + epsilon)
    
    return {
        "Accuracy": accuracy,
        "Dice Coefficient": dice_coefficient,
        "IoU": iou,
        "Boundary F1 Score": boundary_f1_score,
        "Mean Hausdorff": mean_hausdorff,
        "Max Hausdorff": max_hausdorff,
        "False Positive Rate": false_positive_rate,
        "False Negative Rate": false_negative_rate
    }

def calculate_auroc(predictions, targets):
    """Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)."""
    # Ensure predictions are probabilities (apply sigmoid if they're logits)
    predictions = torch.sigmoid(predictions).cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    try:
        auroc = roc_auc_score(targets, predictions)
    except ValueError:
        auroc = 0.5  # Handle cases where only one class is present in the batch
    return auroc

def calculate_paper_metrics(predictions, targets, threshold=0.5):
    """Calculates the metrics as defined in the paper (including the potentially incorrect accuracy)."""
    # Apply sigmoid and threshold
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).int()
    targets = targets.int()

    TP = torch.sum((predictions == 1) & (targets == 1)).item()
    TN = torch.sum((predictions == 0) & (targets == 0)).item()
    FP = torch.sum((predictions == 1) & (targets == 0)).item()
    FN = torch.sum((predictions == 0) & (targets == 1)).item()

    epsilon = 1e-7  # Small value to prevent division by zero

    # Original paper accuracy (likely incorrect)
    paper_accuracy = TP / (TP + FN + epsilon)

    #IoU
    iou = TP / (TP + FP + FN + epsilon)

    #BF Score
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    bf_score = (2 * precision * recall) / (precision + recall + epsilon)
    
    #AUC is already calculated in calculate_auroc()

    return {
        "Accuracy": paper_accuracy,  # This is the 'incorrect' one
        "IoU": iou,
        "BF Score": bf_score,
    }