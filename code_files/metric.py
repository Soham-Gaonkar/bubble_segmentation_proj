# metric.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import directed_hausdorff, cdist
from scipy.ndimage import binary_erosion
import warnings

# --- Helper Function for Boundary Extraction ---

def get_boundary_coords(mask):
    """
    Extracts the coordinates of the boundary pixels from a binary mask.

    Args:
        mask (np.ndarray): A binary 2D numpy array (0s and 1s). Should be bool or uint8.

    Returns:
        np.ndarray: An Nx2 array of [row, col] coordinates of boundary pixels,
                    or None if the mask is empty or has no boundary.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool) # Morphology functions often expect bool

    if not mask.any():
        return None # No foreground pixels

    # Erode the mask to find internal pixels
    eroded_mask = binary_erosion(mask, border_value=0) # Assume background is 0

    # Boundary is the original mask XOR the eroded mask (more robust than subtraction)
    boundary_mask = mask ^ eroded_mask

    if not boundary_mask.any():
         # Handle cases where the object is too small (e.g., single pixel)
         # Consider all foreground pixels as boundary in this case
         coords = np.argwhere(mask)
         return coords if coords.size > 0 else None

    # Get coordinates of boundary pixels
    coords = np.argwhere(boundary_mask) # Returns pairs of [row, col]
    return coords

# --- Main Metric Calculation Function ---

def calculate_all_metrics(predictions, targets, threshold=0.5):
    """
    Calculates a comprehensive set of segmentation metrics. Assumes binary segmentation.

    Args:
        predictions (torch.Tensor): Model output (logits or probabilities). Shape (B, 1, H, W).
        targets (torch.Tensor): Ground truth labels (0 or 1). Shape (B, 1, H, W).
        threshold (float): Threshold for converting probabilities to binary predictions.

    Returns:
        dict: A dictionary containing all calculated metrics.
    """
    if predictions.ndim != 4 or targets.ndim != 4:
        raise ValueError("Inputs must be 4D tensors (B, C, H, W)")
    if predictions.shape[1] != 1 or targets.shape[1] != 1:
         raise ValueError("Inputs must be single-channel (B, 1, H, W)")

    # --- Preprocessing ---
    # Apply sigmoid if predictions are logits
    preds_prob = torch.sigmoid(predictions)
    # Move to CPU and convert to NumPy for calculations
    preds_prob_np = preds_prob.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy().astype(np.uint8) # Ensure uint8 for morphology/counts

    # Binarize predictions based on threshold
    preds_binary_np = (preds_prob_np > threshold).astype(np.uint8)

    # Flatten entire batch for overall pixel counts (or calculate per image and average)
    # Let's calculate overall for TP/TN/FP/FN based metrics for simplicity first
    targets_flat = targets_np.flatten()
    preds_binary_flat = preds_binary_np.flatten()

    # --- Basic Pixel Counts ---
    TP = np.sum((preds_binary_flat == 1) & (targets_flat == 1))
    TN = np.sum((preds_binary_flat == 0) & (targets_flat == 0))
    FP = np.sum((preds_binary_flat == 1) & (targets_flat == 0))
    FN = np.sum((preds_binary_flat == 0) & (targets_flat == 1))

    epsilon = 1e-7 # Small value to prevent division by zero

    # --- Standard Pixel-Based Metrics ---
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon) # Also Sensitivity
    specificity = TN / (TN + FP + epsilon)
    dice_coefficient = (2 * TP) / (2 * TP + FP + FN + epsilon)
    iou = TP / (TP + FP + FN + epsilon) # Jaccard
    false_positive_rate = FP / (FP + TN + epsilon)
    false_negative_rate = FN / (FN + TP + epsilon)

    # --- Boundary F1 Score (BF Score) ---
    # Based on the user's image, BF Score = F1 score from pixel precision/recall
    boundary_f1_score = (2 * precision * recall) / (precision + recall + epsilon)

    # --- Specific AUC Formula (from image) ---
    # AUC = TP / 2(TP + FN) + TN / 2(FP + TN)
    auc_term1 = TP / (2 * (TP + FN) + epsilon)
    auc_term2 = TN / (2 * (FP + TN) + epsilon)
    auc_specific = auc_term1 + auc_term2

    # --- Area Under ROC Curve (AUROC) ---
    # Calculated separately as it needs probabilities
    try:
        targets_roc_flat = targets_np.flatten()
        preds_prob_roc_flat = preds_prob_np.flatten()
        if len(np.unique(targets_roc_flat)) > 1:
            auroc = roc_auc_score(targets_roc_flat, preds_prob_roc_flat)
        else:
            # warnings.warn("Only one class present in target labels for AUROC calculation. Setting AUROC to 0.5.")
            auroc = 0.5 # Undefined or trivial case
    except ValueError as e:
        warnings.warn(f"AUROC calculation failed: {e}. Setting AUROC to 0.5.")
        auroc = 0.5

    # --- Hausdorff Distance Calculation (Average over Batch) ---
    batch_size = preds_binary_np.shape[0]
    hausdorff_means = []
    hausdorff_maxs = []

    for i in range(batch_size):
        pred_slice = preds_binary_np[i, 0] # Get H, W slice
        target_slice = targets_np[i, 0]

        pred_coords = get_boundary_coords(pred_slice)
        target_coords = get_boundary_coords(target_slice)

        current_mean_hd = np.nan # Use NaN as placeholder
        current_max_hd = np.nan

        if pred_coords is not None and target_coords is not None and pred_coords.shape[0] > 0 and target_coords.shape[0] > 0:
            try:
                # Directed Hausdorff distances
                h_pred_to_target, _, _ = directed_hausdorff(pred_coords, target_coords)
                h_target_to_pred, _, _ = directed_hausdorff(target_coords, pred_coords)

                # Max Hausdorff (symmetric)
                current_max_hd = max(h_pred_to_target, h_target_to_pred)

                # Mean Hausdorff (average of mean directed distances)
                dist_pred_to_target = cdist(pred_coords, target_coords).min(axis=1)
                mean_dist_pred = dist_pred_to_target.mean()
                dist_target_to_pred = cdist(target_coords, pred_coords).min(axis=1)
                mean_dist_target = dist_target_to_pred.mean()
                current_mean_hd = (mean_dist_pred + mean_dist_target) / 2.0

            except Exception as e:
                 warnings.warn(f"Hausdorff calculation failed for sample {i}: {e}")
                 # Keep placeholders as NaN

        hausdorff_maxs.append(current_max_hd)
        hausdorff_means.append(current_mean_hd)

    # Calculate batch average, ignoring NaNs
    mean_hausdorff_batch = np.nanmean(hausdorff_means) if not np.all(np.isnan(hausdorff_means)) else -1.0
    max_hausdorff_batch = np.nanmean(hausdorff_maxs) if not np.all(np.isnan(hausdorff_maxs)) else -1.0


    # --- Paper Specific Metrics ---
    # Accuracy (Paper Definition: TP / (TP + FN) which is Recall)
    paper_accuracy = recall # It's just recall

    # IoU (Paper Definition: Standard IoU)
    paper_iou = iou

    # BF Score (Paper Definition: Standard F1 from Precision/Recall)
    paper_bf_score = boundary_f1_score


    # --- Assemble Final Dictionary ---
    results = {
        "Accuracy": accuracy,
        "Dice Coefficient": dice_coefficient,
        "IoU": iou,
        "Boundary F1 Score": boundary_f1_score, # Standard F1 based on pixel P/R
        "AUROC": auroc,                          # Area under ROC curve (needs probabilities)
        "AuC": auc_specific,                     # Specific formula from image
        "BF Score": boundary_f1_score,           # Explicitly add BF Score again (same as Boundary F1)
        "Mean Hausdorff": mean_hausdorff_batch,
        "Max Hausdorff": max_hausdorff_batch,
        "False Positive Rate": false_positive_rate,
        "False Negative Rate": false_negative_rate,
        # Paper Metrics (with distinct keys)
        "Accuracy (Paper)": paper_accuracy,
        "IoU (Paper)": paper_iou,
        "BF Score (Paper)": paper_bf_score,
        # Optional bonus metrics often reported
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity
    }

    return results