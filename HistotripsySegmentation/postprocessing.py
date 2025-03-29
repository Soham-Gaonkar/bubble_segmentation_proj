# postprocessing.py
import cv2
import numpy as np
import torch

def area_thresholding(prediction, area_threshold):
    """
    Removes connected components in the prediction that are smaller than area_threshold.

    Args:
        prediction (torch.Tensor): Binary segmentation prediction (0 or 1).
        area_threshold (int): Minimum area for a connected component to be kept.

    Returns:
        torch.Tensor: Post-processed binary segmentation prediction.
    """
    prediction = prediction.squeeze().cpu().numpy().astype(np.uint8)  # Convert to numpy/cv2-compatible format
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(prediction, connectivity=8)

    # Iterate through connected components, remove small ones
    for i in range(1, num_labels):  # Start from 1 to skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_threshold:
            prediction[labels == i] = 0

    return torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0).float()  # Back to tensor