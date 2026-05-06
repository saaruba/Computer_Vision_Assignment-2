"""
Reusable numeric metrics for feature analysis and tracking evaluation.
This module centralizes common error and summary calculations used in the
assignment pipeline. Keeping these functions in one place avoids duplicated
logic and makes evaluation results consistent across scripts.
"""

import numpy as np


def translation_error(pred_x, pred_y, gt_x, gt_y):
    """
    Compute 2D translation error between predicted and ground-truth centroids.

    """
    dx = float(pred_x) - float(gt_x)
    dy = float(pred_y) - float(gt_y)
    return float(np.sqrt(dx * dx + dy * dy))


def rotation_error(pred_theta, gt_theta):
    """
    Compute wrapped orientation error between predicted and true angles.

    """
    diff = abs(float(pred_theta) - float(gt_theta)) % 180.0
    return float(min(diff, 180.0 - diff))


def rmse(values):
    """
    Compute root mean square error (RMSE) from an array-like sequence.and here we have used RMSE and it emphasizes larger errors and is widely used in tracking/reporting
    to summarize overall prediction quality in one number.

    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float(np.nan)
    return float(np.sqrt(np.mean(np.square(arr))))


def summary_stats(values):
    
    # Compute mean, standard deviation, minimum, and maximum.
    
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
