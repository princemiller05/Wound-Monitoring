"""
Visualization and aggregation helpers for tissue classification.
"""

import cv2
import numpy as np

from pipeline.config import TISSUE_COLORS, TISSUE_PATCH_SIZE


def create_tissue_overlay(image_rgb: np.ndarray, mask: np.ndarray,
                          predictions: list, patch_size=TISSUE_PATCH_SIZE,
                          alpha=0.6) -> np.ndarray:
    """
    Draw a colored patch over each classified region so you can SEE what
    the model decided. Red = granulation, black = necrosis, white = slough.

    Args:
        image_rgb:   original image (any size)
        mask:        binary wound mask (used to get dimensions)
        predictions: list of (y_center, x_center, class_name)
        patch_size:  matches whatever we classified at
        alpha:       transparency of the overlay (0 = invisible, 1 = solid)
    """
    # Bring image to the same resolution as the mask
    img = cv2.resize(image_rgb, (mask.shape[1], mask.shape[0]))
    output = img.copy()
    half = patch_size // 2

    # For each classified patch, paint a colored square on the output
    for y, x, class_name in predictions:
        color = TISSUE_COLORS.get(class_name, (128, 128, 128))   # gray fallback
        # Guard against going out of bounds at the edges
        y1, y2 = max(0, y - half), min(output.shape[0], y + half)
        x1, x2 = max(0, x - half), min(output.shape[1], x + half)

        # Blend the color into the image with the given alpha
        output[y1:y2, x1:x2] = (
            (1 - alpha) * output[y1:y2, x1:x2] +
            alpha * np.array(color)
        ).astype(np.uint8)

    return output


def compute_tissue_percentages(class_counts: dict) -> dict:
    """
    Turn raw patch counts into percentages.

    Input:  {"granulation": 30, "necrosis": 10, "slough": 20}
    Output: {"granulation_pct": 50.0, "necrosis_pct": 16.67, "slough_pct": 33.33}
    """
    total = sum(class_counts.values())
    if total == 0:
        # No patches found — return all zeros to avoid divide-by-zero
        return {
            "granulation_pct": 0.0,
            "slough_pct": 0.0,
            "necrosis_pct": 0.0,
        }

    return {
        "granulation_pct": round(class_counts.get("granulation", 0) / total * 100, 2),
        "slough_pct":      round(class_counts.get("slough", 0)      / total * 100, 2),
        "necrosis_pct":    round(class_counts.get("necrosis", 0)    / total * 100, 2),
    }
