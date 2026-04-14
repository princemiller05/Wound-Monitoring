"""
Post-processing helpers for segmentation output.

Raw MedSAM masks are usually pretty good but sometimes have small stray
blobs or rough edges — these helpers clean all that up.
"""

import cv2
import numpy as np
from skimage import morphology


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean up a raw binary mask:
    1. Keep only the largest connected component (removes stray pixels)
    2. Morphological close to fill small holes
    3. Morphological open to smooth edges

    Returns the cleaned mask — same shape, still binary uint8.
    """
    # Label each connected blob with a unique number
    labeled = morphology.label(mask)
    if labeled.max() == 0:
        return mask   # empty mask, nothing to do

    # Figure out which blob is biggest and keep only that one
    counts = np.bincount(labeled.flat)
    largest = counts[1:].argmax() + 1   # +1 because index 0 is background
    cleaned = (labeled == largest).astype(np.uint8)

    # Smooth the edges with morphological ops (15x15 ellipse kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)   # fills holes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)    # smooths outline
    return cleaned


def extract_wound_crop(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop to just the wound region and black out everything that isn't wound.
    Useful for downstream tasks where you want to focus on the wound only.
    """
    # Make the image match the mask size (1024x1024)
    img_1024 = cv2.resize(img_rgb, (1024, 1024))

    # Find the bounding box of the mask
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return img_1024   # no wound found — return the whole image

    y1, y2 = coords[0].min(), coords[0].max()
    x1, x2 = coords[1].min(), coords[1].max()

    # Crop the image and the mask the same way
    cropped_img = img_1024[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    # Zero out non-wound pixels so the background is pure black
    result = cropped_img.copy()
    result[cropped_mask == 0] = 0
    return result


def create_overlay(img_rgb: np.ndarray, mask: np.ndarray,
                   color=(0, 255, 0), thickness=3) -> np.ndarray:
    """
    Draw the wound contour on the original image in green.
    Great for visualizing how well the segmentation worked.
    """
    img_show = cv2.resize(img_rgb, (1024, 1024))
    # Find the outline of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    overlay = img_show.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def compute_area(mask: np.ndarray) -> int:
    """Area of the wound in pixels — just count the non-zero mask pixels."""
    return int(mask.sum())
