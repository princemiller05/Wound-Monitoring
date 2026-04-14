"""
Small helpers for loading and prepping images before segmentation.
Kept separate from infer.py so they're easy to reuse elsewhere.
"""

import cv2
import numpy as np


def load_image_rgb(path: str) -> np.ndarray:
    """
    Load an image from disk in RGB order.
    OpenCV reads in BGR by default (legacy reasons), which is annoying,
    so we convert it here to keep everything downstream in RGB.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_to_1024(img_rgb: np.ndarray) -> np.ndarray:
    """MedSAM was trained on 1024x1024 images — everything has to fit that."""
    return cv2.resize(img_rgb, (1024, 1024))
