"""
Preprocessing for the tissue classifier:
patch extraction + tensor conversion with ImageNet normalization.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from pipeline.config import (
    TISSUE_PATCH_SIZE, TISSUE_STEP,
    TISSUE_INPUT_SIZE, TISSUE_MEAN, TISSUE_STD
)


# Transform pipeline for inference — no augmentation, just resize + normalize.
# (The training notebook uses different augmented transforms for training.)
inference_transform = transforms.Compose([
    transforms.Resize((TISSUE_INPUT_SIZE, TISSUE_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=TISSUE_MEAN, std=TISSUE_STD)
])


def extract_wound_roi(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero out everything outside the wound, so the classifier only sees wound pixels.
    Returns an image the same size as the mask.
    """
    # Bring the image up to the mask's resolution first
    img_resized = cv2.resize(image_rgb, (mask.shape[1], mask.shape[0]))
    wound_image = img_resized.copy()
    wound_image[mask == 0] = 0
    return wound_image


def sample_patches(image_rgb: np.ndarray, mask: np.ndarray,
                   patch_size=TISSUE_PATCH_SIZE, step=TISSUE_STEP):
    """
    Slide a window across the image and yield patches whose center falls
    inside the wound mask. Using a generator to keep memory low for big images.

    Yields:
        (patch, y_center, x_center) for each valid patch
    """
    # Resize the image to match the mask
    img = cv2.resize(image_rgb, (mask.shape[1], mask.shape[0]))
    half = patch_size // 2   # half size — used to center patches on (y, x)

    # Walk across the image in a grid
    for y in range(half, img.shape[0] - half, step):
        for x in range(half, img.shape[1] - half, step):
            # Only keep patches whose center pixel is inside the wound
            if mask[y, x] > 0:
                patch = img[y - half:y + half, x - half:x + half]
                # Skip any weird edge cases where the crop didn't come out right
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    yield patch, y, x


def prepare_patch_tensor(patch: np.ndarray, device) -> torch.Tensor:
    """Take a numpy patch and produce a model-ready tensor (1, 3, 224, 224)."""
    img = Image.fromarray(patch)                     # numpy -> PIL
    tensor = inference_transform(img).unsqueeze(0).to(device)   # transform + batch dim
    return tensor
