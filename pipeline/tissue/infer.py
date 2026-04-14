"""
Tissue classification inference.
Slides patches across the wound region and classifies each one.
"""

import cv2
import torch
import numpy as np

from pipeline.config import (
    TISSUE_CLASSES, TISSUE_PATCH_SIZE, TISSUE_STEP,
    TISSUE_PRED_DIR
)
from pipeline.schemas import TissueResult
from .preprocess import sample_patches, prepare_patch_tensor
from .utils import compute_tissue_percentages, create_tissue_overlay


@torch.no_grad()   # inference only
def classify_patches(image_rgb, mask, model, device,
                     patch_size=TISSUE_PATCH_SIZE, step=TISSUE_STEP):
    """
    Walk across the wound, classify each patch, track everything.

    Returns:
        class_counts: how many patches were classified as each class
        predictions:  list of (y, x, class_name) for drawing the overlay later
    """
    # Start counts at 0 for each class so we always have consistent keys
    class_counts = {cls: 0 for cls in TISSUE_CLASSES}
    predictions = []

    # For each patch from the wound region, classify it and record the result
    for patch, y, x in sample_patches(image_rgb, mask, patch_size, step):
        tensor = prepare_patch_tensor(patch, device)
        output = model(tensor)
        _, pred_idx = torch.max(output, 1)     # argmax — pick highest scoring class
        class_name = TISSUE_CLASSES[pred_idx.item()]

        class_counts[class_name] += 1
        predictions.append((y, x, class_name))

    return class_counts, predictions


def run_tissue_classification(image_rgb, mask, model, device,
                              image_id="image", save=True) -> TissueResult:
    """
    Top-level tissue classification function.
    Classifies all patches, computes percentages, optionally saves
    a visualization, and returns a TissueResult.
    """
    class_counts, predictions = classify_patches(
        image_rgb, mask, model, device
    )

    pct = compute_tissue_percentages(class_counts)
    total_patches = sum(class_counts.values())
    tissue_map_path = ""

    # Save the visual overlay if requested and we actually classified anything
    if save and total_patches > 0:
        try:
            overlay = create_tissue_overlay(image_rgb, mask, predictions)
            tissue_map_path = str(TISSUE_PRED_DIR / f"{image_id}_tissue_map.png")
            cv2.imwrite(tissue_map_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        except PermissionError as e:
            print(f"  [tissue] Warning: could not save overlay: {e}")

    print(f"  [tissue] G={pct['granulation_pct']:.1f}% "
          f"S={pct['slough_pct']:.1f}% N={pct['necrosis_pct']:.1f}% "
          f"({total_patches} patches)")

    return TissueResult(
        image_id=image_id,
        granulation_pct=pct["granulation_pct"],
        slough_pct=pct["slough_pct"],
        necrosis_pct=pct["necrosis_pct"],
        tissue_map_path=tissue_map_path,
        total_patches=total_patches
    )
