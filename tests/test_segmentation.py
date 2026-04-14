"""
Tests for the segmentation module.
"""

import numpy as np
import pytest


def test_postprocess_mask_keeps_largest_component():
    """Postprocessing should keep only the largest connected component."""
    from pipeline.segmentation.utils import postprocess_mask

    # Create mask with two blobs — large and small
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1   # large blob
    mask[5:10, 5:10] = 1     # small blob

    cleaned = postprocess_mask(mask)

    # Small blob should be removed
    assert cleaned[7, 7] == 0, "Small blob should be removed"
    # Large blob should remain
    assert cleaned[50, 50] == 1, "Large blob should remain"


def test_postprocess_empty_mask():
    """Postprocessing an empty mask should return it unchanged."""
    from pipeline.segmentation.utils import postprocess_mask

    mask = np.zeros((100, 100), dtype=np.uint8)
    result = postprocess_mask(mask)
    assert result.sum() == 0


def test_compute_area():
    """Area computation should match sum of mask pixels."""
    from pipeline.segmentation.utils import compute_area

    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[100:200, 100:200] = 1  # 100x100 = 10000 pixels

    area = compute_area(mask)
    assert area == 10000


def test_extract_wound_crop():
    """Wound crop should isolate wound and zero out background."""
    from pipeline.segmentation.utils import extract_wound_crop

    img = np.ones((512, 512, 3), dtype=np.uint8) * 128
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[400:600, 400:600] = 1

    crop = extract_wound_crop(img, mask)

    # Crop should not be empty
    assert crop.shape[0] > 0 and crop.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
