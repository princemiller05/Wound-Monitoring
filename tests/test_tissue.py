"""
Tests for the tissue classification module.
"""

import numpy as np
import pytest


def test_compute_tissue_percentages():
    """Percentages should sum to ~100."""
    from pipeline.tissue.utils import compute_tissue_percentages

    counts = {"granulation": 50, "necrosis": 30, "slough": 20}
    pct = compute_tissue_percentages(counts)

    total = pct["granulation_pct"] + pct["necrosis_pct"] + pct["slough_pct"]
    assert abs(total - 100.0) < 0.1


def test_compute_tissue_percentages_empty():
    """Empty counts should return all zeros."""
    from pipeline.tissue.utils import compute_tissue_percentages

    counts = {"granulation": 0, "necrosis": 0, "slough": 0}
    pct = compute_tissue_percentages(counts)

    assert pct["granulation_pct"] == 0.0
    assert pct["necrosis_pct"] == 0.0
    assert pct["slough_pct"] == 0.0


def test_sample_patches_within_mask():
    """Patches should only come from masked regions."""
    from pipeline.tissue.preprocess import sample_patches

    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 1  # wound in center

    patches = list(sample_patches(img, mask, patch_size=64, step=32))

    # All patch centers should be within masked region
    for patch, y, x in patches:
        assert mask[y, x] == 1, f"Patch at ({y},{x}) is outside wound mask"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
