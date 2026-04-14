"""
Tests for the healing prediction module.
"""

import numpy as np
import pandas as pd
import pytest


def test_healing_sequence_shrinks():
    """Healing sequence masks should progressively shrink."""
    from pipeline.healing.synthetic_progression import generate_healing_sequence

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1

    seq = generate_healing_sequence(mask)
    areas = [seq[day].sum() for day in sorted(seq.keys())]

    # Each subsequent mask should be smaller or equal
    for i in range(1, len(areas)):
        assert areas[i] <= areas[i - 1], \
            f"Day {sorted(seq.keys())[i]}: area should decrease"


def test_build_features_table():
    """Feature table should produce expected columns."""
    from pipeline.healing.features import build_features_table

    df = pd.DataFrame({
        "wound_id": ["W1"] * 4,
        "day": [0, 7, 14, 21],
        "area_pixels": [1000, 800, 600, 400],
        "granulation_pct": [30, 45, 60, 75],
        "slough_pct": [40, 35, 28, 20],
        "necrosis_pct": [30, 20, 12, 5],
    })

    features = build_features_table(df)

    assert len(features) == 1
    assert "initial_area" in features.columns
    assert "pct_area_reduction" in features.columns
    assert features.iloc[0]["pct_area_reduction"] == 0.6  # (1000-400)/1000


def test_rule_baseline():
    """Rule baseline should classify correctly at threshold."""
    from pipeline.healing.rule_baseline import apply_rule_baseline

    assert apply_rule_baseline(0.35) == "healing"
    assert apply_rule_baseline(0.30) == "healing"
    assert apply_rule_baseline(0.29) == "non_healing"
    assert apply_rule_baseline(0.0) == "non_healing"


def test_longitudinal_dataset():
    """Longitudinal dataset should have correct structure."""
    from pipeline.healing.features import build_longitudinal_dataset

    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 1

    masks_dict = {0: mask, 7: mask, 14: mask, 21: mask}
    df = build_longitudinal_dataset("W1", masks_dict)

    assert len(df) == 4
    assert list(df.columns) == [
        "wound_id", "day", "area_pixels",
        "granulation_pct", "slough_pct", "necrosis_pct"
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
