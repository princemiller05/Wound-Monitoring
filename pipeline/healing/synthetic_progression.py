"""
Synthetic longitudinal wound progression.

Since we usually only have ONE image per patient (day 0), we simulate
what the wound might look like on days 7, 14, and 21 so XGBoost has
enough time points to work with.

This is obviously a simplification — in a real deployment, we'd use
actual follow-up images. But for the demo pipeline, synthetic data lets
us show the full flow end-to-end.
"""

import cv2
import numpy as np

from pipeline.config import HEALING_DAYS


def generate_healing_sequence(base_mask: np.ndarray, seed=42) -> dict:
    """
    Simulate a wound that IS healing.
    We shrink the mask a little more every week by eroding it.

    Returns:
        {day: mask} — e.g. {0: ..., 7: ..., 14: ..., 21: ...}
    """
    kernel = np.ones((7, 7), np.uint8)
    sequence = {HEALING_DAYS[0]: base_mask.copy()}

    current = base_mask.copy()
    # Each week we erode a bit more than the previous — wound getting smaller
    for i, day in enumerate(HEALING_DAYS[1:], 1):
        current = cv2.erode(current, kernel, iterations=2 * i)
        sequence[day] = current.copy()

    return sequence


def generate_non_healing_sequence(base_mask: np.ndarray, seed=42) -> dict:
    """
    Simulate a wound that is NOT healing.
    Area bounces around a bit but never really shrinks — stagnant wound.
    """
    kernel = np.ones((5, 5), np.uint8)
    sequence = {
        HEALING_DAYS[0]: base_mask.copy(),
        HEALING_DAYS[1]: cv2.erode(base_mask, kernel, iterations=1),
        HEALING_DAYS[2]: cv2.dilate(base_mask, kernel, iterations=1),
        HEALING_DAYS[3]: base_mask.copy(),   # back to original size
    }
    return sequence


def generate_tissue_progression(mode="healing", seed=42) -> dict:
    """
    Simulate how tissue composition changes over time.
    Healing: granulation goes UP, necrosis goes DOWN (wound filling in).
    Non-healing: necrosis stays high, nothing improves.
    """
    rng = np.random.default_rng(seed)

    if mode == "healing":
        # Good trajectory — healthy tissue gradually taking over
        progression = {
            0:  {"granulation_pct": 30, "slough_pct": 40, "necrosis_pct": 30},
            7:  {"granulation_pct": 45, "slough_pct": 35, "necrosis_pct": 20},
            14: {"granulation_pct": 60, "slough_pct": 28, "necrosis_pct": 12},
            21: {"granulation_pct": 75, "slough_pct": 20, "necrosis_pct":  5},
        }
    else:
        # Bad trajectory — stuck or getting worse
        progression = {
            0:  {"granulation_pct": 20, "slough_pct": 35, "necrosis_pct": 45},
            7:  {"granulation_pct": 22, "slough_pct": 38, "necrosis_pct": 40},
            14: {"granulation_pct": 18, "slough_pct": 37, "necrosis_pct": 45},
            21: {"granulation_pct": 20, "slough_pct": 35, "necrosis_pct": 45},
        }

    # Add a bit of Gaussian noise so the synthetic data doesn't look too "perfect"
    for day in progression:
        for key in progression[day]:
            noise = rng.normal(0, 2)
            progression[day][key] = max(0, progression[day][key] + noise)

    return progression
