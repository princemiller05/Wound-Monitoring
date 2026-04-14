"""
Build the feature tables that XGBoost needs.

Two levels:
  1. Longitudinal dataset — one row per (wound, day) — all the raw measurements
  2. Feature table — one row per wound — summary stats used for prediction
"""

import numpy as np
import pandas as pd

from pipeline.config import HEALING_DAYS, FEATURE_COLS


def build_longitudinal_dataset(wound_id: str, masks_dict: dict,
                                tissue_dict: dict = None) -> pd.DataFrame:
    """
    Build a row-per-day dataframe for a single wound.

    Args:
        wound_id:    identifier string (e.g. "W1" or "CASE_001")
        masks_dict:  {day: mask} from either real visits or synthetic progression
        tissue_dict: optional {day: {"granulation_pct", ...}}

    Returns:
        DataFrame with columns:
        wound_id, day, area_pixels, granulation_pct, slough_pct, necrosis_pct
    """
    rows = []
    for day in sorted(masks_dict.keys()):
        mask = masks_dict[day]
        area = int(np.sum(mask > 0))   # count wound pixels

        row = {
            "wound_id": wound_id,
            "day": day,
            "area_pixels": area,
        }

        # If we have tissue data for this day, pull it in. Otherwise zeros.
        if tissue_dict and day in tissue_dict:
            row["granulation_pct"] = tissue_dict[day].get("granulation_pct", 0)
            row["slough_pct"]      = tissue_dict[day].get("slough_pct", 0)
            row["necrosis_pct"]    = tissue_dict[day].get("necrosis_pct", 0)
        else:
            row["granulation_pct"] = 0.0
            row["slough_pct"]      = 0.0
            row["necrosis_pct"]    = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def build_features_table(longitudinal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the per-day data into one row per wound with summary features.

    Features used:
        - initial_area       (area on day 0)
        - final_area         (area on last day)
        - pct_area_reduction (how much it shrank, as a fraction)
        - mean_area          (average across all days)
        - std_area           (how much it varied)

    These are what XGBoost was trained on.
    """
    features = []

    # Loop one wound at a time
    for wound_id, group in longitudinal_df.groupby("wound_id"):
        group = group.sort_values("day")
        areas = group["area_pixels"].values

        initial = areas[0]
        final = areas[-1]
        # Guard against divide-by-zero if somehow the initial area is 0
        reduction = (initial - final) / initial if initial > 0 else 0.0

        feat = {
            "wound_id": wound_id,
            "initial_area": initial,
            "final_area": final,
            "pct_area_reduction": round(reduction, 6),
            "mean_area": round(np.mean(areas), 2),
            "std_area": round(np.std(areas), 2),
        }
        features.append(feat)

    return pd.DataFrame(features)
