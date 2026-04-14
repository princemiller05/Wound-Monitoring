"""
Healing prediction inference.

Takes a Day-0 mask, simulates forward progression (or uses real multi-visit
masks), extracts features, runs XGBoost, and returns the result.
"""

import numpy as np
import pandas as pd

from pipeline.config import FEATURE_COLS, HEALING_THRESHOLD, HEALING_DIR
from pipeline.schemas import HealingResult
from .synthetic_progression import (
    generate_healing_sequence, generate_non_healing_sequence,
    generate_tissue_progression
)
from .features import build_longitudinal_dataset, build_features_table
from .rule_baseline import apply_rule_baseline
from .utils import plot_mask_progression, plot_area_trend, plot_tissue_trend


def run_healing_prediction(
    base_mask,
    healing_model,
    case_id="CASE_001",
    tissue_result=None,
    simulate_mode="auto",
    seed=42,
    save=True,
) -> HealingResult:
    """
    Predict healing outcome for a single wound.

    Since we usually only have a Day-0 image, we simulate what the wound
    might look like over 21 days and let XGBoost score that trajectory.

    Args:
        base_mask:      Day-0 binary mask from segmentation
        healing_model:  trained XGBoost model
        case_id:        identifier for the patient/case
        tissue_result:  optional TissueResult (not currently used in features)
        simulate_mode:  "auto" | "healing" | "non_healing"
        seed:           random seed for reproducibility
        save:           save plots + CSVs to outputs/healing/

    Returns:
        HealingResult with probability, predicted label, rule label, key factors.
    """
    rng = np.random.default_rng(seed)

    # ── 1. Pick a simulation mode if auto ────────────────
    # "auto" just randomly picks healing or non-healing — useful for demos
    # where you want to showcase both paths.
    if simulate_mode == "auto":
        simulate_mode = "healing" if rng.random() > 0.5 else "non_healing"

    # ── 2. Generate a synthetic mask sequence across 4 days ──
    if simulate_mode == "healing":
        mask_sequence = generate_healing_sequence(base_mask, seed=seed)
    else:
        mask_sequence = generate_non_healing_sequence(base_mask, seed=seed)

    # ── 3. Generate a synthetic tissue composition trajectory ──
    tissue_progression = generate_tissue_progression(mode=simulate_mode, seed=seed)

    # ── 4. Build the longitudinal dataset (row per day) ──
    longitudinal_df = build_longitudinal_dataset(
        wound_id=case_id,
        masks_dict=mask_sequence,
        tissue_dict=tissue_progression
    )

    # ── 5. Build the feature row (one per wound) ──
    features_df = build_features_table(longitudinal_df)
    if len(features_df) == 0:
        raise ValueError("Feature table is empty — check mask quality")

    # ── 6. Run XGBoost ──
    X = features_df[FEATURE_COLS]
    healing_prob = float(healing_model.predict_proba(X)[0][1])   # P(healing)
    predicted_label = "healing" if healing_prob >= HEALING_THRESHOLD else "non_healing"

    # ── 7. Rule-based baseline for comparison ──
    pct_reduction = float(features_df.iloc[0]["pct_area_reduction"])
    rule_label = apply_rule_baseline(pct_reduction)

    # ── 8. Pull the top features from XGBoost importances ──
    # Useful for explainability — which features drove this prediction?
    try:
        importances = healing_model.feature_importances_
        importance_pairs = sorted(
            zip(FEATURE_COLS, importances),
            key=lambda x: x[1], reverse=True
        )
        key_factors = [name for name, _ in importance_pairs[:3]]
    except Exception:
        key_factors = []

    # ── 9. Save plots + CSVs ──
    saved_files = {}
    if save:
        HEALING_DIR.mkdir(parents=True, exist_ok=True)

        # Wrap in try/except because OneDrive sometimes locks files mid-write
        try:
            # Mask progression — visual of shrinking / stagnant wound
            mask_plot = str(HEALING_DIR / f"{case_id}_mask_progression.png")
            plot_mask_progression(mask_sequence, mask_plot, case_id)
            saved_files["mask_progression"] = mask_plot

            # Area trend line plot
            area_plot = str(HEALING_DIR / f"{case_id}_area_trend.png")
            plot_area_trend(longitudinal_df, area_plot, case_id)
            saved_files["area_trend"] = area_plot

            # Tissue trend line plot
            tissue_plot = str(HEALING_DIR / f"{case_id}_tissue_trend.png")
            plot_tissue_trend(longitudinal_df, tissue_plot, case_id)
            saved_files["tissue_trend"] = tissue_plot

            # CSV of the per-day data (useful for Excel exploration later)
            long_csv = str(HEALING_DIR / f"{case_id}_longitudinal.csv")
            longitudinal_df.to_csv(long_csv, index=False)
            saved_files["longitudinal_csv"] = long_csv

            # CSV of the features that went into XGBoost
            feat_csv = str(HEALING_DIR / f"{case_id}_features.csv")
            features_df.to_csv(feat_csv, index=False)
            saved_files["features_csv"] = feat_csv
        except PermissionError as e:
            print(f"  [healing] Warning: could not save some files (OneDrive lock?): {e}")
            print(f"  [healing] Results are still computed, just not saved to disk.")

    print(f"  [healing] prob={healing_prob:.3f} label={predicted_label} "
          f"rule={rule_label} mode={simulate_mode}")

    return HealingResult(
        case_id=case_id,
        healing_probability=round(healing_prob, 4),
        predicted_label=predicted_label,
        rule_label=rule_label,
        key_factors=key_factors,
        features=features_df.iloc[0].to_dict() if len(features_df) > 0 else None
    )
