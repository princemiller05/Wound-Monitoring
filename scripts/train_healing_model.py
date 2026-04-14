"""
Train and save the XGBoost healing prediction model.
================================================
Replicates Varsha's training logic from her notebook.

Run once:
    python scripts/train_healing_model.py

This generates synthetic wound data (healing + non-healing cases),
extracts features, trains XGBoost, and saves the model to
models/healing/xgb_healing.json

Takes ~2-3 seconds. No GPU needed.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

from pipeline.config import HEALING_MODEL_PATH, FEATURE_COLS


def generate_synthetic_training_data(n_cases=100, seed=42):
    """
    Generate synthetic wound progression data for training.
    Replicates the approach from Varsha's notebook but at larger scale.
    """
    rng = np.random.default_rng(seed)
    all_features = []

    for i in range(n_cases):
        # Random initial wound area
        initial_area = rng.integers(5000, 500000)

        if i % 2 == 0:
            # HEALING case: area decreases over time
            reduction = rng.uniform(0.20, 0.80)
            final_area = int(initial_area * (1 - reduction))
            areas = np.linspace(initial_area, final_area, 4).astype(int)
            # Add some noise
            areas = areas + rng.integers(-500, 500, size=4)
            areas = np.maximum(areas, 100)
            label = 1  # healing
        else:
            # NON-HEALING case: area stays same or increases
            reduction = rng.uniform(-0.10, 0.15)
            final_area = int(initial_area * (1 - reduction))
            areas = np.array([
                initial_area,
                int(initial_area * (1 - rng.uniform(-0.05, 0.08))),
                int(initial_area * (1 - rng.uniform(-0.08, 0.10))),
                final_area
            ])
            areas = np.maximum(areas, 100)
            label = 0  # non-healing

        actual_reduction = (areas[0] - areas[-1]) / areas[0] if areas[0] > 0 else 0

        all_features.append({
            "wound_id": f"W{i}",
            "initial_area": int(areas[0]),
            "final_area": int(areas[-1]),
            "pct_area_reduction": round(actual_reduction, 6),
            "mean_area": round(np.mean(areas), 2),
            "std_area": round(np.std(areas), 2),
            "label": label,
        })

    return pd.DataFrame(all_features)


def main():
    print("=" * 50)
    print("Training XGBoost Healing Model")
    print("=" * 50)

    # 1. Generate training data
    print("\n[1/4] Generating synthetic training data...")
    df = generate_synthetic_training_data(n_cases=200, seed=42)
    print(f"  Generated {len(df)} cases: "
          f"{(df['label']==1).sum()} healing, {(df['label']==0).sum()} non-healing")

    # 2. Split
    print("\n[2/4] Splitting data...")
    X = df[FEATURE_COLS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Train
    print("\n[3/4] Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {acc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['non_healing', 'healing'])}")

    # 4. Save
    print("[4/4] Saving model...")
    HEALING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(HEALING_MODEL_PATH))
    print(f"  Saved to: {HEALING_MODEL_PATH}")

    # Show feature importance
    print("\nFeature importance:")
    for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_),
                            key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.3f}")

    print("\nDone! Model ready for pipeline use.")


if __name__ == "__main__":
    main()
