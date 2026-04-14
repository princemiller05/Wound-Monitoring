"""
Load Varsha's XGBoost healing model.

This was trained on longitudinal features (how wound area changes over time)
and outputs a probability between 0 (non-healing) and 1 (healing).
"""

import xgboost as xgb

from pipeline.config import HEALING_MODEL_PATH


def load_healing_model():
    """
    Load XGBoost from JSON. We use save_model/load_model (not pickle)
    because it's the portable XGBoost way — works across versions.
    """
    model = xgb.XGBClassifier()
    model.load_model(str(HEALING_MODEL_PATH))

    print("[healing] XGBoost model loaded")
    return model
