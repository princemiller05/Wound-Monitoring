"""
Simple rule-based healing baseline.
Used as a sanity check alongside the XGBoost prediction.

The clinical rule-of-thumb: a DFU that reduces by >= 30% in 4 weeks
is on track to heal. Under 30% usually means something's wrong.
"""

from pipeline.config import RULE_REDUCTION_THR


def apply_rule_baseline(pct_area_reduction: float,
                        threshold=RULE_REDUCTION_THR) -> str:
    """
    Args:
        pct_area_reduction: fraction, so 0.35 = 35% smaller at end vs start
        threshold:          cutoff (default 0.30 = 30%)

    Returns:
        "healing" if reduction meets/exceeds threshold, else "non_healing"
    """
    return "healing" if pct_area_reduction >= threshold else "non_healing"
