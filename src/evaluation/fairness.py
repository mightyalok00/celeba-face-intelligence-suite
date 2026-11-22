"""
Fairness, Bias, and Subgroup Performance Audit for Facial Attribute Classifiers.
Calculates Demographic Parity, Disparate Impact Ratio, and Equalized Odds across demographic subgroups.
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, recall_score


def compute_fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_mask: np.ndarray,
    subgroup_name: str = "Protected Group",
) -> Dict[str, Any]:
    """
    Computes fairness metrics between protected (mask=1) and unprivileged/reference (mask=0) subgroups.
    - Demographic Parity: Difference in positive prediction rates P(Y_hat=1 | G=0) vs P(Y_hat=1 | G=1)
    - Disparate Impact Ratio: Ratio of positive prediction rates
    - Equal Opportunity Difference: Difference in True Positive Rates (Recall) between groups
    """
    mask_bool = np.asarray(subgroup_mask, dtype=bool)

    # Subgroup 1 (Protected / Group A)
    y_true_a = y_true[mask_bool]
    y_pred_a = y_pred[mask_bool]

    # Subgroup 0 (Reference / Group B)
    y_true_b = y_true[~mask_bool]
    y_pred_b = y_pred[~mask_bool]

    rate_a = float(np.mean(y_pred_a)) if len(y_pred_a) > 0 else 0.0
    rate_b = float(np.mean(y_pred_b)) if len(y_pred_b) > 0 else 0.0

    # Demographic Parity Difference: |rate_a - rate_b|
    demographic_parity_diff = abs(rate_a - rate_b)

    # Disparate Impact Ratio: min(rate_a/rate_b, rate_b/rate_a)
    if max(rate_a, rate_b) > 0:
        disparate_impact_ratio = min(rate_a, rate_b) / max(rate_a, rate_b)
    else:
        disparate_impact_ratio = 1.0

    # True Positive Rate (Equal Opportunity)
    tpr_a = float(recall_score(y_true_a, y_pred_a, zero_division=0)) if len(y_true_a) > 0 else 0.0
    tpr_b = float(recall_score(y_true_b, y_pred_b, zero_division=0)) if len(y_true_b) > 0 else 0.0
    equal_opportunity_diff = abs(tpr_a - tpr_b)

    return {
        "subgroup": subgroup_name,
        "group_a_size": len(y_true_a),
        "group_b_size": len(y_true_b),
        "group_a_positive_rate": round(rate_a, 4),
        "group_b_positive_rate": round(rate_b, 4),
        "demographic_parity_difference": round(demographic_parity_diff, 4),
        "disparate_impact_ratio": round(disparate_impact_ratio, 4),
        "equal_opportunity_difference": round(equal_opportunity_diff, 4),
        "meets_four_fifths_rule": disparate_impact_ratio >= 0.8,
    }
