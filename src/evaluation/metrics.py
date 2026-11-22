"""
Comprehensive Evaluation Metrics for Binary Classification.
Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Log Loss, and Confusion Matrix.
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_comprehensive_metrics(y_true: Any, y_prob: Any) -> Dict[str, Any]:
    """
    Computes classification metrics from ground truth labels and predicted probabilities.
    Handles degenerate edge cases (all 0s or all 1s).
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32).flatten()
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).flatten()
    y_pred = (y_prob_arr >= 0.5).astype(np.int32)

    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    try:
        roc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        roc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    except ValueError:
        pr_auc = 0.5

    try:
        clipped = np.clip(y_prob_arr, 1e-7, 1.0 - 1e-7)
        loss = float(log_loss(y_true_arr, clipped))
    except ValueError:
        loss = 0.6931

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).tolist()

    return {
        "Accuracy": f"{acc * 100:.2f}%",
        "Accuracy_val": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc, 4),
        "PR-AUC": round(pr_auc, 4),
        "Test Loss": round(loss, 4),
        "Confusion Matrix": cm,
    }
