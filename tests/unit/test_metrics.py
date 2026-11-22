"""
Unit tests for evaluation metrics calculation.
Verifies accuracy, precision, recall, f1, roc-auc, pr-auc, and confusion matrix.
"""

import numpy as np
import pytest

from src.models.train_and_evaluate import compute_comprehensive_metrics


def test_perfect_predictions_metrics():
    """Verify that perfect predictions yield 1.0 / 100% across all metrics."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.01, 0.05, 0.95, 0.99, 0.02, 0.98, 0.04, 0.96])

    metrics = compute_comprehensive_metrics(y_true, y_prob)

    assert metrics["Accuracy"] == "100.00%"
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
    assert metrics["F1-Score"] == 1.0
    assert metrics["ROC-AUC"] == 1.0
    assert metrics["PR-AUC"] == 1.0
    assert metrics["Confusion Matrix"] == [[4, 0], [0, 4]]
    assert metrics["Test Loss"] < 0.1


def test_inverted_predictions_metrics():
    """Verify metrics behaviour under completely inverted predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])

    metrics = compute_comprehensive_metrics(y_true, y_prob)

    assert metrics["Accuracy"] == "0.00%"
    assert metrics["Precision"] == 0.0
    assert metrics["Recall"] == 0.0
    assert metrics["F1-Score"] == 0.0
    assert metrics["ROC-AUC"] == 0.0
    assert metrics["Confusion Matrix"] == [[0, 2], [2, 0]]


def test_metrics_types_and_ranges():
    """Assert valid value boundaries for arbitrary continuous predictions."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=50)
    y_prob = np.random.uniform(0.0, 1.0, size=50)

    metrics = compute_comprehensive_metrics(y_true, y_prob)

    assert 0.0 <= metrics["Precision"] <= 1.0
    assert 0.0 <= metrics["Recall"] <= 1.0
    assert 0.0 <= metrics["F1-Score"] <= 1.0
    assert 0.0 <= metrics["ROC-AUC"] <= 1.0
    assert 0.0 <= metrics["PR-AUC"] <= 1.0
    assert metrics["Test Loss"] >= 0.0
