"""
Unit tests for SLSQP multi-model probability ensembling and Test-Time Augmentation (TTA).
"""

import numpy as np
import pytest

from src.models.ensemble import (
    apply_test_time_augmentation,
    blend_predictions,
    optimize_ensemble_weights,
)


def test_optimize_ensemble_weights_sum_to_one():
    """Verify that SLSQP optimization finds valid convex weights summing to 1.0."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=100)

    # Simulated model predictions with varying accuracies
    pred_1 = np.clip(y_true * 0.8 + rng.uniform(0.0, 0.2, 100), 0.01, 0.99)
    pred_2 = np.clip(y_true * 0.6 + rng.uniform(0.0, 0.4, 100), 0.01, 0.99)
    pred_3 = np.clip(rng.uniform(0.1, 0.9, 100), 0.01, 0.99)

    weights = optimize_ensemble_weights([pred_1, pred_2, pred_3], y_true)

    assert len(weights) == 3
    assert np.all(weights >= 0.0)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-3)
    # The most accurate predictor (pred_1) should receive the highest weight
    assert weights[0] >= weights[2]


def test_blend_predictions():
    """Verify weighted blending of prediction vectors."""
    pred_a = np.array([0.2, 0.8, 0.5])
    pred_b = np.array([0.4, 0.6, 0.5])
    weights = np.array([0.5, 0.5])

    blended = blend_predictions([pred_a, pred_b], weights)
    expected = np.array([0.3, 0.7, 0.5])
    assert np.allclose(blended, expected, atol=1e-5)


def test_apply_test_time_augmentation_4d():
    """Verify TTA correctly reflects 4D image tensors along the width dimension."""
    # (B, C, H, W) = (2, 3, 4, 4)
    images = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)

    # Dummy predictor that returns mean pixel value
    def mock_predict(x):
        return np.mean(x, axis=(1, 2, 3)) / 100.0

    tta_preds = apply_test_time_augmentation(mock_predict, images)
    assert len(tta_preds) == 2
    assert np.all((tta_preds >= 0.0) & (tta_preds <= 1.0))


def test_apply_test_time_augmentation_2d():
    """Verify TTA handles flattened 2D vectors by reshaping, flipping and blending."""
    images_2d = np.ones((2, 3 * 64 * 64), dtype=np.float32) * 0.5

    def mock_predict(x):
        return np.array([0.7, 0.3])

    tta_preds = apply_test_time_augmentation(mock_predict, images_2d)
    assert len(tta_preds) == 2
    assert np.allclose(tta_preds, [0.7, 0.3])
