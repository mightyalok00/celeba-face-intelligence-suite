"""
Unit tests for classical machine learning baselines:
Logistic Regression, Extra Trees, HistGradientBoosting, Single-Fold and 5-Fold Bagged XGBoost.
"""

import numpy as np
import pytest

from src.models.classical_baselines import (
    predict_bagged_models,
    train_5fold_bagged_xgboost,
    train_extra_trees,
    train_hist_gradient_boosting,
    train_logistic_regression,
    train_single_fold_xgboost,
)


@pytest.fixture
def synthetic_classification_data():
    """Generates synthetic binary classification data with non-trivial feature distributions."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    # Target linearly separable with additive noise
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.randn(100) * 0.3 > 0).astype(int)
    groups = np.repeat(np.arange(20), 5)  # 20 distinct identity groups
    return X, y, groups


def test_logistic_regression_training_and_prediction(synthetic_classification_data):
    X, y, _ = synthetic_classification_data
    clf = train_logistic_regression(X, y, C=1.0)
    assert hasattr(clf, "predict_proba")
    probs = clf.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_extra_trees_training_and_prediction(synthetic_classification_data):
    X, y, _ = synthetic_classification_data
    clf = train_extra_trees(X, y, n_estimators=10)
    assert hasattr(clf, "predict_proba")
    probs = clf.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_hist_gradient_boosting_training_and_prediction(synthetic_classification_data):
    X, y, _ = synthetic_classification_data
    clf = train_hist_gradient_boosting(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_single_fold_xgboost_training(synthetic_classification_data):
    X, y, _ = synthetic_classification_data
    clf = train_single_fold_xgboost(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_5fold_bagged_xgboost_and_aggregation(synthetic_classification_data):
    X, y, groups = synthetic_classification_data
    models = train_5fold_bagged_xgboost(X, y, groups, n_splits=3)
    assert len(models) == 3

    ensemble_probs = predict_bagged_models(models, X)
    assert len(ensemble_probs) == 100
    assert np.all((ensemble_probs >= 0.0) & (ensemble_probs <= 1.0))
