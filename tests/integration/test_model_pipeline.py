"""
Integration test for end-to-end model training, SLSQP ensembling, and evaluation pipeline.
Runs on a fast sample to verify complete algorithmic correctness.
"""

import numpy as np
import pytest

from src.models.train_and_evaluate import (
    compute_comprehensive_metrics,
    train_and_evaluate_all_models,
)


def test_end_to_end_pipeline_fast_execution():
    """Verify train_and_evaluate_all_models executes end-to-end without errors on sample subset."""
    results = train_and_evaluate_all_models(sample_limit=200)

    assert results["status"] == "success"
    assert "results" in results
    assert "weights" in results
    assert results["elapsed_seconds"] > 0

    model_list = results["results"]
    assert len(model_list) >= 5

    # Check key architectures exist in model results
    names = [m["Model Configuration"] for m in model_list]
    assert "Single-Fold XGBoost" in names
    assert "5-Fold Bagged XGBoost" in names
    assert "5-Fold Bagged Residual CNN" in names
    assert "5-Fold Ensemble + TTA" in names

    for m in model_list:
        assert "Accuracy" in m
        assert "F1-Score" in m
        assert "ROC-AUC" in m
        assert "PR-AUC" in m
        assert "Confusion Matrix" in m

    # Verify SLSQP ensemble weights sum to ~1.0
    weights = results["weights"]
    assert len(weights) == 3
    assert np.isclose(sum(weights), 1.0, atol=1e-2), f"Weights do not sum to 1.0: {sum(weights)}"
