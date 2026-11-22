"""
End-to-End System Tests for CelebA Face AI.
Validates complete inference flow, fairness calculations, and identity leakage verification.
"""

import numpy as np
import pytest

from src.evaluation.fairness import compute_fairness_audit
from src.evaluation.leakage_check import verify_identity_disjoint_separation
from src.models.inference import ModelInferenceEngine


def test_end_to_end_inference_engine_prediction():
    """Verify ModelInferenceEngine correctly preprocesses image array and executes forward pass with TTA."""
    engine = ModelInferenceEngine()
    dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    res = engine.predict_attribute(dummy_image, architecture="Residual CNN", use_tta=True)

    assert res["source"] == "model_inference"
    assert res["architecture"] == "Residual CNN"
    assert res["use_tta"] is True
    assert 0.0 <= res["probability"] <= 1.0
    assert res["prediction"] in (0, 1)
    assert 0.5 <= res["confidence"] <= 1.0
    assert res["is_ground_truth_lookup"] is False


def test_end_to_end_fairness_audit():
    """Verify compute_fairness_audit computes disparate impact and demographic parity."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 1, 0, 0, 0])
    subgroup = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # Protected vs reference

    audit = compute_fairness_audit(y_true, y_pred, subgroup, subgroup_name="Male")

    assert audit["subgroup"] == "Male"
    assert "disparate_impact_ratio" in audit
    assert "demographic_parity_difference" in audit
    assert "equal_opportunity_difference" in audit
    assert isinstance(audit["meets_four_fifths_rule"], (bool, np.bool_))


def test_end_to_end_identity_leakage_verification():
    """Verify verify_identity_disjoint_separation correctly flags leaks and passes disjoint groups."""
    train_ids = np.array([101, 102, 103, 104])
    test_ids = np.array([201, 202, 203])

    check = verify_identity_disjoint_separation(train_ids, test_ids)
    assert check["is_strictly_disjoint"] is True
    assert check["identity_leakage_percentage"] == 0.0

    # Test leaky case
    leaky_test_ids = np.array([101, 202])
    leaky_check = verify_identity_disjoint_separation(train_ids, leaky_test_ids)
    assert leaky_check["is_strictly_disjoint"] is False
    assert leaky_check["overlapping_identities_count"] == 1
