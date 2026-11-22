"""
Unit tests for data preprocessing, identity extraction, and train/test splitting.
Verifies strictly 0% identity leakage across splits.
"""

import numpy as np
import pytest
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def test_zero_identity_leakage_in_group_split(sample_feature_matrix):
    """Assert that train and test identities have completely disjoint sets."""
    X, y, groups = sample_feature_matrix

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    train_identities = set(groups[train_idx])
    test_identities = set(groups[test_idx])

    overlap = train_identities.intersection(test_identities)
    assert len(overlap) == 0, f"Identity leakage detected: {overlap}"
    assert len(train_identities) > 0
    assert len(test_identities) > 0


def test_stratified_group_kfold_separation(sample_feature_matrix):
    """Assert that all 5 cross-validation folds maintain strict identity disjointness."""
    X, y, groups = sample_feature_matrix

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        tr_groups = set(groups[tr_idx])
        val_groups = set(groups[val_idx])
        assert tr_groups.isdisjoint(val_groups), f"Fold {fold} has identity overlap!"


def test_feature_normalization():
    """Verify standard [-1, 1] CelebA binary attribute normalization to [0, 1]."""
    raw_celeba_vals = np.array([-1, 1, -1, 1, -1])
    normalized = (raw_celeba_vals + 1) / 2.0
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))
    assert np.array_equal(normalized, np.array([0.0, 1.0, 0.0, 1.0, 0.0]))
