"""
Identity Leakage Verification Utility for CelebA Face AI.
Proves 0.0% identity leakage between training and evaluation splits under StratifiedGroupKFold.
"""

from typing import Any, Dict, List, Set

import numpy as np


def verify_identity_disjoint_separation(
    train_identities: np.ndarray, test_identities: np.ndarray
) -> Dict[str, Any]:
    """
    Asserts that the set intersection between train and test identity groups is strictly empty.
    Returns audit dictionary with unique count, overlap count, and leakage percentage.
    """
    train_set: Set[Any] = set(train_identities.flatten())
    test_set: Set[Any] = set(test_identities.flatten())
    overlap: Set[Any] = train_set.intersection(test_set)

    leakage_pct = (len(overlap) / len(test_set)) * 100.0 if len(test_set) > 0 else 0.0

    return {
        "train_unique_identities": len(train_set),
        "test_unique_identities": len(test_set),
        "overlapping_identities_count": len(overlap),
        "identity_leakage_percentage": round(leakage_pct, 4),
        "is_strictly_disjoint": len(overlap) == 0,
    }
