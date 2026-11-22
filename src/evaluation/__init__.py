"""
Evaluation & Audit Subpackage for CelebA Face AI.
Includes metrics calculation, identity leakage checks, and fairness analysis.
"""

from src.evaluation.fairness import compute_fairness_audit
from src.evaluation.leakage_check import verify_identity_disjoint_separation
from src.evaluation.metrics import compute_comprehensive_metrics

__all__ = [
    "compute_comprehensive_metrics",
    "verify_identity_disjoint_separation",
    "compute_fairness_audit",
]
