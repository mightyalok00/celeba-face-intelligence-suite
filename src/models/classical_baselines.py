"""
Classical Machine Learning Baselines for Face Attribute Classification.
Preserves Logistic Regression, ExtraTrees, HistGradientBoosting,
Single-Fold XGBoost, and 5-Fold Bagged XGBoost as strong comparative baselines.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

XGBClassifier: Any = None
try:
    from xgboost import XGBClassifier as _RealXGBClassifier
    XGBClassifier = _RealXGBClassifier
except ImportError:
    XGBClassifier = None


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, C: float = 0.1
) -> LogisticRegression:
    """Trains L2-regularized Logistic Regression baseline."""
    clf = LogisticRegression(C=C, max_iter=200, random_state=42, solver="lbfgs")
    clf.fit(X_train, y_train)
    return clf


def train_extra_trees(
    X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100
) -> ExtraTreesClassifier:
    """Trains ExtraTrees (Extremely Randomized Trees) ensemble baseline."""
    clf = ExtraTreesClassifier(
        n_estimators=n_estimators, max_depth=12, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf


def train_hist_gradient_boosting(
    X_train: np.ndarray, y_train: np.ndarray
) -> HistGradientBoostingClassifier:
    """Trains Histogram-based Gradient Boosting classifier."""
    clf = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.08, max_depth=6, random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


def train_single_fold_xgboost(
    X_train: np.ndarray, y_train: np.ndarray
) -> Any:
    """Trains Single-Fold Gradient Boosted Decision Tree (XGBoost)."""
    if XGBClassifier is not None:
        clf = XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            reg_lambda=1.5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return clf
    return train_hist_gradient_boosting(X_train, y_train)


def train_5fold_bagged_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> List[Any]:
    """
    Trains a 5-Fold Stratified Group Bagged XGBoost ensemble.
    Guarantees that each base estimator is trained under disjoint group partitions.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models: List[Any] = []

    for fold_idx, (train_idx, _) in enumerate(sgkf.split(X_train, y_train, groups=groups)):
        X_f = X_train[train_idx]
        y_f = y_train[train_idx]
        if XGBClassifier is not None:
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                reg_lambda=1.5,
                subsample=0.85,
                random_state=42 + fold_idx,
                eval_metric="logloss",
                n_jobs=-1,
            )
            clf.fit(X_f, y_f)
            models.append(clf)
        else:
            clf = train_hist_gradient_boosting(X_f, y_f)
            models.append(clf)

    return models


def predict_bagged_models(models: List[Any], X_test: np.ndarray) -> np.ndarray:
    """Computes arithmetic mean probability across bagged models."""
    preds = [m.predict_proba(X_test)[:, 1] for m in models]
    return np.mean(preds, axis=0)
