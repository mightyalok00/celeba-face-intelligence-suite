"""
Ensemble Learning & Test-Time Augmentation (TTA) Engine.
Implements out-of-fold probability blending with SLSQP optimization
and real horizontal image reflection Test-Time Augmentation (TTA).
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize  # type: ignore
from sklearn.metrics import log_loss


def optimize_ensemble_weights(
    oof_predictions: List[np.ndarray], y_true: np.ndarray
) -> np.ndarray:
    """
    Finds optimal convex combination weights w_i >= 0, sum(w_i) = 1.0
    to minimize binary cross-entropy (log loss) on Out-Of-Fold (OOF) validation data.
    Never optimizes weights on final test data.
    """
    K = len(oof_predictions)
    init_weights = np.ones(K) / K

    def objective(w: np.ndarray) -> float:
        w_norm = w / (np.sum(w) + 1e-12)
        blend = sum(w_norm[i] * oof_predictions[i] for i in range(K))
        blend = np.clip(blend, 1e-7, 1.0 - 1e-7)
        return float(log_loss(y_true, blend))

    bounds = [(0.0, 1.0) for _ in range(K)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )

    if res.success:
        weights = res.x / np.sum(res.x)
    else:
        weights = init_weights
    return weights


def blend_predictions(
    predictions: List[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    """Blends multiple model probability vectors using optimal weights."""
    assert len(predictions) == len(weights)
    blend = sum(weights[i] * predictions[i] for i in range(len(weights)))
    return np.clip(blend, 0.0, 1.0)


def apply_test_time_augmentation(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    images: np.ndarray,
) -> np.ndarray:
    """
    Genuine Image-Based Test-Time Augmentation (TTA).
    Evaluates original image and horizontally flipped image:
    P_TTA(x) = 0.5 * P(x) + 0.5 * P(flip_horizontal(x))
    """
    # 1. Prediction on original inputs
    p_orig = predict_fn(images)

    # 2. Horizontal flip
    if images.ndim == 4:
        # Shape: (B, C, H, W) or (B, H, W, C)
        if images.shape[1] in (1, 3):
            # (B, C, H, W) -> flip along width (axis 3)
            images_flipped = np.flip(images, axis=3).copy()
        else:
            # (B, H, W, C) -> flip along width (axis 2)
            images_flipped = np.flip(images, axis=2).copy()
    elif images.ndim == 2:
        # If flattened vector of shape (B, 3*64*64), reshape, flip, flatten
        B = images.shape[0]
        reshaped = images.reshape(B, 64, 64, -1)
        images_flipped = np.flip(reshaped, axis=2).reshape(B, -1).copy()
    else:
        images_flipped = images

    # 3. Prediction on flipped inputs
    p_flip = predict_fn(images_flipped)

    # 4. Arithmetic mean blend
    return 0.5 * p_orig + 0.5 * p_flip
