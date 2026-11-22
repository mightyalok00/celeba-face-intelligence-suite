# Model Card: SLSQP Multi-Architecture Bagged Ensemble + TTA

## Model Details

- **Architecture**: Heterogeneous ensemble combining 5-Fold Bagged XGBoost, 4-Stage Residual Conv2D CNN, and Vision Transformer (ViT).
- **Weight Optimization**: Sequential Least Squares Programming (SLSQP) constrained optimization: $\min_{w} \mathcal{L}_{\text{logloss}}(y, \sum w_i P_i)$ subject to $w_i \ge 0, \sum w_i = 1.0$, optimized strictly on Out-of-Fold (OOF) validation folds.
- **Inference Technique**: Image-based Test-Time Augmentation (TTA) using multi-view reflection and photometric averaging.

## Evaluation Metrics (Identity-Disjoint Holdout Test Set)

- **Accuracy**: 82.24%
- **Precision**: 0.8371
- **Recall**: 0.8043
- **F1-Score**: 0.8204
- **Test Loss**: 0.4215

## Generalization & Identity Separation

- Strict `StratifiedGroupKFold` partitioning guarantees **0.0% identity leakage**.
- Test identities were never exposed to feature extraction, tree splits, neural gradient updates, or ensemble weight optimization.
