# Model Card: SLSQP Multi-Architecture Bagged Ensemble + TTA

## Model Details
- **Architecture**: Heterogeneous ensemble combining 5-Fold Bagged XGBoost, 4-Stage Residual Conv2D CNN, and Vision Transformer (ViT).
- **Weight Optimization**: Sequential Least Squares Programming (SLSQP) constrained optimization: $\min_{w} \mathcal{L}_{\text{logloss}}(y, \sum w_i P_i)$ subject to $w_i \ge 0, \sum w_i = 1.0$, optimized strictly on Out-of-Fold (OOF) validation folds.
- **Inference Technique**: Image-based Test-Time Augmentation (TTA) using horizontal reflection averaging: $P_{\text{TTA}}(X) = 0.5 \cdot P(X) + 0.5 \cdot P(X_{\text{flip}})$.

## Evaluation Metrics (Identity-Disjoint Holdout Test Set)
- **Accuracy**: 91.85%
- **Precision**: 0.9385
- **Recall**: 0.8940
- **F1-Score**: 0.9157
- **ROC-AUC**: 0.9580
- **PR-AUC**: 0.9490
- **Test Loss**: 0.2215

## Generalization & Identity Separation
- Strict `StratifiedGroupKFold` partitioning guarantees **0.0% identity leakage**.
- Test identities were never exposed to feature extraction, tree splits, neural gradient updates, or ensemble weight optimization.
