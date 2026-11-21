# CelebA Face AI Model Checkpoints & Architecture Registry

This directory contains configuration specifications, serialization schemas, and architecture metadata for the 8 evaluated models.

## Architectural Regimes

| Model | Type | Input Shape | Key Hyperparameters | Test Accuracy | Test F1 |
| --- | --- | --- | --- | --- | --- |
| **ANN** | Multi-Layer Perceptron | `(12288,)` | Units: [256, 128, 64], Adam(1e-3), Dropout 0.3 | 77.87% | 0.7676 |
| **Spatial BiLSTM** | Recurrent Sequence | `(64, 192)` | BiLSTM(64) -> BiLSTM(32), Adam(5e-4) | 86.10% | 0.8507 |
| **Residual CNN** | 4-Stage ResNet | `(64, 64, 3)` | In-Model Aug, Filters: [32, 64, 128, 256], Adam(3e-4) | 84.60% | 0.8326 |
| **Vision Transformer** | Patch Self-Attention | `(64, 64, 3)` | Patch 16, Dim 64, 4 Heads, 2 Layers, Adam(1e-4) | 90.25% | 0.8985 |
| **Single-Fold XGBoost** | Gradient Tree Boosting | `(12288,)` | Trees: 120, Depth: 5, LR: 0.08, L2: 1.5 | 89.15% | 0.8861 |
| **5-Fold Bagged XGBoost** | Stratified Group Bagging | `(12288,)` | 5 Fold Averaged Probability | 89.87% | 0.8945 |
| **5-Fold Ensemble** | Probability Blending | Multi-modal | SLSQP Weights: [0.4285, 0.1840, 0.3875] | 91.40% | 0.9100 |
| **Ensemble + TTA** | Test-Time Augmentation | Augmented | Reflection Averaging: `0.5 * P(X) + 0.5 * P(X_flip)` | 91.85% | 0.9157 |
