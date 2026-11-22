# CelebA Face AI Model Checkpoints & Architecture Registry

This directory contains configuration specifications, serialization schemas, and architecture metadata for the 8 evaluated models.

## Architectural Regimes

| Model | Type | Input Shape | Key Hyperparameters | Test Accuracy | Test F1 |
| :---- | :--- | :---------- | :------------------ | :------------ | :------ |
| **ANN** | Multi-Layer Perceptron | `(12288,)` | Units: [256, 128, 64], Adam(1e-3), Dropout 0.3 | 80.00% | 0.7857 |
| **Spatial BiLSTM** | Recurrent Sequence | `(64, 192)` | BiLSTM(64) -> BiLSTM(32), Adam(5e-4) | 83.33% | 0.8148 |
| **Residual CNN** | 4-Stage ResNet | `(64, 64, 3)` | In-Model Aug, Filters: [32, 64, 128, 256], Adam(3e-4) | 83.33% | 0.8276 |
| **Vision Transformer** | Patch Self-Attention | `(64, 64, 3)` | Patch 16, Dim 64, 4 Heads, 2 Layers, Adam(1e-4) | 83.33% | 0.8276 |
| **Single-Fold XGBoost** | Gradient Tree Boosting | `(12288,)` | Trees: 120, Depth: 5, LR: 0.08, L2: 1.5 | 83.33% | 0.8148 |
| **5-Fold Bagged XGBoost** | Stratified Group Bagging | `(12288,)` | 5 Fold Averaged Probability | 83.33% | 0.8276 |
| **5-Fold Ensemble** | Probability Blending | Multi-modal | SLSQP Weights: [0.3457, 0.3959, 0.2584] | 83.33% | 0.8276 |
| **Ensemble + TTA** | Test-Time Augmentation | Augmented | Reflection Averaging: `0.5 * P(X) + 0.5 * P(X_flip)` | 83.33% | 0.8276 |

## Checkpoint Loading Specification

```python
import json
from pathlib import Path

# Load serialized model architecture metadata
registry_path = Path("models/model_registry.json")
with open(registry_path, "r", encoding="utf-8") as f:
    registry = json.load(f)

print(f"Loaded {len(registry['architectures'])} verified model checkpoints.")
```
