"""
Artificial Neural Network (ANN / Deep MLP) for Face Attribute Classification.
Implements deep fully-connected layers with BatchNorm, Dropout, and ReLU activations
accepting flattened pixel tensors (B, 12288) or 4D image tensors (B, 3, 64, 64).
"""

from typing import List

import torch
import torch.nn as nn


class ANNClassifier(nn.Module):
    """
    Deep Multi-Layer Perceptron (ANN) with configurable hidden dimensions.
    """
    def __init__(
        self,
        input_dim: int = 12288,  # 3 * 64 * 64
        hidden_dims: List[int] | None = None,
        num_classes: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        # Adapter to project 39 attribute metadata features if input_dim differs
        self.feat_adapter = nn.Linear(39, input_dim) if input_dim != 39 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        if x.shape[1] == 39 and self.input_dim != 39:
            x = self.feat_adapter(x)
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sigmoid probabilities for binary classification."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
