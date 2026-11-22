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

    Accepts raw flattened pixel representations (B, 12288) or 4D image tensors (B, C, H, W).
    Equipped with an input adapter to project 39-D facial attribute metadata vectors
    when operating in hybrid tabular-image evaluation modes.
    """

    def __init__(
        self,
        input_dim: int = 12288,  # 3 channels * 64 height * 64 width
        hidden_dims: List[int] | None = None,
        num_classes: int = 1,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize the multi-layer perceptron architecture.

        Args:
            input_dim: Total flattened input feature dimension (default: 12288).
            hidden_dims: List of hidden layer dimensions (default: [256, 128, 64]).
            num_classes: Classification output dimension (default: 1 for binary smiling logit).
            dropout: Dropout probability applied after each hidden activation.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers: List[nn.Module] = []
        prev_dim = input_dim

        # Build fully connected blocks with Batch Normalization and ReLU
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final linear classification head producing unscaled logits
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim

        # Dual-mode feature adapter: projects 39 metadata attributes to expected input_dim
        self.feat_adapter = nn.Linear(39, input_dim) if input_dim != 39 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation pass.

        Args:
            x: Input tensor of shape (B, C, H, W), (B, input_dim), or (B, 39).

        Returns:
            torch.Tensor: Binary logits of shape (B, num_classes).
        """
        # Step 1: Flatten spatial 4D image tensors to 2D matrix (B, D)
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        # Step 2: Apply linear projection if input consists of 39 metadata attributes
        if x.shape[1] == 39 and self.input_dim != 39:
            x = self.feat_adapter(x)
        # Step 3: Propagate through multi-layer perceptron layers
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes calibrated sigmoid probabilities for binary classification.

        Args:
            x: Input tensor of shape (B, ...).

        Returns:
            torch.Tensor: Sigmoid probability in [0.0, 1.0].
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
