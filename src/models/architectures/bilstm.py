"""
Spatial BiDirectional LSTM (RNN) Architecture for Face Attribute Classification.
Treats image rows sequentially (e.g. 64 rows of 192 features for 64x64x3 faces)
to model vertical spatial dependencies and symmetry across facial landmarks.
"""

import torch
import torch.nn as nn


class SpatialBiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM operating on spatial image row slices.
    Input shape: (Batch, 3, 64, 64) -> Transposed to (Batch, 64, 192) -> BiLSTM -> Linear.
    """

    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.feature_dim = img_size * in_channels  # e.g., 64 * 3 = 192

        self.bilstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )
        # Adapter to reshape 39 metadata attribute features to image tensor (B, 3, 64, 64)
        self.feat_adapter = nn.Sequential(
            nn.Linear(39, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3 * 64 * 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes bidirectional sequence modeling across spatial image row transitions.

        Args:
            x: Tensor of shape (B, 3, 64, 64) for images, or (B, 39) for attribute vectors.

        Returns:
            torch.Tensor: Binary logits of shape (B, num_classes).
        """
        # Step 1: If input is tabular metadata (B, 39), project and reshape to (B, 3, 64, 64)
        if x.dim() == 2 and x.shape[1] == 39:
            x = self.feat_adapter(x).view(-1, 3, 64, 64)

        # Step 2: Slice horizontal rows into sequential time steps: (B, H, W * C) -> (B, 64, 192)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B, H, W * C)
        elif x.dim() == 2:
            # Flattened input fallback
            B = x.shape[0]
            x = x.view(B, self.img_size, -1)

        # Step 3: BiLSTM recurrent pass capturing both top-to-bottom and bottom-to-top dependencies
        lstm_out, _ = self.bilstm(x)

        # Step 4: Temporal average pooling across the 64 spatial time steps
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sigmoid probabilities for binary classification."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
