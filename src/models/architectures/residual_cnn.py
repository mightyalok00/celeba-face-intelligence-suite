"""
Residual 2D Convolutional Neural Network (ResNet) for Face Attribute Classification.
Implements genuine Residual Blocks with skip connections, batch normalization,
and adaptive pooling for image-based classification on CelebA face tensors (B, 3, H, W).
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Standard 2D Residual Block with identity shortcut or 1x1 projection.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class ResidualConv2DCNN(nn.Module):
    """
    4-Stage Residual CNN designed for 64x64 CelebA facial images.
    Input shape: (Batch, 3, 64, 64) -> Output: (Batch, num_classes) logits.
    """

    def __init__(self, num_classes: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_channels = 32

        # Initial stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 4 Residual Stages: 32 -> 64 -> 128 -> 256
        self.stage1 = self._make_layer(32, stride=1)
        self.stage2 = self._make_layer(64, stride=2)
        self.stage3 = self._make_layer(128, stride=2)
        self.stage4 = self._make_layer(256, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
        # Adapter to reshape 39 metadata attribute features to image tensor (B, 3, 64, 64)
        self.feat_adapter = nn.Sequential(
            nn.Linear(39, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3 * 64 * 64),
        )

    def _make_layer(self, out_channels: int, stride: int) -> nn.Sequential:
        layer = ResidualBlock(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return nn.Sequential(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes residual CNN forward pass with dual-mode support for images and metadata.

        Args:
            x: Tensor of shape (B, 3, 64, 64) for images, or (B, 39) for attribute vectors.

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        # Step 1: If input is tabular metadata (B, 39), project and reshape to (B, 3, 64, 64)
        if x.dim() == 2 and x.shape[1] == 39:
            x = self.feat_adapter(x).view(-1, 3, 64, 64)

        # Step 2: Stem convolution (3 -> 32 channels)
        out = self.stem(x)

        # Step 3: Four hierarchical residual stages capturing fine-to-coarse features
        out = self.stage1(out)  # 32 channels, 64x64 spatial resolution
        out = self.stage2(out)  # 64 channels, 32x32 spatial resolution
        out = self.stage3(out)  # 128 channels, 16x16 spatial resolution
        out = self.stage4(out)  # 256 channels, 8x8 spatial resolution

        # Step 4: Global spatial pooling, regularization, and classification projection
        out = self.pool(out)  # Condense 8x8 spatial grid into (B, 256, 1, 1)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        return self.fc(out)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sigmoid probabilities for binary classification."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
