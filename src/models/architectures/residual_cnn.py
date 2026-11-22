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

    def _make_layer(self, out_channels: int, stride: int) -> nn.Sequential:
        layer = ResidualBlock(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return nn.Sequential(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        return self.fc(out)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sigmoid probabilities for binary classification."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
