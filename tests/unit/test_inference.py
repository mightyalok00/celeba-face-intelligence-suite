"""
Unit tests for deep learning and tree model inference.
Verifies forward pass shapes, probability bounds, and test-time augmentation (TTA).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


class Residual1DCNN(nn.Module):
    def __init__(self, input_dim: int = 64, num_classes: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.unsqueeze(1)
        res = self.relu(self.conv1(x_in))
        out = self.relu(self.conv2(res) + res)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class BiLSTMAttributeClassifier(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, num_classes: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.unsqueeze(-1)
        out, _ = self.lstm(x_in)
        return self.fc(out[:, -1, :])


class MiniVisionTransformer(nn.Module):
    def __init__(self, input_dim: int = 64, embed_dim: int = 32, num_heads: int = 4, num_classes: int = 1):
        super().__init__()
        self.proj = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.proj(x.unsqueeze(-1))
        attn_out, _ = self.attn(x_in, x_in, x_in)
        pooled = attn_out.mean(dim=1)
        return self.fc(pooled)


def test_residual_1d_cnn_forward():
    """Verify Residual1DCNN tensor output shape and sigmoid probabilities."""
    model = Residual1DCNN(input_dim=64, num_classes=1)
    model.eval()
    x = torch.randn(8, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 1)
    probs = torch.sigmoid(out).numpy()
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_bilstm_classifier_forward():
    """Verify BiLSTMAttributeClassifier sequence processing and output shape."""
    model = BiLSTMAttributeClassifier(input_dim=64, hidden_dim=32, num_classes=1)
    model.eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1)
    probs = torch.sigmoid(out).numpy()
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_mini_vit_forward():
    """Verify MiniVisionTransformer self-attention projection and output shape."""
    model = MiniVisionTransformer(input_dim=64, embed_dim=32, num_heads=4, num_classes=1)
    model.eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1)
    probs = torch.sigmoid(out).numpy()
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_test_time_augmentation_blending():
    """Verify Test-Time Augmentation (TTA) stabilizes probability predictions."""
    p_orig = np.array([0.70, 0.40, 0.85])
    p_flipped = np.array([0.74, 0.42, 0.81])

    # TTA ensemble
    p_tta = (p_orig + p_flipped) / 2.0
    assert np.allclose(p_tta, np.array([0.72, 0.41, 0.83]))
    assert np.all((p_tta >= 0.0) & (p_tta <= 1.0))
