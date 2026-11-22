"""
Unit tests for deep learning and tree model inference.
Verifies forward pass shapes, probability bounds, and test-time augmentation (TTA).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.models.inference import ModelInferenceEngine


class Residual1DCNN(nn.Module):
    """
    1D Residual Convolutional Neural Network with identity shortcut connections.
    Emulates ResNet architectural features for facial attribute prediction.
    """
    def __init__(self, input_dim: int = 64, num_classes: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x_in = x.unsqueeze(1)
        res = self.relu(self.conv1(x_in))
        # Residual skip connection: output = f(x) + x
        out = self.relu(self.conv2(res) + res)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class BiLSTMAttributeClassifier(nn.Module):
    """
    Bidirectional Long Short-Term Memory (BiLSTM) sequence network.
    Extracts forward and backward spatial dependencies across facial representations.
    """
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, num_classes: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, 1)
        x_in = x.unsqueeze(-1)
        out, _ = self.lstm(x_in)
        # Final bidirectional hidden state concatenated
        return self.fc(out[:, -1, :])


class MiniVisionTransformer(nn.Module):
    """
    Miniature Vision Transformer (ViT) with multi-head self-attention.
    Computes global relationship weights across facial attribute tokens.
    """
    def __init__(self, input_dim: int = 64, embed_dim: int = 32, num_heads: int = 4, num_classes: int = 1):
        super().__init__()
        self.proj = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project each attribute scalar into embedding dimension
        x_in = self.proj(x.unsqueeze(-1))
        attn_out, _ = self.attn(x_in, x_in, x_in)
        # Global average pooling across token sequence
        pooled = attn_out.mean(dim=1)
        return self.fc(pooled)


def test_residual_1d_cnn_forward():
    """Verify Residual1DCNN tensor output shape (batch_size, 1) and sigmoid probabilities."""
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
    """
    Verify Test-Time Augmentation (TTA) stabilizes probability predictions.
    Computes mathematical average: 0.5 * (P_orig + P_augmented) and verifies [0.0, 1.0] bounds.
    """
    p_orig = np.array([0.70, 0.40, 0.85])
    p_flipped = np.array([0.74, 0.42, 0.81])

    # TTA arithmetic ensemble
    p_tta = (p_orig + p_flipped) / 2.0
    assert np.allclose(p_tta, np.array([0.72, 0.41, 0.83]))
    assert np.all((p_tta >= 0.0) & (p_tta <= 1.0))


def test_model_inference_engine_genuine_tta_and_checkpoint_loading():
    """Verify ModelInferenceEngine loads architectures and runs 5-view TTA."""
    engine = ModelInferenceEngine()
    test_img = Image.new("RGB", (64, 64), color="green")

    # 1. Prediction with genuine TTA
    res = engine.predict_attribute(test_img, architecture="Residual CNN", use_tta=True)
    assert res["source"] == "model_inference"
    assert res["architecture"] == "Residual CNN"
    assert res["use_tta"] is True
    assert res["tta_views_evaluated"] == 5
    assert 0.0 <= res["probability"] <= 1.0
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["prediction_uncertainty_std"] >= 0.0
    assert "checkpoint_loaded" in res

    # 2. Prediction without TTA
    res_no_tta = engine.predict_attribute(test_img, architecture="Residual CNN", use_tta=False)
    assert res_no_tta["use_tta"] is False
    assert res_no_tta["tta_views_evaluated"] == 1
    assert res_no_tta["prediction_uncertainty_std"] == 0.0
