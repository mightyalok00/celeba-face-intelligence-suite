"""
Unit tests for deep learning architectures and model serialization.
Verifies forward pass shapes, probability bounds, and state_dict checkpoint serialization.
"""

import io

import pytest
import torch

from src.models.architectures.ann import ANNClassifier
from src.models.architectures.bilstm import SpatialBiLSTMClassifier
from src.models.architectures.residual_cnn import ResidualConv2DCNN
from src.models.architectures.vision_transformer import VisionTransformerClassifier


def test_residual_conv2d_cnn_forward_and_serialization():
    """Verify Residual 2D-CNN forward pass and checkpoint serialization."""
    model = ResidualConv2DCNN(num_classes=1).eval()
    x = torch.randn(2, 3, 64, 64)

    # 1. Forward pass logits and probabilities
    with torch.no_grad():
        logits = model(x)
        assert logits.shape == (2, 1)

        probs = model.predict_proba(x)
        assert probs.shape == (2, 1)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

        # 2. Checkpoint serialization roundtrip
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        new_model = ResidualConv2DCNN(num_classes=1).eval()
        new_model.load_state_dict(torch.load(buffer, weights_only=True))
        new_logits = new_model(x)
        assert torch.allclose(logits, new_logits, atol=1e-5)


def test_vision_transformer_forward_and_attention():
    """Verify Vision Transformer forward pass on image patches."""
    vit = VisionTransformerClassifier(img_size=64, patch_size=8, num_classes=1)
    x = torch.randn(2, 3, 64, 64)

    logits = vit(x)
    assert logits.shape == (2, 1)

    probs = vit.predict_proba(x)
    assert probs.shape == (2, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_spatial_bilstm_forward():
    """Verify Spatial BiLSTM on 2D image row slices."""
    bilstm = SpatialBiLSTMClassifier(img_size=64, in_channels=3, hidden_dim=32, num_classes=1)
    x = torch.randn(2, 3, 64, 64)

    logits = bilstm(x)
    assert logits.shape == (2, 1)

    probs = bilstm.predict_proba(x)
    assert probs.shape == (2, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_ann_dense_classifier_forward():
    """Verify ANN / Multi-Layer Perceptron forward pass."""
    ann = ANNClassifier(input_dim=12288, hidden_dims=[128, 64], num_classes=1)
    x = torch.randn(2, 12288)

    logits = ann(x)
    assert logits.shape == (2, 1)

    probs = ann.predict_proba(x)
    assert probs.shape == (2, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
