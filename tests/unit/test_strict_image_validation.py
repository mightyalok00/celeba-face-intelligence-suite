"""
Unit tests for strict image validation, zero-tensor elimination, and smoke synthetic fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.train_and_evaluate import (
    _generate_synthetic_fixture,
    load_image_tensors,
)


def test_generate_synthetic_fixture_dimensions_and_distribution():
    """Verify synthetic fixture generates valid float32 RGB arrays in [0.0, 1.0]."""
    fixture = _generate_synthetic_fixture(target_size=(32, 32), seed_str="test_face_01")
    assert fixture.shape == (3, 32, 32)
    assert fixture.dtype == np.float32
    assert np.all(fixture >= 0.0)
    assert np.all(fixture <= 1.0)
    # Ensure fixture is non-zero (eliminates zero-tensor substitution)
    assert np.mean(fixture) > 0.2


def test_load_image_tensors_strict_validation_raises_when_missing(tmp_path):
    """
    Verify that in production mode (allow_synthetic=False), missing images strictly
    raise FileNotFoundError instead of silently substituting zero tensors.
    """
    df = pd.DataFrame({"image_id": ["nonexistent_999991.jpg", "nonexistent_999992.jpg"]})

    with pytest.raises(FileNotFoundError) as exc_info:
        load_image_tensors(df, target_size=(32, 32), allow_synthetic=False)

    assert "Missing 2 required face image files" in str(exc_info.value)


def test_load_image_tensors_smoke_mode_generates_synthetic_fixtures():
    """
    Verify that in smoke mode (allow_synthetic=True), missing images generate
    explicit non-zero synthetic fixtures with expected tensor shapes.
    """
    df = pd.DataFrame({"image_id": ["fake_001.jpg", "fake_002.jpg"]})
    tensors = load_image_tensors(df, target_size=(32, 32), allow_synthetic=True)
    assert tensors.shape == (2, 3, 32, 32)
    assert tensors.dtype == np.float32
    assert np.all(tensors >= 0.0)
    assert np.all(tensors <= 1.0)
    # Mean intensity must be positive (never silent zero-tensor)
    assert np.mean(tensors) > 0.1
