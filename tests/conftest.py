"""
Pytest configuration and shared test fixtures for CelebA Face AI & Intelligence System.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Clean sys.path: prioritize current project root and remove conflicting external paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if "ai-long-video-automation" not in p]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def clean_test_environment(monkeypatch):
    """Ensure standard environment variables for tests without leaking secrets."""
    monkeypatch.setenv("MYSQL_HOST", "127.0.0.1")
    monkeypatch.setenv("MYSQL_USER", "test_user")
    monkeypatch.setenv("MYSQL_PASSWORD", "test_pass_secure")
    monkeypatch.setenv("MYSQL_DATABASE", "test_celeba_db")
    monkeypatch.setenv("MYSQL_PORT", "3306")


@pytest.fixture
def sample_feature_matrix():
    """Generates a synthetic binary attribute feature matrix for 100 samples."""
    np.random.seed(42)
    # 40 binary CelebA attribute columns (-1 or 1)
    X = np.random.choice([-1.0, 1.0], size=(100, 40))
    y = ((X[:, 31] > 0) | (X[:, 20] > 0)).astype(int)  # synthetic smile label
    # 20 distinct identities with 5 images each
    groups = np.repeat(np.arange(1, 21), 5)
    return X, y, groups


@pytest.fixture
def mock_metadata_df():
    """Returns a synthetic pandas DataFrame resembling CelebA attributes."""
    data = {
        "image_id": [f"{i:06d}.jpg" for i in range(1, 21)],
        "person_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        "Smiling": [1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1],
        "Male": [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        "Young": [1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1]
    }
    return pd.DataFrame(data)
