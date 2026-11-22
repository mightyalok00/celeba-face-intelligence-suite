"""
Unit tests for evaluation plot and figure generation routines.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.models.generate_plots import generate_all_evaluation_figures


def test_generate_all_evaluation_figures_with_mock_reports(tmp_path):
    """
    Verify generate_all_evaluation_figures parses benchmark CSVs and renders PNG figures
    without raising unhandled matplotlib exceptions.
    """
    mock_reports_dir = tmp_path / "reports"
    mock_reports_dir.mkdir(parents=True)
    (mock_reports_dir / "figures").mkdir()
    (mock_reports_dir / "confusion_matrices").mkdir()
    (mock_reports_dir / "roc_curves").mkdir()

    # Create dummy benchmark CSV with required metric columns
    df = pd.DataFrame(
        [
            {
                "Model Configuration": "Residual CNN",
                "Accuracy": "85.0%",
                "Precision": 0.82,
                "Recall": 0.88,
                "F1-Score": 0.85,
                "ROC-AUC": 0.91,
                "PR-AUC": 0.89,
                "Confusion Matrix": "[[45, 5], [6, 44]]",
            },
            {
                "Model Configuration": "Vision Transformer",
                "Accuracy": "82.0%",
                "Precision": 0.80,
                "Recall": 0.84,
                "F1-Score": 0.82,
                "ROC-AUC": 0.88,
                "PR-AUC": 0.86,
                "Confusion Matrix": "[[42, 8], [9, 41]]",
            },
        ]
    )
    csv_file = mock_reports_dir / "model_benchmark_detailed.csv"
    df.to_csv(csv_file, index=False)

    with patch("src.models.generate_plots.REPORTS_DIR", mock_reports_dir), \
         patch("src.models.generate_plots.FIGURES_DIR", mock_reports_dir / "figures"), \
         patch("src.models.generate_plots.CM_DIR", mock_reports_dir / "confusion_matrices"), \
         patch("src.models.generate_plots.ROC_DIR", mock_reports_dir / "roc_curves"):

        generate_all_evaluation_figures()

        # Verify output figures were generated
        fig_dir = mock_reports_dir / "figures"
        assert any(fig_dir.glob("*.png"))
