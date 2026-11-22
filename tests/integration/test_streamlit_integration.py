"""
Integration tests for Streamlit application components and MCP tool interfaces.
"""

from unittest.mock import patch

import pytest

from app.app import (
    execute_mcp_face_identification,
    load_benchmark_metrics,
    load_gallery_data,
    query_mysql,
)


def test_load_gallery_data_returns_list():
    """
    Verify load_gallery_data loads the pre-indexed gallery JSON structure
    used for the Google Photos interface and curated offline fallbacks.
    """
    gallery = load_gallery_data()
    assert isinstance(gallery, list)
    if len(gallery) > 0:
        first = gallery[0]
        assert "filename" in first
        assert "person_name" in first
        assert "ground_truth_id" in first


def test_load_benchmark_metrics_structure():
    """
    Verify load_benchmark_metrics loads real evaluated CSV benchmark results,
    confirming consistent column naming and inclusion of all evaluated architectures.
    """
    df = load_benchmark_metrics()
    assert df is not None
    assert "Model Configuration" in df.columns
    assert "Accuracy" in df.columns
    assert "F1-Score" in df.columns
    assert len(df) >= 5


def test_mcp_face_identification_fallback_behavior():
    """
    Verify that when MySQL is offline, execute_mcp_face_identification
    transparently falls back to ground-truth reference data and labels it clearly as offline fallback.
    """
    with patch("app.app.query_mysql", return_value=(None, "Database offline")):
        req, resp, structured = execute_mcp_face_identification("001553.jpg")

        # 1. Verify JSON-RPC 2.0 tool invocation structure
        assert req["method"] == "tools/call"
        assert req["params"]["name"] == "mysql_identify_face"

        # 2. Verify fallback attribution and data integrity
        if not isinstance(structured, dict):
            raise AssertionError("execute_mcp_face_identification should return a dict")
        assert structured.get("is_fallback") is True
        assert "Database Offline" in str(structured.get("source", ""))
        assert structured["image_id"] == "001553.jpg"


def test_mcp_face_identification_live_db_match():
    """
    Verify structured parsing when MySQL returns a valid match from the celeba_db database,
    confirming live provenance and attribute extraction.
    """
    import pandas as pd
    mock_df = pd.DataFrame([{
        "image_id": "000001.jpg",
        "file_name": "000001.jpg",
        "detected_person_name": "Celebrity Alpha",
        "identity_cluster_id": 101,
        "detected_gender": "Female",
        "detected_age_group": "Young",
        "detected_expression": "Smiling",
        "wearing_eyeglasses": "No",
        "hair_type": "Brown",
        "total_photos_of_person": 25,
        "file_path": "img_align_celeba/000001.jpg"
    }])

    with patch("app.app.query_mysql", return_value=(mock_df, None)):
        req, resp, structured = execute_mcp_face_identification("000001.jpg")
        if not isinstance(structured, dict):
            raise AssertionError("execute_mcp_face_identification should return a dict")
        assert structured.get("is_fallback") is False
        assert structured["person_name"] == "Celebrity Alpha"
        assert structured["identity_cluster_id"] == 101
