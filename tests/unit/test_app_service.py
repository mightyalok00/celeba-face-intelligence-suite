"""
Unit tests for shared application service layer (app/service.py).
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.service import (
    execute_mcp_face_identification,
    load_benchmark_metrics,
    load_gallery_data,
    query_mysql,
)


def test_load_gallery_data():
    """Verify load_gallery_data returns parsed JSON list."""
    data = load_gallery_data()
    assert isinstance(data, list)


def test_load_benchmark_metrics():
    """Verify load_benchmark_metrics loads a DataFrame or returns None cleanly."""
    df = load_benchmark_metrics()
    if df is not None:
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


def test_query_mysql_success():
    """Verify query_mysql delegates to execute_parameterized_query and formats DataFrame."""
    with patch("src.database.connection.execute_parameterized_query") as mock_exec:
        mock_exec.return_value = ([{"col1": 1, "col2": "test"}], None)
        df, err = query_mysql("SELECT * FROM test", None)
        assert err is None
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["col1"] == 1


def test_query_mysql_failure():
    """Verify query_mysql handles database exceptions gracefully."""
    with patch("src.database.connection.execute_parameterized_query") as mock_exec:
        mock_exec.return_value = (None, "Connection timeout")
        df, err = query_mysql("SELECT 1", None)
        assert df is None
        assert err == "Connection timeout"


def test_execute_mcp_face_identification_offline_fallback():
    """Verify execute_mcp_face_identification handles offline database by providing gallery fallback."""
    with patch("app.service.query_mysql", return_value=(None, "DB offline")):
        req, resp, structured = execute_mcp_face_identification("001553.jpg")
        assert req["jsonrpc"] == "2.0"
        assert req["params"]["name"] == "mysql_identify_face"
        assert structured is not None
        assert structured["is_fallback"] is True
        assert structured["image_id"] == "001553.jpg"


def test_execute_mcp_face_identification_not_found():
    """Verify execute_mcp_face_identification returns JSON-RPC error when image does not exist."""
    with patch("app.service.query_mysql", return_value=(None, "No row")), \
         patch("app.service.load_gallery_data", return_value=[]):
        req, resp, structured = execute_mcp_face_identification("nonexistent_999999.jpg")
        assert "error" in resp
        assert resp["error"]["code"] == -32602
        assert structured is None
