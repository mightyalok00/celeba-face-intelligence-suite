"""
Integration tests for MySQL database connectivity and queries.
Validates behavior in both connected environments and offline fallbacks.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.database.connection import execute_parameterized_query


@patch("src.database.connection.pymysql.connect")
def test_full_query_roundtrip_mocked(mock_connect):
    """
    Verify that execute_parameterized_query executes a parameterized query,
    retrieves dictionary-mapped rows, and closes the connection cleanly.
    """
    # 1. Setup mock cursor and simulated row result
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {"image_id": "000001.jpg", "file_name": "000001.jpg", "person_id": 2880}
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    # 2. Execute parameterized SQL query
    sql = "SELECT image_id, file_name, person_id FROM celeba_identities WHERE person_id = %s"
    data, err = execute_parameterized_query(sql, (2880,))

    # 3. Assert error-free execution and narrow type for static analyzers
    assert err is None
    if not isinstance(data, list):
        raise AssertionError("execute_parameterized_query returned None for valid query")

    # 4. Assert row contents and connection lifecycle disposal
    assert len(data) == 1
    assert data[0]["person_id"] == 2880
    mock_conn.close.assert_called_once()


def test_offline_mode_graceful_handling():
    """
    Verify that when the MySQL database is unreachable or offline,
    execute_parameterized_query returns (None, error_str) gracefully without raising unhandled exceptions.
    """
    # Simulate unavailable database endpoint with unreachable host and port
    with patch.dict("os.environ", {"MYSQL_HOST": "192.0.2.1", "MYSQL_PORT": "1234"}):
        data, err = execute_parameterized_query("SELECT 1", None)
        assert data is None
        assert err is not None
