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
    if not isinstance(data, list) or len(data) == 0:
        pytest.fail("Query failed or returned empty data")
    row = data[0]
    assert isinstance(row, dict)
    assert row.get("person_id") == 2880
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


def test_live_mysql_roundtrip_when_service_available():
    """
    Verify genuine live MySQL connection and query round-trip without mocks
    when an active MySQL service endpoint is available (e.g. in GitHub Actions CI container).
    Gracefully skips if MySQL service is not reachable in local developer environment.
    """
    import importlib
    import os

    host = os.getenv("MYSQL_HOST")
    if not host or host == "192.0.2.1":
        pytest.skip("MYSQL_HOST not configured for live integration test; skipping.")

    try:
        pymysql = importlib.import_module("pymysql")
    except ImportError:
        pytest.skip("pymysql not installed; skipping live test.")

    try:
        conn = pymysql.connect(
            host=host,
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "celeba_user"),
            password=os.getenv("MYSQL_PASSWORD", "celeba_pass_2026"),
            database=os.getenv("MYSQL_DATABASE", "celeba_db"),
            connect_timeout=2,
        )
        conn.close()

        # Execute live parameterized query through application database layer
        data, err = execute_parameterized_query("SELECT 1 AS alive", None)
        assert err is None
        if not isinstance(data, list) or len(data) == 0:
            pytest.fail("Live query failed or returned empty data")
        row = data[0]
        assert isinstance(row, dict)
        assert row.get("alive") == 1
    except Exception as e:
        if os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true":
            pytest.fail(f"Live MySQL service is required in CI but failed to connect to {host}: {e}")
        pytest.skip(f"Live MySQL service not available at {host}: {e}; skipping live verification in local dev.")
