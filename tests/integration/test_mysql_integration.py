"""
Integration tests for MySQL database connectivity and queries.
Validates behavior in both connected environments and offline fallbacks.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.database.connection import execute_parameterized_query


@patch("src.database.connection.pymysql.connect")
def test_full_query_roundtrip_mocked(mock_connect):
    """Verify that execute_parameterized_query handles query execution and connection teardown."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {"image_id": "000001.jpg", "file_name": "000001.jpg", "person_id": 2880}
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    sql = "SELECT image_id, file_name, person_id FROM celeba_identities WHERE person_id = %s"
    data, err = execute_parameterized_query(sql, (2880,))

    assert err is None
    assert len(data) == 1
    assert data[0]["person_id"] == 2880
    mock_conn.close.assert_called_once()


def test_offline_mode_graceful_handling():
    """Verify that when database is unavailable, an error string is returned without crashing."""
    # Using an invalid host/port to simulate offline database
    with patch.dict("os.environ", {"MYSQL_HOST": "192.0.2.1", "MYSQL_PORT": "1234"}):
        data, err = execute_parameterized_query("SELECT 1", None)
        assert data is None
        assert err is not None
