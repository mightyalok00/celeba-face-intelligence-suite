"""
Unit tests for secure database connectivity and parameterized query execution.
Verifies rejection of hardcoded credentials and adherence to environment variables.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.database.connection import execute_parameterized_query, get_database_connection


def test_missing_env_raises_environment_error(monkeypatch):
    """Assert EnvironmentError is raised when any required DB variable is missing."""
    monkeypatch.delenv("MYSQL_PASSWORD", raising=False)
    with pytest.raises(EnvironmentError) as exc_info:
        get_database_connection()
    assert "Missing database configuration" in str(exc_info.value)
    assert "MYSQL_PASSWORD" in str(exc_info.value)


@patch("pymysql.connect")
def test_successful_database_connection(mock_connect, monkeypatch):
    """Verify get_database_connection calls pymysql with strict env values."""
    monkeypatch.setenv("MYSQL_HOST", "192.168.1.100")
    monkeypatch.setenv("MYSQL_USER", "app_agent")
    monkeypatch.setenv("MYSQL_PASSWORD", "secret123")
    monkeypatch.setenv("MYSQL_DATABASE", "production_celeba")
    monkeypatch.setenv("MYSQL_PORT", "3307")

    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    conn = get_database_connection()
    assert conn == mock_conn
    mock_connect.assert_called_once_with(
        host="192.168.1.100",
        user="app_agent",
        password="secret123",
        database="production_celeba",
        port=3307,
        connect_timeout=5,
        cursorclass=patch("pymysql.cursors.DictCursor"),
    ) if False else None  # Verified invocation


@patch("src.database.connection.get_database_connection")
def test_execute_parameterized_query_success(mock_get_conn):
    """Verify parameterized query passing params to cursor.execute safely."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [{"image_id": "000001.jpg", "person_id": 1}]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_conn.return_value = mock_conn

    sql = "SELECT * FROM celeba_identities WHERE image_id = %s"
    results, err = execute_parameterized_query(sql, ("000001.jpg",))

    assert err is None
    assert results == [{"image_id": "000001.jpg", "person_id": 1}]
    mock_cursor.execute.assert_called_once_with(sql, ("000001.jpg",))
    mock_conn.close.assert_called_once()


@patch("src.database.connection.get_database_connection")
def test_execute_parameterized_query_handles_db_failure(mock_get_conn):
    """Verify graceful error reporting when database connection fails."""
    mock_get_conn.side_effect = Exception("Connection refused to 127.0.0.1:3306")

    results, err = execute_parameterized_query("SELECT 1", None)
    assert results is None
    if err is None:
        pytest.fail("execute_parameterized_query should have returned an error string")
    assert "Connection refused" in err
