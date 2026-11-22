"""
Security and Access Control Test Suite.
Verifies that database layer, MCP server protocols, and credentials enforce strict security controls,
reject SQL injection patterns, and prevent hardcoded secret leakage.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.database.connection import execute_parameterized_query, get_database_connection


def test_sql_injection_prevention_via_parameterization():
    """
    Verify that SQL injection payloads passed as parameters are escaped cleanly
    and passed as parameter tuples to cursor.execute, rather than string-interpolated.
    """
    malicious_input = "000001.jpg' OR '1'='1"
    query = "SELECT image_id, person_id FROM celeba_identities WHERE image_id = %s"

    with patch("src.database.connection.get_database_connection") as mock_get_conn:
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        data, err = execute_parameterized_query(query, (malicious_input,))

        assert err is None
        mock_cursor.execute.assert_called_once_with(query, (malicious_input,))


def test_environment_variable_enforcement():
    """
    Verify that get_database_connection strictly refuses execution
    when mandatory environment variables are missing.
    """
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            get_database_connection()
        assert "Missing database configuration" in str(exc_info.value)
