"""
Unit tests for database connection pooling and liveness validation.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.database.connection import ConnectionPool, get_pooled_connection


def test_connection_pool_get_and_release():
    """Verify that connection pool reuses healthy connections and manages capacity."""
    pool = ConnectionPool(max_connections=2)

    mock_conn = MagicMock()
    mock_conn.ping.return_value = None

    with patch.object(pool, "_create_new_connection", return_value=mock_conn):
        # 1. Acquire connection
        conn1 = pool.get_connection()
        assert conn1 is mock_conn

        # 2. Release connection back to pool
        pool.release_connection(conn1)
        assert len(pool._pool) == 1

        # 3. Re-acquire from pool (ping should be called to validate liveness)
        conn2 = pool.get_connection()
        assert conn2 is mock_conn
        mock_conn.ping.assert_called_with(reconnect=True)
        assert len(pool._pool) == 0


def test_pooled_connection_context_manager():
    """Verify context manager automatically releases connection back to pool."""
    mock_conn = MagicMock()
    with patch("src.database.connection._GLOBAL_POOL.get_connection", return_value=mock_conn):
        with patch("src.database.connection._GLOBAL_POOL.release_connection") as mock_rel:
            with get_pooled_connection() as conn:
                assert conn is mock_conn
            mock_rel.assert_called_once_with(mock_conn)
