"""
CelebA Face AI - Secure Relational Database Connection & Pooling Layer
Provides strict environment-driven database connectivity, thread-safe connection pooling,
liveness validation (ping-on-acquire), and parameterized query execution.
Eliminates insecure default credentials and SQL injection vectors.
"""

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

import pymysql
import pymysql.cursors

logger = logging.getLogger("celeba_db")


class ConnectionPool:
    """
    Thread-safe connection pool with connection validation and auto-reconnect.
    """
    def __init__(self, max_connections: int = 5) -> None:
        self.max_connections = max_connections
        self._pool: List[pymysql.Connection] = []
        self._lock = threading.Lock()

    def get_connection(self) -> pymysql.Connection:
        """Retrieves a healthy connection from the pool or creates a new one."""
        with self._lock:
            while self._pool:
                conn = self._pool.pop()
                try:
                    conn.ping(reconnect=True)
                    return conn
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
        return self._create_new_connection()

    def release_connection(self, conn: pymysql.Connection) -> None:
        """Returns a healthy connection to the pool or closes it if over capacity."""
        with self._lock:
            if len(self._pool) < self.max_connections:
                try:
                    conn.ping(reconnect=False)
                    self._pool.append(conn)
                    return
                except Exception:
                    pass
        try:
            conn.close()
        except Exception:
            pass

    def _create_new_connection(self) -> pymysql.Connection:
        required_variables = [
            "MYSQL_HOST",
            "MYSQL_USER",
            "MYSQL_PASSWORD",
            "MYSQL_DATABASE",
        ]
        missing = [var for var in required_variables if not os.getenv(var)]
        if missing:
            raise EnvironmentError(
                f"Missing database configuration: {missing}. "
                "Set environment variables or see config/database.env.example."
            )

        return pymysql.connect(
            host=os.environ["MYSQL_HOST"],
            user=os.environ["MYSQL_USER"],
            password=os.environ["MYSQL_PASSWORD"],
            database=os.environ["MYSQL_DATABASE"],
            port=int(os.getenv("MYSQL_PORT", "3306")),
            connect_timeout=5,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )


_GLOBAL_POOL = ConnectionPool(max_connections=5)


def get_database_connection() -> pymysql.Connection:
    """
    Establishes a validated connection to MySQL celeba_db using environment variables.
    Strictly verifies required configuration variables and rejects hardcoded fallbacks.
    """
    return _GLOBAL_POOL._create_new_connection()


@contextmanager
def get_pooled_connection() -> Generator[pymysql.Connection, None, None]:
    """Context manager for acquiring and releasing a connection from the pool."""
    conn = _GLOBAL_POOL.get_connection()
    try:
        yield conn
    finally:
        _GLOBAL_POOL.release_connection(conn)


def execute_parameterized_query(
    query: str, params: Optional[Tuple[Any, ...]] = None
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Executes a parameterized SQL query with resource disposal.
    Guarantees parameterized query execution to prevent SQL injection.
    """
    try:
        conn = get_database_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                results = cursor.fetchall()
            return results, None
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Database query execution failed: %s", e)
        return None, str(e)
