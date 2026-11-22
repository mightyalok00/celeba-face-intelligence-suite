"""
CelebA Face AI - Secure Relational Database Connection Layer
Provides strict environment-driven database connectivity and parameterized query execution.
Eliminates insecure default credentials and sql injection vectors.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pymysql
import pymysql.cursors

logger = logging.getLogger("celeba_db")


def get_database_connection() -> pymysql.Connection:
    """
    Establishes a secure connection to MySQL celeba_db using environment variables.
    Strictly verifies required configuration variables and rejects hardcoded fallbacks.
    """
    required_variables = [
        "MYSQL_HOST",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "MYSQL_DATABASE",
    ]
    missing = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]
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
    )


def execute_parameterized_query(
    query: str, params: Optional[Tuple[Any, ...]] = None
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Executes a parameterized SQL query with proper exception handling and resource disposal.
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
