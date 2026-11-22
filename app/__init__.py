"""
CelebA Face AI & Intelligence System - Application Package
"""

from app.service import (
    execute_mcp_face_identification,
    load_benchmark_metrics,
    load_gallery_data,
    query_mysql,
)

__all__ = [
    "load_gallery_data",
    "load_benchmark_metrics",
    "query_mysql",
    "execute_mcp_face_identification",
]
