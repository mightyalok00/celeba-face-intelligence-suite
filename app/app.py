"""
CelebA Face AI & Intelligence System - Application Core
Re-exports core services, Google AI Studio integration, and backward-compatible interfaces.

Author: Alok
"""

from app.google_ai_studio_app import main
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
    "main",
]

if __name__ == "__main__":
    main()
