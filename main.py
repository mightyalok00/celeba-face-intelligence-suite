"""
CelebA Face AI & Intelligence System - Universal Python Entrypoint
Compatible with local workstations, Docker, and container execution.

Author: Alok
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.google_ai_studio_app import main

if __name__ == "__main__":
    main()
