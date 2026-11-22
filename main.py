"""
CelebA Face AI & Intelligence System - Main Entrypoint
Compatible with Google AI Studio, Google Cloud Run, Render, and container platforms.

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
