"""
CelebA Face AI & Intelligence System - Web Application Launcher
Clean, maintainable entrypoint delegating directly to app.app.main().

Author: Alok
"""

import os
import sys

# Ensure application package and root are on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.app import main

if __name__ == "__main__":
    if "--server" in sys.argv or "--dev" in sys.argv:
        from app.google_ai_studio_app import main as studio_main

        studio_main()
    else:
        main()
