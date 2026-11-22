"""
CelebA Face AI & Intelligence System - Main Entrypoint
Universal entrypoint compatible with platforms invoking `python main.py`.

Author: Alok
"""

import os
import sys

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def main() -> None:
    """Delegates directly to the web development server runner."""
    # Set headless flags
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    port = os.getenv("PORT", "8501")
    target_script = os.path.join(BASE_DIR, "app", "app.py")

    try:
        from streamlit.web import cli as stcli

        sys.argv = [
            "streamlit",
            "run",
            target_script,
            "--server.port",
            str(port),
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]
        sys.exit(stcli.main())
    except Exception as e:
        print(f"[main.py] Streamlit CLI launch encountered: {e}. Falling back to main().")
        from app.app import main as streamlit_main

        streamlit_main()


if __name__ == "__main__":
    main()
