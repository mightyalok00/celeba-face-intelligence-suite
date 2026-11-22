"""
CelebA Face AI & Intelligence System - Web Application Launcher
Universal entrypoint compatible with local development, Streamlit CLI,
Google AI Studio / Cloud Run, Render.com, and container dev environments.

Author: Alok
"""

import os
import sys

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def start_server() -> None:
    """
    Starts the web application development server.
    Detects environment PORT and binds to 0.0.0.0 to satisfy cloud health checks.
    """
    # Enforce headless mode and suppress telemetry / onboarding prompts
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    port = os.getenv("PORT", "8501")
    target_script = os.path.join(BASE_DIR, "app", "app.py")

    # If executed via 'streamlit run app.py', delegate to main()
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            from app.app import main

            main()
            return
    except Exception:
        pass

    # If executed directly via 'python app.py', launch the Streamlit web server
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
        # Fallback to direct execution
        print(f"[Launcher] Streamlit CLI launch encountered: {e}. Falling back to main().")
        from app.app import main

        main()


if __name__ == "__main__":
    start_server()
