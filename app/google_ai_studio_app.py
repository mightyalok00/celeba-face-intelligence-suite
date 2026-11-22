"""
CelebA Face AI - Google AI Studio Multimodal Application & Web Server
CLI and interactive HTTP application interface for running Google AI Studio
Gemini face attribute analysis, expression reasoning, biometric intelligence,
and local deep learning checkpoint cross-validation.

Author: Alok
"""

import argparse
import json
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.service import (
    execute_mcp_face_identification,
    load_benchmark_metrics,
    load_gallery_data,
    query_mysql,
)
from src.ai_studio.gemini_face_analyzer import GeminiFaceIntelligenceStudio


def run_google_ai_studio_analysis(
    image_path: str, model_name: str = "gemini-1.5-flash"
) -> Dict[str, Any]:
    """
    Executes multimodal face intelligence analysis using Google AI Studio.
    Cross-references deep learning model predictions and database metadata.

    Args:
        image_path: String path to the target face image.
        model_name: Target Google AI Studio Gemini model (e.g., 'gemini-1.5-flash').

    Returns:
        Dictionary containing multimodal analysis and local deep learning verification.
    """
    p = Path(image_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    if not p.exists():
        err_msg = f"Error: Target image path '{image_path}' does not exist."
        print(err_msg, flush=True)
        return {"status": "error", "message": err_msg}

    print("=" * 70, flush=True)
    print("  CELEBA FACE INTELLIGENCE - GOOGLE AI STUDIO APPLICATION", flush=True)
    print(f"  Target Image: {p.name}", flush=True)
    print(f"  AI Studio Model: {model_name}", flush=True)
    print("=" * 70, flush=True)

    # 1. Google AI Studio Gemini Multimodal Analysis
    studio = GeminiFaceIntelligenceStudio(model_name=model_name)
    print("\n[1/2] Querying Google AI Studio Gemini Multimodal API...", flush=True)
    result = studio.analyze_face(p)

    print(f"Status: {result.get('status', 'unknown')} ({result.get('source', '')})", flush=True)
    print("Structured Gemini Biometric Output:", flush=True)
    print(json.dumps(result.get("analysis", {}), indent=2), flush=True)

    # 2. Local Deep Learning Neural Network Verification (PyTorch Residual CNN + ViT)
    print(
        "\n[2/2] Running Local Deep Learning Checkpoint Verification (PyTorch Residual CNN + ViT)...",
        flush=True,
    )
    dl_summary: Dict[str, Any] = {}
    try:
        from src.models.inference import ModelInferenceEngine

        engine = ModelInferenceEngine()
        dl_pred = engine.predict_attribute(p, architecture="Residual CNN", use_tta=True)
        dl_summary = {
            "architecture": dl_pred["architecture"],
            "smiling_probability": dl_pred["probability"],
            "confidence": dl_pred["confidence"],
            "tta_views_evaluated": dl_pred["tta_views_evaluated"],
            "prediction_uncertainty_std": dl_pred["prediction_uncertainty_std"],
            "checkpoint_loaded": dl_pred["checkpoint_loaded"],
        }
        print(f"  • Deep Learning Model: {dl_pred['architecture']}", flush=True)
        print(f"  • Smiling Probability: {dl_pred['probability']:.4f}", flush=True)
        print(f"  • Confidence: {dl_pred['confidence']:.2%}", flush=True)
        print(f"  • Test-Time Augmentation (TTA): {dl_pred['tta_views_evaluated']} views evaluated", flush=True)
        print(f"  • Uncertainty (std across views): {dl_pred['prediction_uncertainty_std']:.4f}", flush=True)
        print(f"  • Checkpoint Loaded: {dl_pred['checkpoint_loaded']}", flush=True)
    except Exception as e:
        print(f"  • Deep learning check skipped: {e}", flush=True)
        dl_summary = {"error": str(e)}

    print("\n" + "=" * 70, flush=True)
    print("  Analysis Complete. Independently Verifiable by Google AI Studio.", flush=True)
    print("=" * 70, flush=True)

    return {
        "status": "success",
        "image": p.name,
        "google_ai_studio": result,
        "deep_learning_verification": dl_summary,
    }


class ResilientThreadingHTTPServer(ThreadingHTTPServer):
    """Threading HTTP Server with address reuse and daemonized client workers."""

    allow_reuse_address = True
    daemon_threads = True


class GoogleAIStudioRequestHandler(SimpleHTTPRequestHandler):
    """
    HTTP Request Handler serving the Google AI Studio Web Application,
    Google Photos interface, container health probes, and JSON REST APIs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def end_headers(self) -> None:
        """Adds CORS and security headers for Google AI Studio embedding."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        """Handles CORS preflight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_HEAD(self) -> None:
        """Handles pre-warming HEAD probes from cloud runtimes and health checkers."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # Healthcheck and root HEAD requests
        if path in (
            "/",
            "/index.html",
            "/health",
            "/healthz",
            "/_ah/health",
            "/_ah/start",
            "/_ah/warmup",
            "/api/health",
            "/ping",
            "/status",
            "/api/dev/status",
        ):
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8" if path in ("/", "/index.html") else "application/json",
            )
            self.end_headers()
            return

        super().do_HEAD()

    def do_GET(self) -> None:
        """Handles API GET requests, pre-warming health probes, and static web UI routing."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # 1. Root & Web UI routing
        if path in ("/", "/index.html"):
            index_path = PROJECT_ROOT / "frontend" / "index.html"
            if not index_path.exists():
                index_path = PROJECT_ROOT / "index.html"
            if index_path.exists():
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(index_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        # 2. Container pre-warming and load balancer health endpoints
        if path in (
            "/health",
            "/healthz",
            "/_ah/health",
            "/_ah/start",
            "/_ah/warmup",
            "/api/health",
            "/ping",
            "/status",
            "/api/dev/status",
        ):
            self._send_json(
                {
                    "status": "healthy",
                    "service": "Google AI Studio Face Intelligence",
                    "provider": "Google AI Studio / Gemini API",
                    "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
                    "ready": True,
                }
            )
            return

        # 3. Gallery JSON dataset stream
        if path in ("/api/gallery", "/gallery_data.json"):
            gallery = load_gallery_data()
            self._send_json(gallery)
            return

        # 4. Production benchmark results API
        if path == "/api/benchmarks":
            df = load_benchmark_metrics()
            records = df.to_dict(orient="records") if df is not None else []
            self._send_json(records)
            return

        # Fallback to standard static file server
        super().do_GET()

    def do_POST(self) -> None:
        """Handles API POST requests for Google AI Studio face analysis and MCP tools."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        if path == "/api/analyze":
            img = payload.get("image", payload.get("image_path", "img_align_celeba/000001.png"))
            model = payload.get("model", "gemini-1.5-flash")
            response_data = run_google_ai_studio_analysis(img, model_name=model)
            self._send_json(response_data)
            return

        if path in ("/api/mcp", "/api/mcp/identify"):
            image_id = payload.get("image_id", "000001.jpg")
            req, resp, structured = execute_mcp_face_identification(image_id)
            self._send_json({"request": req, "response": resp, "structured": structured})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Helper to send a JSON response."""
        encoded = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def start_google_ai_studio_server(port: int = 8080, host: str = "0.0.0.0") -> None:  # nosec B104
    """
    Starts the high-performance Google AI Studio Web Application server.
    Serves the Google Photos UI, REST API, and Gemini multimodal endpoints.

    Args:
        port: Port to bind to (e.g., 8080 or from os.getenv('PORT')).
        host: Host address (defaults to '0.0.0.0' for cloud container compatibility).
    """
    # Safe port resolution from environment
    env_port = os.getenv("PORT")
    if env_port and env_port.strip().isdigit():
        port = int(env_port.strip())

    server_address = (host, port)
    server = ResilientThreadingHTTPServer(server_address, GoogleAIStudioRequestHandler)

    print("=" * 70, flush=True)
    print("  CELEBA FACE INTELLIGENCE - GOOGLE AI STUDIO WEB APPLICATION", flush=True)
    print(f"  Server listening on: http://{host}:{port}", flush=True)
    print("  Google AI Studio API: Active (gemini-1.5-flash & gemini-1.5-pro)", flush=True)
    print(f"  Web Interface: http://localhost:{port}/", flush=True)
    print("  Health Probe: http://localhost:{port}/health", flush=True)
    print("=" * 70, flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Google AI Studio server...", flush=True)
        server.shutdown()
    except Exception as e:
        print(f"\nServer loop terminated: {e}", flush=True)


def main() -> None:
    """CLI and server entry point for Google AI Studio application."""
    # Enforce unbuffered stdout for immediate cloud container logging
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Default port resolution
    raw_port = os.getenv("PORT")
    default_port = int(raw_port.strip()) if raw_port and raw_port.strip().isdigit() else 8080

    parser = argparse.ArgumentParser(
        description="Google AI Studio Multimodal Face Intelligence Application"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run interactive Google AI Studio web application server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Port to bind the server to (defaults to $PORT or 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",  # nosec B104
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Target facial image for CLI analysis (if omitted, boots web server)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="Google AI Studio Gemini model (gemini-1.5-flash or gemini-1.5-pro)",
    )

    # Use parse_known_args so unexpected runner flags never cause the dev server process to die
    args, _ = parser.parse_known_args()

    # If --image is specified, execute CLI analysis; otherwise start Google AI Studio web server
    if args.image:
        run_google_ai_studio_analysis(args.image, args.model)
    else:
        start_google_ai_studio_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
