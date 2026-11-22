"""
Unit tests for Google AI Studio web application routes, health probes, and analysis routines.
"""

import json
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.google_ai_studio_app import (
    GoogleAIStudioRequestHandler,
    ResilientThreadingHTTPServer,
    main,
    run_google_ai_studio_analysis,
)


def test_run_google_ai_studio_analysis_missing_file():
    """Verify run_google_ai_studio_analysis returns error dictionary when target file is missing."""
    res = run_google_ai_studio_analysis("nonexistent_path_to_image_xyz.jpg")
    assert res["status"] == "error"
    assert "does not exist" in res["message"]


def test_run_google_ai_studio_analysis_mocked_success(tmp_path):
    """Verify run_google_ai_studio_analysis executes analysis and cross-validates deep learning predictions."""
    test_img = tmp_path / "test_face.png"
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(200, 150, 120))
    img.save(test_img)

    mock_studio_res = {
        "status": "success",
        "source": "gemini_mock",
        "analysis": {"expression": {"is_smiling": True, "confidence": 0.95}},
    }

    mock_dl_pred = {
        "architecture": "Residual CNN",
        "probability": 0.92,
        "confidence": 0.92,
        "tta_views_evaluated": 2,
        "prediction_uncertainty_std": 0.02,
        "checkpoint_loaded": True,
    }

    with patch("app.google_ai_studio_app.GeminiFaceIntelligenceStudio") as mock_studio_cls, \
         patch("src.models.inference.ModelInferenceEngine") as mock_engine_cls:

        mock_studio = MagicMock()
        mock_studio.analyze_face.return_value = mock_studio_res
        mock_studio_cls.return_value = mock_studio

        mock_engine = MagicMock()
        mock_engine.predict_attribute.return_value = mock_dl_pred
        mock_engine_cls.return_value = mock_engine

        result = run_google_ai_studio_analysis(str(test_img))
        assert result["status"] == "success"
        assert result["google_ai_studio"]["status"] == "success"
        assert result["deep_learning_verification"]["architecture"] == "Residual CNN"
        assert result["deep_learning_verification"]["smiling_probability"] == 0.92


def test_google_ai_studio_server_in_process_probes():
    """
    Verify ResilientThreadingHTTPServer and GoogleAIStudioRequestHandler respond to
    pre-warming HEAD probes, GET health routes, gallery APIs, and POST endpoints.
    """
    # Bind to port 0 for ephemeral dynamic port assignment
    server = ResilientThreadingHTTPServer(("127.0.0.1", 0), GoogleAIStudioRequestHandler)
    port = server.server_port
    base_url = f"http://127.0.0.1:{port}"

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    try:
        # 1. Test HEAD /health pre-warming probe
        req_head = urllib.request.Request(f"{base_url}/health", method="HEAD")
        with urllib.request.urlopen(req_head, timeout=3) as res:
            assert res.status == 200

        # 2. Test GET /health
        with urllib.request.urlopen(f"{base_url}/health", timeout=3) as res:
            assert res.status == 200
            body = json.loads(res.read().decode())
            assert body["status"] == "healthy"
            assert body["ready"] is True

        # 3. Test GET /api/gallery
        with urllib.request.urlopen(f"{base_url}/api/gallery", timeout=3) as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert isinstance(data, list)

        # 4. Test GET /api/benchmarks
        with urllib.request.urlopen(f"{base_url}/api/benchmarks", timeout=3) as res:
            assert res.status == 200

        # 5. Test OPTIONS CORS Preflight
        req_options = urllib.request.Request(f"{base_url}/", method="OPTIONS")
        with urllib.request.urlopen(req_options, timeout=3) as res:
            assert res.status == 204

        # 6. Test POST /api/mcp
        post_data = json.dumps({"image_id": "001553.jpg"}).encode("utf-8")
        req_post = urllib.request.Request(
            f"{base_url}/api/mcp",
            data=post_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_post, timeout=3) as res:
            assert res.status == 200
            resp_json = json.loads(res.read().decode())
            assert "request" in resp_json

    finally:
        server.shutdown()


def test_main_cli_execution():
    """Verify main() entrypoint parses arguments and triggers analysis or server boot cleanly."""
    with patch("sys.argv", ["google_ai_studio_app.py", "--image", "test_img.png"]), \
         patch("app.google_ai_studio_app.run_google_ai_studio_analysis") as mock_analysis:
        main()
        mock_analysis.assert_called_once_with("test_img.png", "gemini-1.5-flash")
