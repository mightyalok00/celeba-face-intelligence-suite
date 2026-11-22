"""
Unit tests for Google AI Studio face intelligence analyzer.
Verifies configuration detection, prompt execution, structured output schema,
and graceful offline fallback handling.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.ai_studio.gemini_face_analyzer import GeminiFaceIntelligenceStudio


def test_gemini_studio_unconfigured_initialization():
    """Verify analyzer initializes safely without API key in offline mode."""
    with patch.dict("os.environ", {}, clear=True):
        studio = GeminiFaceIntelligenceStudio(api_key="")
        assert not studio.is_configured
        assert studio.model is None


def test_gemini_studio_offline_fallback_schema():
    """Verify structured JSON fallback schema when running offline."""
    with patch.dict("os.environ", {}, clear=True):
        studio = GeminiFaceIntelligenceStudio(api_key="")
        test_img = Image.new("RGB", (64, 64), color="blue")

        result = studio.analyze_face(test_img)

        assert result["status"] == "success"
        assert result["source"] == "google_ai_studio_offline_mode"
        assert "analysis" in result

        analysis = result["analysis"]
        assert "expression" in analysis
        assert "is_smiling" in analysis["expression"]
        assert "demographics" in analysis
        assert "facial_attributes" in analysis
        assert "reasoning_summary" in analysis


def test_gemini_studio_live_mock_response():
    """Verify parsing and structure when Gemini API returns valid JSON."""
    mock_payload = {
        "expression": {"is_smiling": True, "confidence": 0.95, "primary_emotion": "Smiling"},
        "demographics": {
            "apparent_age_group": "Young",
            "apparent_gender": "Female",
            "confidence": 0.90,
        },
        "facial_attributes": {
            "hair_color": "Blond",
            "wearing_eyeglasses": False,
            "facial_hair": False,
            "notable_features": [],
        },
        "reasoning_summary": "Mocked test rationale.",
    }

    studio = GeminiFaceIntelligenceStudio(api_key="mock_key")
    studio._is_configured = True
    studio.model = MagicMock()
    mock_response = MagicMock()
    import json

    mock_response.text = json.dumps(mock_payload)
    studio.model.generate_content.return_value = mock_response

    test_img = Image.new("RGB", (64, 64), color="red")
    result = studio.analyze_face(test_img)

    assert result["status"] == "success"
    assert result["source"] == "google_ai_studio_gemini"
    assert result["analysis"]["expression"]["is_smiling"] is True
    assert result["analysis"]["demographics"]["apparent_gender"] == "Female"
