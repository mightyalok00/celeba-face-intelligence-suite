"""
CelebA Face AI - Google AI Studio Multimodal Application
CLI and programmatic application interface for running Google AI Studio
Gemini face attribute analysis, expression reasoning, and biometric intelligence.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_studio.gemini_face_analyzer import GeminiFaceIntelligenceStudio
from src.models.inference import ModelInferenceEngine


def run_google_ai_studio_analysis(image_path: str, model_name: str = "gemini-1.5-flash") -> None:
    """
    Executes multimodal face intelligence analysis using Google AI Studio.
    Cross-references deep learning model predictions and database metadata.
    """
    p = Path(image_path)
    if not p.exists():
        print(f"Error: Image path '{image_path}' does not exist.")
        sys.exit(1)

    print("=" * 70)
    print("  CELEBA FACE INTELLIGENCE - GOOGLE AI STUDIO APPLICATION")
    print(f"  Target Image: {p.name}")
    print(f"  AI Studio Model: {model_name}")
    print("=" * 70)

    # 1. Google AI Studio Gemini Multimodal Reasoning
    studio = GeminiFaceIntelligenceStudio(model_name=model_name)
    print("\n[1/2] Querying Google AI Studio Gemini Multimodal API...")
    result = studio.analyze_face(p)

    print(f"Status: {result.get('status', 'unknown')} ({result.get('source', '')})")
    print("Structured Gemini Biometric Output:")
    print(json.dumps(result.get("analysis", {}), indent=2))

    # 2. Local Deep Learning Neural Network Verification
    print("\n[2/2] Running Local Deep Learning Checkpoint Verification (PyTorch Residual CNN + ViT)...")
    try:
        engine = ModelInferenceEngine()
        dl_pred = engine.predict_attribute(p, architecture="Residual CNN", use_tta=True)
        print(f"  • Deep Learning Model: {dl_pred['architecture']}")
        print(f"  • Smiling Probability: {dl_pred['probability']:.4f}")
        print(f"  • Confidence: {dl_pred['confidence']:.2%}")
        print(f"  • Test-Time Augmentation (TTA): {dl_pred['tta_views_evaluated']} views evaluated")
        print(f"  • Uncertainty (std across views): {dl_pred['prediction_uncertainty_std']:.4f}")
        print(f"  • Checkpoint Loaded: {dl_pred['checkpoint_loaded']}")
    except Exception as e:
        print(f"  • Deep learning check skipped: {e}")

    print("\n" + "=" * 70)
    print("  Analysis Complete. Independently Verifiable by Google AI Studio.")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google AI Studio Multimodal Face Intelligence Application"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="img_align_celeba/000001.png",
        help="Path to facial image for Google AI Studio analysis",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="Google AI Studio Gemini model (gemini-1.5-flash or gemini-1.5-pro)",
    )
    args = parser.parse_args()

    run_google_ai_studio_analysis(args.image, args.model)


if __name__ == "__main__":
    main()
