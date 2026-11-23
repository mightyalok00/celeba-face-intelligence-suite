"""
CelebA AI — Face Intelligence Studio (Python FastAPI ML Inference Microservice)

Wraps the existing PyTorch architectures (ANN, Spatial BiLSTM, Residual CNN,
Vision Transformer) and XGBoost / Ensemble pipelines for real-time image inference.
Preserves all existing src/ and app/ modules without modifying their interfaces.

Run locally or in Docker alongside the Node.js Express gateway:
    uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.service import load_benchmark_metrics, query_mysql
from src.models.inference import ModelInferenceEngine

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

app = FastAPI(
    title="CelebA AI — ML Inference Backend",
    version="2.5.0",
    description="Real PyTorch & XGBoost facial attribute inference service for CelebA AI Studio.",
)

import os

_default_origins = ["http://127.0.0.1:3000", "http://localhost:3000"]
_extra_origins = [
    o.strip()
    for o in os.environ.get("CELEBA_CORS_ORIGINS", "").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_engine: Optional[ModelInferenceEngine] = None
_engine_error: Optional[str] = None


def get_inference_engine() -> ModelInferenceEngine:
    global _engine, _engine_error
    if _engine is None:
        try:
            _engine = ModelInferenceEngine(strict=False)
            _engine_error = None
        except Exception as exc:
            _engine_error = str(exc)
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize ModelInferenceEngine: {exc}",
            ) from exc
    return _engine


def validate_uploaded_image(raw_bytes: bytes, content_type: Optional[str]) -> Image.Image:
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB size limit.")
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Allowed: JPEG, PNG, WebP.",
        )

    # Magic-byte verification
    is_jpeg = raw_bytes.startswith(b"\xff\xd8\xff")
    is_png = raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP"
    if not (is_jpeg or is_png or is_webp):
        raise HTTPException(
            status_code=400,
            detail="Invalid image signature. File contents do not match JPEG, PNG, or WebP.",
        )

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Corrupt or unreadable image: {exc}") from exc


@app.get("/health")
def health() -> Dict[str, Any]:
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    checkpoints = {
        "bilstm.pt": (ckpt_dir / "bilstm.pt").exists(),
        "vision_transformer.pt": (ckpt_dir / "vision_transformer.pt").exists(),
        "residual_cnn.pt": (ckpt_dir / "residual_cnn.pt").exists(),
        "ann.pt": (ckpt_dir / "ann.pt").exists(),
    }
    expected_architectures = ["Vision Transformer", "Spatial BiLSTM", "Residual CNN", "ANN"]
    engine_ready = False
    verified: List[str] = []
    device = "cpu"
    try:
        eng = get_inference_engine()
        engine_ready = True
        verified = sorted(eng.verified_checkpoints)
        device = str(eng.device)
    except HTTPException:
        engine_ready = False

    missing = [arch for arch in expected_architectures if arch not in verified]
    return {
        "status": "healthy" if engine_ready else "degraded",
        "service": "celeba-fastapi-ml",
        "engine_ready": engine_ready,
        "device": device,
        "verified_checkpoints": verified,
        "missing_checkpoints": missing,
        "checkpoint_files": checkpoints,
        "error": _engine_error,
    }


@app.get("/api/models")
def list_models() -> Dict[str, Any]:
    registry_path = PROJECT_ROOT / "models" / "model_registry.json"
    registry: Dict[str, Any] = {}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

    eng = get_inference_engine()
    return {
        "registry": registry,
        "loaded_checkpoints": eng.loaded_checkpoints,
        "verified_checkpoints": sorted(eng.verified_checkpoints),
        "device": str(eng.device),
    }


@app.get("/api/benchmarks")
def get_benchmarks() -> Dict[str, Any]:
    df = load_benchmark_metrics()
    if df is None:
        raise HTTPException(status_code=404, detail="Benchmark CSV reports not found.")
    return {"benchmarks": df.to_dict(orient="records")}


@app.get("/api/mysql/status")
def mysql_status() -> Dict[str, Any]:
    df, err = query_mysql("SELECT COUNT(*) AS identity_count FROM celeba_identities")
    if err:
        return {"connected": False, "error": err}
    count = int(df.iloc[0]["identity_count"]) if df is not None and not df.empty else 0
    return {"connected": True, "database": "celeba_db", "identity_count": count}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    architecture: str = Form("Vision Transformer"),
    use_tta: bool = Form(True),
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    raw_bytes = await file.read()
    # In-memory validation only; uploaded facial image is never persisted to disk
    img = validate_uploaded_image(raw_bytes, file.content_type)

    eng = get_inference_engine()
    supported = {"Residual CNN", "Vision Transformer", "Spatial BiLSTM", "ANN"}
    if architecture not in supported:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Architecture '{architecture}' is not available for live PyTorch checkpoint inference. "
                f"Supported PyTorch checkpoints: {sorted(supported)}. "
                "Tabular/pixel XGBoost models require serialized booster artifacts not present in models/checkpoints/."
            ),
        )

    if architecture not in eng.verified_checkpoints:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Verified trained checkpoint for '{architecture}' was not found on disk. "
                f"Verified checkpoints currently available: {sorted(eng.verified_checkpoints)}. "
                "Refusing to return uninitialized weights as real predictions."
            ),
        )

    result = eng.predict_attribute(img, architecture=architecture, use_tta=use_tta)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    result["latency_ms"] = elapsed_ms
    result["device"] = str(eng.device)
    result["retained_on_disk"] = False
    return result


def _load_slsqp_weights() -> Dict[str, float]:
    """Loads calibrated SLSQP weights from latest_training_run.json without hardcoding fallback fake predictions."""
    run_path = PROJECT_ROOT / "reports" / "benchmarks" / "latest_training_run.json"
    if not run_path.exists():
        return {}
    try:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        raw_weights = payload.get("optimal_ensemble_weights") or {}
        mapped: Dict[str, float] = {}
        for key, value in raw_weights.items():
            lowered = str(key).lower()
            weight_val = float(value)
            if "residual" in lowered or "cnn" in lowered:
                mapped["Residual CNN"] = weight_val
            elif "vision" in lowered or "vit" in lowered:
                mapped["Vision Transformer"] = weight_val
            elif "bilstm" in lowered:
                mapped["Spatial BiLSTM"] = weight_val
            elif "ann" in lowered:
                mapped["ANN"] = weight_val
        return mapped
    except Exception:
        return {}


@app.post("/api/predict/consensus")
async def predict_consensus(
    file: UploadFile = File(...),
    use_tta: bool = Form(True),
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    raw_bytes = await file.read()
    # In-memory validation only; uploaded facial image is never persisted to disk
    img = validate_uploaded_image(raw_bytes, file.content_type)

    eng = get_inference_engine()
    ordered_candidates = ["Residual CNN", "Vision Transformer", "Spatial BiLSTM", "ANN"]
    available = [arch for arch in ordered_candidates if arch in eng.verified_checkpoints]

    if not available:
        raise HTTPException(
            status_code=503,
            detail=(
                "No verified PyTorch model checkpoints are available on disk for consensus evaluation. "
                "Refusing to return uninitialized weights as real predictions."
            ),
        )

    slsqp_raw = _load_slsqp_weights()
    raw_weights = {arch: max(0.0, float(slsqp_raw.get(arch, 0.0))) for arch in available}
    weight_sum = sum(raw_weights.values())

    if weight_sum <= 0.0:
        norm_weights = {arch: 1.0 / len(available) for arch in available}
        weighting_strategy = "uniform_over_verified_checkpoints"
    else:
        norm_weights = {arch: raw_weights[arch] / weight_sum for arch in available}
        weighting_strategy = "slsqp_renormalized_over_verified_checkpoints"

    per_model: List[Dict[str, Any]] = []
    weighted_prob = 0.0
    smiling_votes = 0

    for arch in available:
        m_t0 = time.perf_counter()
        single = eng.predict_attribute(img, architecture=arch, use_tta=use_tta)
        m_elapsed_ms = round((time.perf_counter() - m_t0) * 1000, 2)
        prob = float(single["probability"])
        pred = int(single["prediction"])
        w = norm_weights[arch]
        weighted_prob += prob * w
        if pred == 1:
            smiling_votes += 1
        per_model.append(
            {
                "architecture": arch,
                "probability": round(prob, 4),
                "prediction": pred,
                "confidence": round(float(single["confidence"]), 4),
                "uncertainty_std": round(float(single["prediction_uncertainty_std"]), 4),
                "slsqp_raw_weight": round(raw_weights[arch], 4),
                "normalized_weight": round(w, 4),
                "checkpoint_loaded": single.get("checkpoint_loaded", ""),
                "latency_ms": m_elapsed_ms,
            }
        )

    consensus_prediction = 1 if weighted_prob >= 0.5 else 0
    majority_votes = max(smiling_votes, len(available) - smiling_votes)
    agreement_ratio = round(majority_votes / len(available), 4)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "source": "pytorch_verified_consensus",
        "use_tta": use_tta,
        "weighting_strategy": weighting_strategy,
        "models_evaluated": len(available),
        "weighted_probability": round(weighted_prob, 4),
        "consensus_prediction": consensus_prediction,
        "agreement_ratio": agreement_ratio,
        "per_model": per_model,
        "latency_ms": elapsed_ms,
        "retained_on_disk": False,
    }
