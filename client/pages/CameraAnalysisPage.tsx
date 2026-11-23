import { useEffect, useRef, useState } from "react";
import { AlertCircle, Camera, CameraOff, Cpu, ShieldCheck, Sparkles, Trash2 } from "lucide-react";
import { GlassCard } from "../components/GlassCard";
import type {
  GeminiAnalysisResult,
  MlPredictionResult,
  ModelsResponse,
  PageId,
  SystemHealthResponse,
} from "../types";

const MODEL_OPTIONS = [
  "Vision Transformer",
  "Spatial BiLSTM",
  "Residual CNN",
  "ANN",
];

interface CameraAnalysisPageProps {
  health?: SystemHealthResponse | null;
  models?: ModelsResponse | null;
  onNavigate?: (page: PageId) => void;
}

export function CameraAnalysisPage({ health, models, onNavigate }: CameraAnalysisPageProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [consentChecked, setConsentChecked] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const [capturedFrame, setCapturedFrame] = useState<string | null>(null);
  const [selectedArchitecture, setSelectedArchitecture] = useState("Vision Transformer");

  const mlBackend = health?.services.pythonMlBackend;
  const backendConnected = Boolean(mlBackend?.reachable && mlBackend?.engineReady);
  const liveVerifiedSet = new Set(
    mlBackend?.verifiedCheckpoints ?? models?.mlBackend.verifiedCheckpoints ?? []
  );
  const diskCheckpointMap = new Map(
    (models?.checkpoints ?? health?.checkpoints ?? []).map((c) => [c.architecture, c])
  );
  const selectedModelReady = backendConnected && liveVerifiedSet.has(selectedArchitecture);

  const [mlLoading, setMlLoading] = useState(false);
  const [mlResult, setMlResult] = useState<MlPredictionResult | null>(null);
  const [mlError, setMlError] = useState<string | null>(null);

  const [geminiLoading, setGeminiLoading] = useState(false);
  const [geminiResult, setGeminiResult] = useState<GeminiAnalysisResult | null>(null);
  const [geminiError, setGeminiError] = useState<string | null>(null);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  };

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    if (!consentChecked) return;
    setCameraError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Browser camera access (getUserMedia) is unavailable in this context.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      setCameraActive(true);
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Camera permission denied or unavailable.";
      setCameraError(msg);
      setCameraActive(false);
    }
  };

  const captureSnapshot = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
    setCapturedFrame(dataUrl);
    setMlResult(null);
    setMlError(null);
    setGeminiResult(null);
    setGeminiError(null);
  };

  const runGeminiSnapshot = async () => {
    if (!capturedFrame) return;
    setGeminiLoading(true);
    setGeminiError(null);
    setGeminiResult(null);
    try {
      const resp = await fetch("/api/gemini/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: capturedFrame, mime_type: "image/jpeg" }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setGeminiError(data.message || `Gemini error (HTTP ${resp.status})`);
      } else {
        setGeminiResult(data as GeminiAnalysisResult);
      }
    } catch (err) {
      setGeminiError(err instanceof Error ? err.message : "Failed to reach Gemini API.");
    } finally {
      setGeminiLoading(false);
    }
  };

  const runMlSnapshot = async () => {
    if (!capturedFrame) return;
    setMlLoading(true);
    setMlError(null);
    setMlResult(null);
    try {
      const resp = await fetch("/api/ml/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: capturedFrame,
          mime_type: "image/jpeg",
          architecture: selectedArchitecture,
          use_tta: true,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setMlError(data.message || `ML backend error (HTTP ${resp.status})`);
      } else {
        setMlResult(data as MlPredictionResult);
      }
    } catch (err) {
      setMlError(err instanceof Error ? err.message : "Failed to reach ML backend.");
    } finally {
      setMlLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-50">Live Camera Analysis</h2>
        <p className="text-sm text-slate-400 mt-1">
          Capture an explicit-consent camera frame for transient in-memory expression and visual
          quality evaluation.
        </p>
      </div>

      {/* Explicit Consent Gate */}
      <GlassCard className="space-y-4">
        <div className="flex items-start gap-3">
          <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
          <div className="space-y-2 flex-1">
            <h3 className="text-base font-semibold text-slate-100">
              Camera Consent & Non-Retention Policy
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              Video frames remain local in your browser until you explicitly click{" "}
              <strong className="text-slate-100">Capture Frame</strong> and trigger analysis.
              Captured frames are never saved to disk or databases, and the studio will not identify
              individuals or infer sensitive personal characteristics.
            </p>
            <label className="flex items-center gap-2.5 text-sm text-slate-200 cursor-pointer pt-1">
              <input
                type="checkbox"
                checked={consentChecked}
                onChange={(e) => {
                  setConsentChecked(e.target.checked);
                  if (!e.target.checked && cameraActive) stopCamera();
                }}
                className="rounded border-white/25 bg-slate-950 text-sky-400 focus:ring-sky-400"
              />
              <span>
                I consent to enabling my camera for transient, non-retained facial attribute analysis.
              </span>
            </label>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-white/10">
          {!cameraActive ? (
            <button
              type="button"
              disabled={!consentChecked}
              onClick={startCamera}
              className="px-4 py-2.5 min-h-[42px] rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-40 disabled:cursor-not-allowed text-slate-950 text-sm font-semibold inline-flex items-center gap-2 transition-colors"
            >
              <Camera className="w-4 h-4" />
              <span>Enable Camera</span>
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={captureSnapshot}
                className="px-4 py-2.5 min-h-[42px] rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 text-sm font-semibold inline-flex items-center gap-2 transition-colors"
              >
                <Camera className="w-4 h-4" />
                <span>Capture Frame</span>
              </button>
              <button
                type="button"
                onClick={stopCamera}
                className="px-4 py-2.5 min-h-[42px] rounded-lg border border-red-400/30 bg-red-500/15 hover:bg-red-500/25 text-red-200 text-sm font-medium inline-flex items-center gap-2 transition-colors"
              >
                <CameraOff className="w-4 h-4" />
                <span>Stop Camera</span>
              </button>
            </>
          )}
        </div>

        {cameraError && (
          <div className="p-3 rounded-lg border border-red-500/40 bg-red-950/30 text-xs text-red-200 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
            <span>{cameraError}</span>
          </div>
        )}
      </GlassCard>

      {/* Camera Feed + Snapshot Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        <GlassCard className="space-y-3">
          <h3 className="text-base font-semibold text-slate-100">Live Video Preview</h3>
          <div className="aspect-video rounded-lg bg-slate-950 border border-white/10 overflow-hidden flex items-center justify-center">
            {cameraActive ? (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="text-sm text-slate-400 px-6 text-center">
                Camera is inactive. Confirm consent and click Enable Camera above.
              </div>
            )}
          </div>
        </GlassCard>

        <GlassCard className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-slate-100">Captured Frame</h3>
            {capturedFrame && (
              <button
                type="button"
                onClick={() => {
                  setCapturedFrame(null);
                  setMlResult(null);
                  setMlError(null);
                  setGeminiResult(null);
                  setGeminiError(null);
                }}
                className="text-xs text-slate-400 hover:text-slate-200 inline-flex items-center gap-1"
              >
                <Trash2 className="w-3.5 h-3.5" /> Clear Frame
              </button>
            )}
          </div>

          <div className="aspect-video rounded-lg bg-slate-950 border border-white/10 overflow-hidden flex items-center justify-center">
            {capturedFrame ? (
              <img
                src={capturedFrame}
                alt="Captured camera snapshot"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-sm text-slate-400 px-6 text-center">
                No frame captured yet.
              </div>
            )}
          </div>

          <div className="flex flex-col sm:flex-row gap-3">
            <select
              aria-label="Select ML architecture for snapshot"
              value={selectedArchitecture}
              onChange={(e) => setSelectedArchitecture(e.target.value)}
              className="flex-1 rounded-lg border border-white/15 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            >
              {MODEL_OPTIONS.map((m) => {
                const live = backendConnected && liveVerifiedSet.has(m);
                const onDisk = diskCheckpointMap.get(m)?.exists ?? false;
                const suffix = live
                  ? " (Ready)"
                  : onDisk
                    ? " (Backend Offline)"
                    : " (Checkpoint Missing)";
                return (
                  <option key={m} value={m} disabled={!live && !onDisk}>
                    {m}
                    {suffix}
                  </option>
                );
              })}
            </select>

            <button
              type="button"
              disabled={!capturedFrame || mlLoading || !selectedModelReady}
              onClick={runMlSnapshot}
              title={
                !backendConnected
                  ? "FastAPI ML backend is disconnected"
                  : !selectedModelReady
                    ? `No verified checkpoint loaded for ${selectedArchitecture}`
                    : "Run live PyTorch inference"
              }
              className="px-4 py-2 min-h-[40px] rounded-lg border border-white/15 bg-white/10 hover:bg-white/15 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-semibold text-slate-100 inline-flex items-center justify-center gap-1.5"
            >
              <Cpu className="w-4 h-4" />
              <span>{mlLoading ? "Running..." : "ML Predict"}</span>
            </button>

            <button
              type="button"
              disabled={!capturedFrame || geminiLoading}
              onClick={runGeminiSnapshot}
              className="px-4 py-2 min-h-[40px] rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-40 text-xs font-semibold text-slate-950 inline-flex items-center justify-center gap-1.5"
            >
              <Sparkles className="w-4 h-4" />
              <span>{geminiLoading ? "Analyzing..." : "Gemini Analyze"}</span>
            </button>
          </div>

          {!backendConnected && (
            <div className="p-2.5 rounded-lg border border-amber-500/30 bg-amber-950/20 text-xs text-amber-200 flex items-center justify-between gap-2">
              <span>
                PyTorch FastAPI backend is offline ({mlBackend?.url ?? "http://127.0.0.1:8001"}).
                ML Predict is disabled until connected.
              </span>
              {onNavigate && (
                <button
                  type="button"
                  onClick={() => onNavigate("system-health")}
                  className="text-sky-300 hover:text-sky-200 underline underline-offset-4 shrink-0"
                >
                  Configure
                </button>
              )}
            </div>
          )}

          {mlError && (
            <div className="p-3 rounded-lg border border-amber-500/40 bg-amber-950/25 text-xs text-amber-200">
              {mlError}
            </div>
          )}

          {mlResult && (
            <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10 text-xs space-y-1">
              <div className="font-semibold text-slate-100">
                ML Result ({mlResult.architecture}):{" "}
                {mlResult.prediction === 1 ? "Smiling" : "Neutral"} (
                {(mlResult.probability * 100).toFixed(2)}%)
              </div>
              <div className="text-slate-400 font-mono">
                Uncertainty ±{mlResult.prediction_uncertainty_std.toFixed(4)} ·{" "}
                {mlResult.tta_views_evaluated} TTA views
              </div>
            </div>
          )}

          {geminiError && (
            <div className="p-3 rounded-lg border border-red-500/40 bg-red-950/25 text-xs text-red-200">
              {geminiError}
            </div>
          )}

          {geminiResult && (
            <div className="p-3.5 rounded-lg bg-slate-950/60 border border-white/10 text-xs space-y-2">
              <div className="font-semibold text-slate-100">
                Expression: {geminiResult.analysis.expression.primary_emotion} (
                {(geminiResult.analysis.expression.confidence * 100).toFixed(1)}%)
              </div>
              <div className="text-slate-300">
                Hair: {geminiResult.analysis.visual_attributes.hair_color} · Eyeglasses:{" "}
                {geminiResult.analysis.visual_attributes.wearing_eyeglasses ? "Yes" : "No"} ·
                Lighting: {geminiResult.analysis.image_quality.lighting}
              </div>
              <div className="text-slate-400 leading-relaxed">
                {geminiResult.analysis.reasoning_summary}
              </div>
            </div>
          )}
        </GlassCard>
      </div>
    </div>
  );
}
