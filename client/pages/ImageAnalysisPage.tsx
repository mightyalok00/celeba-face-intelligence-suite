import { useEffect, useState, type ChangeEvent, type DragEvent } from "react";
import { AlertCircle, CheckCircle2, Cpu, ShieldCheck, Sparkles, Upload } from "lucide-react";
import { GlassCard } from "../components/GlassCard";
import type { GallerySample, GeminiAnalysisResult, MlPredictionResult } from "../types";

const MODEL_OPTIONS = [
  "Vision Transformer",
  "Spatial BiLSTM",
  "Residual CNN",
  "ANN",
  "Single-Fold XGBoost",
  "5-Fold Bagged XGBoost",
  "5-Fold Ensemble + TTA",
];

const ALLOWED_TYPES = new Set(["image/jpeg", "image/jpg", "image/png", "image/webp"]);
const MAX_BYTES = 10 * 1024 * 1024;

export function ImageAnalysisPage() {
  const [selectedFileDataUrl, setSelectedFileDataUrl] = useState<string | null>(null);
  const [selectedMimeType, setSelectedMimeType] = useState<string>("image/jpeg");
  const [selectedFileName, setSelectedFileName] = useState<string>("");
  const [selectedArchitecture, setSelectedArchitecture] = useState<string>("Vision Transformer");
  const [useTta, setUseTta] = useState<boolean>(true);
  const [fileError, setFileError] = useState<string | null>(null);

  const [gallerySamples, setGallerySamples] = useState<GallerySample[]>([]);
  const [mlLoading, setMlLoading] = useState(false);
  const [mlResult, setMlResult] = useState<MlPredictionResult | null>(null);
  const [mlError, setMlError] = useState<string | null>(null);

  const [geminiLoading, setGeminiLoading] = useState(false);
  const [geminiResult, setGeminiResult] = useState<GeminiAnalysisResult | null>(null);
  const [geminiError, setGeminiError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/gallery")
      .then((r) => (r.ok ? r.json() : []))
      .then((data: GallerySample[]) => {
        if (Array.isArray(data)) {
          setGallerySamples(data.slice(0, 8));
        }
      })
      .catch(() => {});
  }, []);

  const processFile = (file: File) => {
    setFileError(null);
    setMlResult(null);
    setMlError(null);
    setGeminiResult(null);
    setGeminiError(null);

    if (!ALLOWED_TYPES.has(file.type.toLowerCase())) {
      setFileError("Unsupported file format. Upload a JPEG, PNG, or WebP image.");
      return;
    }
    if (file.size > MAX_BYTES) {
      setFileError("Image exceeds the 10 MB maximum upload limit.");
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setSelectedFileDataUrl(reader.result);
        setSelectedMimeType(file.type);
        setSelectedFileName(file.name);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    e.target.value = "";
  };

  const handleDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) processFile(file);
  };

  const loadSample = (sample: GallerySample) => {
    setFileError(null);
    setMlResult(null);
    setMlError(null);
    setGeminiResult(null);
    setGeminiError(null);
    setSelectedFileDataUrl(sample.thumbnail);
    setSelectedMimeType("image/jpeg");
    setSelectedFileName(`CelebA Sample #${sample.id}`);
  };

  const runMlInference = async () => {
    if (!selectedFileDataUrl) return;
    setMlLoading(true);
    setMlError(null);
    setMlResult(null);
    try {
      const resp = await fetch("/api/ml/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: selectedFileDataUrl,
          mime_type: selectedMimeType,
          architecture: selectedArchitecture,
          use_tta: useTta,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setMlError(data.message || `Inference service error (HTTP ${resp.status})`);
      } else {
        setMlResult(data as MlPredictionResult);
      }
    } catch (err) {
      setMlError(err instanceof Error ? err.message : "Failed to reach ML inference backend.");
    } finally {
      setMlLoading(false);
    }
  };

  const runGeminiAnalysis = async () => {
    if (!selectedFileDataUrl) return;
    setGeminiLoading(true);
    setGeminiError(null);
    setGeminiResult(null);
    try {
      const resp = await fetch("/api/gemini/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: selectedFileDataUrl,
          mime_type: selectedMimeType,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setGeminiError(data.message || `Gemini analysis error (HTTP ${resp.status})`);
      } else {
        setGeminiResult(data as GeminiAnalysisResult);
      }
    } catch (err) {
      setGeminiError(err instanceof Error ? err.message : "Failed to reach Gemini endpoint.");
    } finally {
      setGeminiLoading(false);
    }
  };

  const clearImage = () => {
    setSelectedFileDataUrl(null);
    setSelectedFileName("");
    setFileError(null);
    setMlResult(null);
    setMlError(null);
    setGeminiResult(null);
    setGeminiError(null);
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-50">Facial Attribute & Multimodal Image Analysis</h2>
        <p className="text-sm text-slate-400 mt-1">
          Upload an image or select a reference CelebA sample to execute PyTorch/XGBoost inference and
          privacy-preserving Gemini visual analysis.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Upload & Controls */}
        <GlassCard className="lg:col-span-5 space-y-5">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-slate-100">Image Input</h3>
            {selectedFileDataUrl && (
              <button
                type="button"
                onClick={clearImage}
                className="text-xs font-medium text-slate-400 hover:text-slate-200 underline underline-offset-4"
              >
                Clear Image
              </button>
            )}
          </div>

          <label
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="block rounded-xl border border-dashed border-white/20 hover:border-sky-400/60 bg-slate-950/60 p-6 text-center cursor-pointer transition-colors"
          >
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
              className="sr-only"
            />
            {selectedFileDataUrl ? (
              <div className="space-y-3">
                <img
                  src={selectedFileDataUrl}
                  alt="Selected upload preview"
                  referrerPolicy="no-referrer"
                  className="mx-auto max-h-64 rounded-lg object-contain border border-white/10"
                />
                <div className="text-xs text-slate-300 font-mono truncate">{selectedFileName}</div>
              </div>
            ) : (
              <div className="space-y-2 py-4">
                <Upload className="w-8 h-8 mx-auto text-sky-400" />
                <div className="text-sm font-medium text-slate-200">
                  Drop an image here or click to browse
                </div>
                <div className="text-xs text-slate-400">
                  JPEG, PNG, or WebP · Max 10 MB · Magic-byte verified
                </div>
              </div>
            )}
          </label>

          {fileError && (
            <div className="p-3 rounded-lg border border-red-500/40 bg-red-950/30 text-xs text-red-200 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
              <span>{fileError}</span>
            </div>
          )}

          {/* Model & TTA Controls */}
          <div className="space-y-3 pt-2 border-t border-white/10">
            <div>
              <label htmlFor="archSelect" className="block text-xs font-medium text-slate-300 mb-1.5">
                ML Architecture
              </label>
              <select
                id="archSelect"
                value={selectedArchitecture}
                onChange={(e) => setSelectedArchitecture(e.target.value)}
                className="w-full rounded-lg border border-white/15 bg-slate-950 px-3 py-2.5 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-400"
              >
                {MODEL_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            </div>

            <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={useTta}
                onChange={(e) => setUseTta(e.target.checked)}
                className="rounded border-white/20 bg-slate-950 text-sky-400 focus:ring-sky-400"
              />
              <span>Enable 5-View Test-Time Augmentation (TTA)</span>
            </label>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
              <button
                type="button"
                disabled={!selectedFileDataUrl || mlLoading}
                onClick={runMlInference}
                className="px-4 py-2.5 min-h-[42px] rounded-lg border border-white/15 bg-white/10 hover:bg-white/15 disabled:opacity-40 disabled:cursor-not-allowed text-slate-100 text-xs sm:text-sm font-semibold transition-colors inline-flex items-center justify-center gap-2"
              >
                <Cpu className="w-4 h-4" />
                <span>{mlLoading ? "Running ML..." : "Run ML Model"}</span>
              </button>

              <button
                type="button"
                disabled={!selectedFileDataUrl || geminiLoading}
                onClick={runGeminiAnalysis}
                className="px-4 py-2.5 min-h-[42px] rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-40 disabled:cursor-not-allowed text-slate-950 text-xs sm:text-sm font-semibold transition-colors inline-flex items-center justify-center gap-2"
              >
                <Sparkles className="w-4 h-4" />
                <span>{geminiLoading ? "Analyzing..." : "Gemini Analysis"}</span>
              </button>
            </div>
          </div>

          <div className="flex items-start gap-2.5 text-xs text-slate-400 pt-2 border-t border-white/10">
            <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span>
              Zero disk retention. Analysis inspects expression and visual attributes only—no personal
              identification or sensitive characteristic inference.
            </span>
          </div>

          {/* Reference Gallery Picker */}
          {gallerySamples.length > 0 && (
            <div className="pt-2 border-t border-white/10 space-y-2">
              <div className="text-xs font-medium text-slate-300">
                Or test with a curated CelebA reference sample
              </div>
              <div className="grid grid-cols-4 gap-2">
                {gallerySamples.map((sample) => (
                  <button
                    key={sample.id}
                    type="button"
                    onClick={() => loadSample(sample)}
                    className="rounded-lg overflow-hidden border border-white/10 hover:border-sky-400 transition-colors aspect-square bg-slate-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
                    title={`Load CelebA sample #${sample.id}`}
                  >
                    <img
                      src={sample.thumbnail}
                      alt={`CelebA sample ${sample.id}`}
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover"
                    />
                  </button>
                ))}
              </div>
            </div>
          )}
        </GlassCard>

        {/* Right: Results Panels */}
        <div className="lg:col-span-7 space-y-6">
          {/* ML Backend Result / Honest Error */}
          <GlassCard className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-slate-100">
                PyTorch / XGBoost Inference Output
              </h3>
              <span className="text-xs font-mono text-slate-400">{selectedArchitecture}</span>
            </div>

            {mlLoading && (
              <div className="py-6 text-sm text-slate-300">
                Dispatching validated image buffer to Python FastAPI inference service...
              </div>
            )}

            {mlError && (
              <div className="p-4 rounded-lg border border-amber-500/40 bg-amber-950/25 text-xs sm:text-sm text-amber-200 space-y-1.5">
                <div className="font-semibold flex items-center gap-2 text-amber-300">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>ML Inference Backend Unavailable</span>
                </div>
                <p className="text-amber-100/90 leading-relaxed">{mlError}</p>
              </div>
            )}

            {mlResult && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
                    <div className="text-xs text-slate-400">Prediction</div>
                    <div className="text-lg font-bold text-slate-50 mt-1">
                      {mlResult.prediction === 1 ? "Smiling" : "Neutral"}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
                    <div className="text-xs text-slate-400">Probability</div>
                    <div className="text-lg font-bold text-sky-300 mt-1 font-mono tabular-nums">
                      {(mlResult.probability * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
                    <div className="text-xs text-slate-400">TTA Uncertainty</div>
                    <div className="text-lg font-bold text-slate-100 mt-1 font-mono tabular-nums">
                      ±{mlResult.prediction_uncertainty_std.toFixed(4)}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
                    <div className="text-xs text-slate-400">Latency</div>
                    <div className="text-lg font-bold text-slate-100 mt-1 font-mono tabular-nums">
                      {mlResult.latency_ms ?? 0} ms
                    </div>
                  </div>
                </div>
                <div className="text-xs text-slate-400 font-mono">
                  Checkpoint: {mlResult.checkpoint_loaded} · TTA Views: {mlResult.tta_views_evaluated}
                </div>
              </div>
            )}

            {!mlLoading && !mlError && !mlResult && (
              <p className="text-sm text-slate-400">
                Select an image and click <strong className="text-slate-200">Run ML Model</strong> to
                query the FastAPI PyTorch/XGBoost service.
              </p>
            )}
          </GlassCard>

          {/* Gemini Multimodal Output */}
          <GlassCard className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-slate-100">
                Gemini Multimodal Visual Assessment
              </h3>
              {geminiResult && (
                <span className="text-xs font-mono text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> {geminiResult.model} ·{" "}
                  {geminiResult.latency_ms} ms
                </span>
              )}
            </div>

            {geminiLoading && (
              <div className="py-6 text-sm text-slate-300">
                Running server-side Gemini multimodal visual attribute & quality extraction...
              </div>
            )}

            {geminiError && (
              <div className="p-4 rounded-lg border border-red-500/40 bg-red-950/25 text-xs sm:text-sm text-red-200 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 text-red-400 mt-0.5" />
                <span>{geminiError}</span>
              </div>
            )}

            {geminiResult && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="p-3.5 rounded-lg bg-slate-950/60 border border-white/10">
                    <div className="text-xs text-slate-400">Expression</div>
                    <div className="text-base font-semibold text-slate-100 mt-1">
                      {geminiResult.analysis.expression.primary_emotion}
                    </div>
                    <div className="text-xs text-sky-400 font-mono tabular-nums mt-0.5">
                      Confidence: {(geminiResult.analysis.expression.confidence * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="p-3.5 rounded-lg bg-slate-950/60 border border-white/10">
                    <div className="text-xs text-slate-400">Visual Attributes</div>
                    <div className="text-sm text-slate-100 mt-1">
                      Hair: {geminiResult.analysis.visual_attributes.hair_color}
                    </div>
                    <div className="text-xs text-slate-400 mt-0.5">
                      Eyeglasses:{" "}
                      {geminiResult.analysis.visual_attributes.wearing_eyeglasses ? "Yes" : "No"} ·
                      Facial Hair:{" "}
                      {geminiResult.analysis.visual_attributes.facial_hair ? "Yes" : "No"}
                    </div>
                  </div>

                  <div className="p-3.5 rounded-lg bg-slate-950/60 border border-white/10">
                    <div className="text-xs text-slate-400">Capture Quality</div>
                    <div className="text-sm text-slate-100 mt-1">
                      {geminiResult.analysis.image_quality.lighting}
                    </div>
                    <div className="text-xs text-slate-400 mt-0.5">
                      Framing: {geminiResult.analysis.image_quality.framing} · Occlusion:{" "}
                      {geminiResult.analysis.image_quality.occlusion_level}
                    </div>
                  </div>
                </div>

                {geminiResult.analysis.visual_attributes.notable_visual_cues.length > 0 && (
                  <div className="text-xs text-slate-300">
                    <span className="text-slate-400 font-medium">Notable visual cues: </span>
                    {geminiResult.analysis.visual_attributes.notable_visual_cues.join(" · ")}
                  </div>
                )}

                <div className="p-3.5 rounded-lg bg-slate-950/60 border border-white/10 text-sm text-slate-200 leading-relaxed">
                  {geminiResult.analysis.reasoning_summary}
                </div>

                <div className="text-xs text-slate-400">{geminiResult.privacy_notice}</div>
              </div>
            )}

            {!geminiLoading && !geminiError && !geminiResult && (
              <p className="text-sm text-slate-400">
                Click <strong className="text-slate-200">Gemini Analysis</strong> to evaluate facial
                expression, accessories, and photographic quality via the server-side Gemini API.
              </p>
            )}
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
