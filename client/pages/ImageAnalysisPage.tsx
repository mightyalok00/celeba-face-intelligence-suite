import { useEffect, useState, type ChangeEvent, type DragEvent } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Download,
  GitCompare,
  ShieldCheck,
  Sparkles,
  SlidersHorizontal,
  Upload,
} from "lucide-react";
import { GlassCard } from "../components/GlassCard";
import type {
  GallerySample,
  GeminiAnalysisResult,
  MlConsensusResult,
  MlPredictionResult,
} from "../types";

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
  const [thresholdPct, setThresholdPct] = useState<number>(50);
  const [gridOpacityPct, setGridOpacityPct] = useState<number>(35);
  const [brightnessPct, setBrightnessPct] = useState<number>(100);
  const [fileError, setFileError] = useState<string | null>(null);

  const [gallerySamples, setGallerySamples] = useState<GallerySample[]>([]);
  const [mlLoading, setMlLoading] = useState(false);
  const [mlResult, setMlResult] = useState<MlPredictionResult | null>(null);
  const [mlError, setMlError] = useState<string | null>(null);

  const [consensusLoading, setConsensusLoading] = useState(false);
  const [consensusResult, setConsensusResult] = useState<MlConsensusResult | null>(null);
  const [consensusError, setConsensusError] = useState<string | null>(null);

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

  const resetOutputs = () => {
    setFileError(null);
    setMlResult(null);
    setMlError(null);
    setConsensusResult(null);
    setConsensusError(null);
    setGeminiResult(null);
    setGeminiError(null);
  };

  const processFile = (file: File) => {
    resetOutputs();

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
    resetOutputs();
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

  const runConsensusCheck = async () => {
    if (!selectedFileDataUrl) return;
    setConsensusLoading(true);
    setConsensusError(null);
    setConsensusResult(null);
    try {
      const resp = await fetch("/api/ml/consensus", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: selectedFileDataUrl,
          mime_type: selectedMimeType,
          use_tta: useTta,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setConsensusError(data.message || `Consensus service error (HTTP ${resp.status})`);
      } else {
        setConsensusResult(data as MlConsensusResult);
      }
    } catch (err) {
      setConsensusError(
        err instanceof Error ? err.message : "Failed to reach ML consensus endpoint."
      );
    } finally {
      setConsensusLoading(false);
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

  const downloadSessionReport = (format: "json" | "md") => {
    if (!mlResult && !consensusResult && !geminiResult) return;

    const payload = {
      generatedAt: new Date().toISOString(),
      inputLabel: selectedFileName || "In-memory upload",
      decisionThresholdPct: thresholdPct,
      ttaEnabled: useTta,
      privacyNote: "Zero image bytes stored or included in this report.",
      singleModelInference: mlResult,
      multiArchitectureConsensus: consensusResult,
      geminiMultimodalAssessment: geminiResult,
    };

    let content = "";
    let mime = "application/json;charset=utf-8";
    let ext = "json";

    if (format === "json") {
      content = JSON.stringify(payload, null, 2);
    } else {
      mime = "text/markdown;charset=utf-8";
      ext = "md";
      const lines = [
        "# CelebA AI — Session Analysis Report",
        `- **Generated:** ${payload.generatedAt}`,
        `- **Input:** ${payload.inputLabel}`,
        `- **Decision Threshold:** ${thresholdPct}%`,
        `- **Privacy Notice:** ${payload.privacyNote}`,
        "",
      ];
      if (mlResult) {
        lines.push(
          "## Single-Model PyTorch Inference",
          `- **Architecture:** ${mlResult.architecture}`,
          `- **Verdict:** ${mlResult.probability * 100 >= thresholdPct ? "Smiling" : "Neutral"}`,
          `- **Probability:** ${(mlResult.probability * 100).toFixed(2)}%`,
          `- **TTA Std:** ±${mlResult.prediction_uncertainty_std.toFixed(4)}`,
          ""
        );
      }
      if (consensusResult) {
        lines.push(
          "## Multi-Architecture Consensus",
          `- **Strategy:** ${consensusResult.weighting_strategy}`,
          `- **Weighted Probability:** ${(consensusResult.weighted_probability * 100).toFixed(2)}%`,
          `- **Agreement Ratio:** ${(consensusResult.agreement_ratio * 100).toFixed(1)}%`,
          ""
        );
      }
      if (geminiResult) {
        lines.push(
          "## Gemini Multimodal Assessment",
          `- **Model:** ${geminiResult.model}`,
          `- **Primary Emotion:** ${geminiResult.analysis.expression.primary_emotion} (${(geminiResult.analysis.expression.confidence * 100).toFixed(1)}%)`,
          `- **Hair Color:** ${geminiResult.analysis.visual_attributes.hair_color}`,
          `- **Eyeglasses / Facial Hair:** ${geminiResult.analysis.visual_attributes.wearing_eyeglasses ? "Yes" : "No"} / ${geminiResult.analysis.visual_attributes.facial_hair ? "Yes" : "No"}`,
          `- **Lighting / Framing / Occlusion:** ${geminiResult.analysis.image_quality.lighting} · ${geminiResult.analysis.image_quality.framing} · ${geminiResult.analysis.image_quality.occlusion_level}`,
          `- **Summary:** ${geminiResult.analysis.reasoning_summary}`,
          ""
        );
      }
      content = lines.join("\n");
    }

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `celeba_analysis_report.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const clearImage = () => {
    setSelectedFileDataUrl(null);
    setSelectedFileName("");
    resetOutputs();
  };

  const hasAnyResult = Boolean(mlResult || consensusResult || geminiResult);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-slate-50">
            Facial Attribute & Multimodal Image Analysis
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Upload an image or select a reference CelebA sample to execute PyTorch inference,
            multi-architecture consensus, and privacy-preserving Gemini analysis.
          </p>
        </div>

        {hasAnyResult && (
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => downloadSessionReport("md")}
              className="px-3 py-2 min-h-[38px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Report (MD)</span>
            </button>
            <button
              type="button"
              onClick={() => downloadSessionReport("json")}
              className="px-3 py-2 min-h-[38px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export JSON</span>
            </button>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Upload & Controls */}
        <GlassCard className="lg:col-span-5 space-y-5">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-slate-100">Image Input & Calibration</h3>
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
                <div className="relative inline-block mx-auto rounded-lg overflow-hidden border border-white/10 bg-slate-950">
                  <img
                    src={selectedFileDataUrl}
                    alt="Selected upload preview"
                    referrerPolicy="no-referrer"
                    style={{ filter: `brightness(${brightnessPct}%)` }}
                    className="max-h-64 object-contain block"
                  />
                  {/* Interactive Facial Alignment Inspection Overlay */}
                  {gridOpacityPct > 0 && (
                    <svg
                      viewBox="0 0 200 200"
                      preserveAspectRatio="none"
                      aria-hidden="true"
                      style={{ opacity: gridOpacityPct / 100 }}
                      className="absolute inset-0 w-full h-full pointer-events-none"
                    >
                      <line x1="66" y1="0" x2="66" y2="200" stroke="#38BDF8" strokeWidth="0.6" strokeDasharray="3 3" />
                      <line x1="134" y1="0" x2="134" y2="200" stroke="#38BDF8" strokeWidth="0.6" strokeDasharray="3 3" />
                      <line x1="0" y1="66" x2="200" y2="66" stroke="#38BDF8" strokeWidth="0.6" strokeDasharray="3 3" />
                      <line x1="0" y1="134" x2="200" y2="134" stroke="#38BDF8" strokeWidth="0.6" strokeDasharray="3 3" />
                      <ellipse cx="100" cy="96" rx="48" ry="64" fill="none" stroke="#38BDF8" strokeWidth="0.9" />
                      <line x1="100" y1="24" x2="100" y2="176" stroke="#34D399" strokeWidth="0.6" />
                      <line x1="56" y1="84" x2="144" y2="84" stroke="#34D399" strokeWidth="0.6" />
                    </svg>
                  )}
                </div>
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

          {/* Visual Inspection & Calibration Sliders */}
          <div className="rounded-lg border border-white/10 bg-slate-950/60 p-3.5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200 inline-flex items-center gap-1.5">
                <SlidersHorizontal className="w-3.5 h-3.5 text-sky-400" />
                Visual Inspection Sliders
              </span>
              {(gridOpacityPct !== 35 || brightnessPct !== 100) && (
                <button
                  type="button"
                  onClick={() => {
                    setGridOpacityPct(35);
                    setBrightnessPct(100);
                  }}
                  className="text-[11px] text-sky-400 hover:text-sky-300"
                >
                  Reset
                </button>
              )}
            </div>

            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="gridOpacitySlider" className="text-slate-300">
                  Alignment Grid Overlay
                </label>
                <span className="font-mono text-sky-300 tabular-nums">{gridOpacityPct}%</span>
              </div>
              <input
                id="gridOpacitySlider"
                type="range"
                min={0}
                max={100}
                step={5}
                value={gridOpacityPct}
                onChange={(e) => setGridOpacityPct(Number(e.target.value))}
                className="w-full accent-sky-400 cursor-pointer"
              />
            </div>

            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="brightnessSlider" className="text-slate-300">
                  Preview Exposure / Brightness
                </label>
                <span className="font-mono text-sky-300 tabular-nums">{brightnessPct}%</span>
              </div>
              <input
                id="brightnessSlider"
                type="range"
                min={75}
                max={125}
                step={5}
                value={brightnessPct}
                onChange={(e) => setBrightnessPct(Number(e.target.value))}
                className="w-full accent-sky-400 cursor-pointer"
              />
            </div>
          </div>

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

            <div className="rounded-lg border border-white/10 bg-slate-950/60 p-3 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="decisionThresholdSlider" className="font-medium text-slate-300">
                  Decision Threshold
                </label>
                <span className="font-mono font-semibold text-sky-300 tabular-nums">
                  {thresholdPct}%
                </span>
              </div>
              <input
                id="decisionThresholdSlider"
                type="range"
                min={50}
                max={95}
                step={5}
                value={thresholdPct}
                onChange={(e) => setThresholdPct(Number(e.target.value))}
                className="w-full accent-sky-400 cursor-pointer"
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
              <button
                type="button"
                disabled={!selectedFileDataUrl || mlLoading}
                onClick={runMlInference}
                className="px-4 py-2.5 min-h-[42px] rounded-lg border border-white/15 bg-white/10 hover:bg-white/15 disabled:opacity-40 disabled:cursor-not-allowed text-slate-100 text-xs sm:text-sm font-semibold transition-colors inline-flex items-center justify-center gap-2 whitespace-nowrap"
              >
                <Cpu className="w-4 h-4" />
                <span>{mlLoading ? "Running ML..." : "Run ML Model"}</span>
              </button>

              <button
                type="button"
                disabled={!selectedFileDataUrl || geminiLoading}
                onClick={runGeminiAnalysis}
                className="px-4 py-2.5 min-h-[42px] rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-40 disabled:cursor-not-allowed text-slate-950 text-xs sm:text-sm font-semibold transition-colors inline-flex items-center justify-center gap-2 whitespace-nowrap"
              >
                <Sparkles className="w-4 h-4" />
                <span>{geminiLoading ? "Analyzing..." : "Gemini Analysis"}</span>
              </button>
            </div>

            <button
              type="button"
              disabled={!selectedFileDataUrl || consensusLoading}
              onClick={runConsensusCheck}
              className="w-full px-4 py-2.5 min-h-[42px] rounded-lg border border-sky-400/30 bg-sky-500/10 hover:bg-sky-500/20 disabled:opacity-40 disabled:cursor-not-allowed text-sky-200 text-xs sm:text-sm font-semibold transition-colors inline-flex items-center justify-center gap-2"
            >
              <GitCompare className="w-4 h-4" />
              <span>
                {consensusLoading
                  ? "Running Multi-Model Consensus..."
                  : "Run Multi-Model Consensus Check"}
              </span>
            </button>
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
                    <div className="text-xs text-slate-400">Verdict (&ge;{thresholdPct}%)</div>
                    <div className="text-lg font-bold text-slate-50 mt-1">
                      {mlResult.probability * 100 >= thresholdPct ? "Smiling" : "Neutral"}
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

          {/* Multi-Architecture Consensus Panel */}
          {(consensusLoading || consensusError || consensusResult) && (
            <GlassCard className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-base font-semibold text-slate-100">
                  Multi-Architecture Consensus Evaluation
                </h3>
                {consensusResult && (
                  <span className="text-xs font-mono text-sky-300 tabular-nums">
                    {consensusResult.models_evaluated} models · {consensusResult.latency_ms ?? 0} ms
                  </span>
                )}
              </div>

              {consensusLoading && (
                <div className="py-4 text-sm text-slate-300">
                  Evaluating verified checkpoints and aggregating SLSQP ensemble consensus...
                </div>
              )}

              {consensusError && (
                <div className="p-4 rounded-lg border border-amber-500/40 bg-amber-950/25 text-xs sm:text-sm text-amber-200 space-y-1.5">
                  <div className="font-semibold flex items-center gap-2 text-amber-300">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>Consensus Backend Unavailable</span>
                  </div>
                  <p className="text-amber-100/90 leading-relaxed">{consensusError}</p>
                </div>
              )}

              {consensusResult && (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-3">
                    <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
                      <div className="text-xs text-slate-400">Consensus (&ge;{thresholdPct}%)</div>
                      <div className="text-lg font-bold text-slate-50 mt-1">
                        {consensusResult.weighted_probability * 100 >= thresholdPct
                          ? "Smiling"
                          : "Neutral"}
                      </div>
                    </div>
                    <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
                      <div className="text-xs text-slate-400">Weighted Probability</div>
                      <div className="text-lg font-bold text-sky-300 mt-1 font-mono tabular-nums">
                        {(consensusResult.weighted_probability * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
                      <div className="text-xs text-slate-400">Model Agreement</div>
                      <div className="text-lg font-bold text-emerald-300 mt-1 font-mono tabular-nums">
                        {(consensusResult.agreement_ratio * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-white/10 text-slate-400">
                          <th className="py-2 pr-3 font-medium">Architecture</th>
                          <th className="py-2 px-3 font-medium text-right">Probability</th>
                          <th className="py-2 px-3 font-medium text-right">Normalized Weight</th>
                          <th className="py-2 pl-3 font-medium text-right">Uncertainty</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/5">
                        {consensusResult.per_model.map((m) => (
                          <tr key={m.architecture}>
                            <td className="py-2 pr-3 font-medium text-slate-100">{m.architecture}</td>
                            <td className="py-2 px-3 text-right font-mono tabular-nums text-sky-300">
                              {(m.probability * 100).toFixed(2)}%
                            </td>
                            <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-300">
                              {(m.normalized_weight * 100).toFixed(1)}%
                            </td>
                            <td className="py-2 pl-3 text-right font-mono tabular-nums text-slate-400">
                              ±{m.uncertainty_std.toFixed(4)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </GlassCard>
          )}

          {/* Gemini Multimodal Output */}
          <GlassCard className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-slate-100">
                Gemini Multimodal Visual Assessment
              </h3>
              {geminiResult && (
                <span className="text-xs font-mono text-emerald-400 flex items-center gap-1 tabular-nums">
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
