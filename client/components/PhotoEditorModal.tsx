import { useState, useRef, useEffect, useCallback } from "react";
import {
  X,
  RotateCw,
  FlipHorizontal,
  Sliders,
  Sparkles,
  Download,
  Undo2,
  Redo2,
  RotateCcw,
  Eye,
  Save,
} from "lucide-react";

interface PhotoEditorModalProps {
  imageUrl: string;
  filename: string;
  onClose: () => void;
  onSaveEdited: (newUrl: string, newFilename: string) => void;
}

type FilterPreset = "normal" | "vivid" | "cinematic" | "bw" | "warm" | "cool";
type ExportFormat = "image/jpeg" | "image/png" | "image/webp";

interface EditState {
  brightness: number; // 0 to 200, default 100
  contrast: number; // 0 to 200, default 100
  saturation: number; // 0 to 200, default 100
  exposure: number; // -100 to 100, default 0
  sharpness: number; // 0 to 100, default 0
  rotation: number; // 0, 90, 180, 270
  flipH: boolean;
  filter: FilterPreset;
}

interface HistoryStore {
  entries: EditState[];
  index: number;
}

const MAX_HISTORY_ENTRIES = 60;

const DEFAULT_STATE: EditState = {
  brightness: 100,
  contrast: 100,
  saturation: 100,
  exposure: 0,
  sharpness: 0,
  rotation: 0,
  flipH: false,
  filter: "normal",
};

const FILTER_PRESETS: Array<{ id: FilterPreset; label: string }> = [
  { id: "normal", label: "Natural" },
  { id: "vivid", label: "Vivid" },
  { id: "cinematic", label: "Cinematic" },
  { id: "bw", label: "B & W" },
  { id: "warm", label: "Warm Glow" },
  { id: "cool", label: "Cool Tone" },
];

/**
 * Applies a 3x3 sharpening convolution kernel in-place on a 2D canvas context.
 */
function applyCanvasSharpen(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  amountPct: number
): void {
  if (amountPct <= 0 || width < 3 || height < 3) return;
  const strength = (amountPct / 100) * 1.2;
  const imgData = ctx.getImageData(0, 0, width, height);
  const src = new Uint8ClampedArray(imgData.data);
  const dst = imgData.data;

  for (let y = 1; y < height - 1; y += 1) {
    const rowOffset = y * width;
    for (let x = 1; x < width - 1; x += 1) {
      const idx = (rowOffset + x) * 4;
      const up = ((y - 1) * width + x) * 4;
      const down = ((y + 1) * width + x) * 4;
      const left = idx - 4;
      const right = idx + 4;

      for (let c = 0; c < 3; c += 1) {
        const center = src[idx + c];
        const neighbors =
          (src[up + c] + src[down + c] + src[left + c] + src[right + c]) * 0.25;
        const sharpened = center + (center - neighbors) * strength;
        dst[idx + c] = sharpened < 0 ? 0 : sharpened > 255 ? 255 : sharpened;
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);
}

export function PhotoEditorModal({
  imageUrl,
  filename,
  onClose,
  onSaveEdited,
}: PhotoEditorModalProps) {
  const [historyStore, setHistoryStore] = useState<HistoryStore>({
    entries: [DEFAULT_STATE],
    index: 0,
  });
  const [activeTab, setActiveTab] = useState<"adjust" | "filters" | "transform">("adjust");
  const [showBeforeAfter, setShowBeforeAfter] = useState(false);
  const [exportFormat, setExportFormat] = useState<ExportFormat>("image/jpeg");
  const [exportQuality, setExportQuality] = useState<number>(0.92);
  const [exportError, setExportError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageElement, setImageElement] = useState<HTMLImageElement | null>(null);

  const state = historyStore.entries[historyStore.index] ?? DEFAULT_STATE;
  const canUndo = historyStore.index > 0;
  const canRedo = historyStore.index < historyStore.entries.length - 1;

  // Load source image with clean teardown
  useEffect(() => {
    let active = true;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (active) setImageElement(img);
    };
    img.onerror = () => {
      if (active) setImageElement(null);
    };
    img.src = imageUrl;
    return () => {
      active = false;
      img.onload = null;
      img.onerror = null;
    };
  }, [imageUrl]);

  // Atomic history update (StrictMode-safe, zero stale closures)
  const updateState = useCallback((updater: (prev: EditState) => EditState) => {
    setHistoryStore((prev) => {
      const current = prev.entries[prev.index] ?? DEFAULT_STATE;
      const next = updater(current);
      const sliced = prev.entries.slice(0, prev.index + 1);
      sliced.push(next);
      if (sliced.length > MAX_HISTORY_ENTRIES) {
        sliced.shift();
      }
      return {
        entries: sliced,
        index: sliced.length - 1,
      };
    });
  }, []);

  const handleUndo = useCallback(() => {
    setHistoryStore((prev) =>
      prev.index > 0 ? { ...prev, index: prev.index - 1 } : prev
    );
  }, []);

  const handleRedo = useCallback(() => {
    setHistoryStore((prev) =>
      prev.index < prev.entries.length - 1 ? { ...prev, index: prev.index + 1 } : prev
    );
  }, []);

  const handleReset = useCallback(() => {
    setHistoryStore({ entries: [DEFAULT_STATE], index: 0 });
  }, []);

  // Keyboard accessibility: Esc closes, Ctrl/Cmd+Z undoes, Ctrl/Cmd+Shift+Z or Ctrl+Y redoes
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) handleRedo();
        else handleUndo();
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "y") {
        e.preventDefault();
        handleRedo();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, handleUndo, handleRedo]);

  // Render to canvas with filters, clamped exposure, and 3x3 sharpening convolution
  const renderCanvas = useCallback(
    (targetCanvas: HTMLCanvasElement | null, overrideState?: EditState) => {
      if (!targetCanvas || !imageElement) return;
      const ctx = targetCanvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;

      const current = overrideState ?? state;
      const rot = current.rotation;
      const isRotated90or270 = rot === 90 || rot === 270;

      const w = isRotated90or270 ? imageElement.height : imageElement.width;
      const h = isRotated90or270 ? imageElement.width : imageElement.height;

      targetCanvas.width = w;
      targetCanvas.height = h;

      ctx.save();
      ctx.clearRect(0, 0, w, h);

      ctx.translate(w / 2, h / 2);
      ctx.rotate((rot * Math.PI) / 180);
      if (current.flipH) {
        ctx.scale(-1, 1);
      }

      const effectiveBrightness = Math.max(0, current.brightness + current.exposure);
      let filterString = `brightness(${effectiveBrightness}%) contrast(${current.contrast}%) saturate(${current.saturation}%)`;

      if (current.filter === "vivid") {
        filterString += " saturate(140%) contrast(110%)";
      } else if (current.filter === "cinematic") {
        filterString += " contrast(120%) brightness(95%) sepia(20%)";
      } else if (current.filter === "bw") {
        filterString += " grayscale(100%) contrast(110%)";
      } else if (current.filter === "warm") {
        filterString += " sepia(30%) saturate(120%)";
      } else if (current.filter === "cool") {
        filterString += " hue-rotate(180deg) saturate(90%)";
      }

      ctx.filter = filterString;
      ctx.drawImage(imageElement, -imageElement.width / 2, -imageElement.height / 2);
      ctx.restore();

      if (current.sharpness > 0) {
        try {
          applyCanvasSharpen(ctx, w, h, current.sharpness);
        } catch {
          // Cross-origin tainted canvas safely falls back to CSS filter rendering
        }
      }
    },
    [imageElement, state]
  );

  useEffect(() => {
    renderCanvas(canvasRef.current, showBeforeAfter ? DEFAULT_STATE : state);
  }, [imageElement, state, showBeforeAfter, renderCanvas]);

  const buildExportPayload = (): { dataUrl: string; editedName: string } | null => {
    if (!canvasRef.current) return null;
    try {
      setExportError(null);
      renderCanvas(canvasRef.current, state);
      const dataUrl = canvasRef.current.toDataURL(exportFormat, exportQuality);
      const ext =
        exportFormat === "image/png" ? "png" : exportFormat === "image/webp" ? "webp" : "jpg";
      const editedName = filename.replace(/\.[^/.]+$/, "") + `_edited.${ext}`;
      return { dataUrl, editedName };
    } catch {
      setExportError("Could not export edited image due to browser canvas restrictions.");
      return null;
    }
  };

  const handleSaveToVault = () => {
    const payload = buildExportPayload();
    if (!payload) return;
    onSaveEdited(payload.dataUrl, payload.editedName);
    onClose();
  };

  const handleDownloadOnly = () => {
    const payload = buildExportPayload();
    if (!payload) return;
    const link = document.createElement("a");
    link.href = payload.dataUrl;
    link.download = payload.editedName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Photo editor for ${filename}`}
      className="fixed inset-0 z-50 bg-black/95 backdrop-blur-xl flex flex-col items-center justify-between p-3 sm:p-6 overflow-y-auto lg:overflow-hidden animate-fade-in"
      onClick={onClose}
    >
      {/* Editor Header */}
      <div
        className="w-full max-w-7xl flex flex-wrap items-center justify-between gap-3 text-slate-100 border-b border-white/10 pb-3 sm:pb-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 min-w-0">
          <Sliders className="w-5 h-5 text-sky-400 shrink-0" aria-hidden="true" />
          <div className="min-w-0">
            <h2 className="text-sm sm:text-base font-bold font-mono truncate">{filename}</h2>
            <p className="text-xs text-slate-400">
              Non-Destructive Canvas Editor · Offline Ready
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleUndo}
            disabled={!canUndo}
            aria-label="Undo edit (Ctrl+Z)"
            className="p-2 min-w-[40px] min-h-[40px] flex items-center justify-center rounded-lg bg-white/10 hover:bg-white/20 disabled:opacity-40 text-slate-200 transition-colors"
            title="Undo (Ctrl+Z)"
          >
            <Undo2 className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={handleRedo}
            disabled={!canRedo}
            aria-label="Redo edit (Ctrl+Y)"
            className="p-2 min-w-[40px] min-h-[40px] flex items-center justify-center rounded-lg bg-white/10 hover:bg-white/20 disabled:opacity-40 text-slate-200 transition-colors"
            title="Redo (Ctrl+Y)"
          >
            <Redo2 className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={handleReset}
            aria-label="Reset all adjustments"
            className="p-2 min-w-[40px] min-h-[40px] flex items-center justify-center rounded-lg bg-white/10 hover:bg-white/20 text-slate-200 transition-colors"
            title="Reset All"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <div className="hidden sm:block h-5 w-[1px] bg-white/15 mx-1" />

          <select
            value={exportFormat}
            aria-label="Export image format"
            onChange={(e) => setExportFormat(e.target.value as ExportFormat)}
            className="px-2.5 py-2 min-h-[40px] rounded-lg bg-slate-900 border border-white/15 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400"
          >
            <option value="image/jpeg">JPEG</option>
            <option value="image/png">PNG</option>
            <option value="image/webp">WebP</option>
          </select>

          {exportFormat !== "image/png" && (
            <label className="hidden md:inline-flex items-center gap-1.5 text-[11px] text-slate-300 bg-slate-900 border border-white/15 rounded-lg px-2.5 py-1.5">
              <span>Q</span>
              <input
                type="range"
                min={50}
                max={100}
                step={2}
                value={Math.round(exportQuality * 100)}
                onChange={(e) => setExportQuality(Number(e.target.value) / 100)}
                className="w-16 accent-sky-400"
                aria-label="Export quality percentage"
              />
              <span className="font-mono tabular-nums text-sky-300 w-7 text-right">
                {Math.round(exportQuality * 100)}%
              </span>
            </label>
          )}

          <button
            type="button"
            onClick={handleDownloadOnly}
            aria-label="Download edited image"
            className="px-3 py-2 min-h-[40px] rounded-lg bg-white/10 hover:bg-white/20 text-slate-100 font-medium text-xs inline-flex items-center gap-1.5 transition-colors"
            title="Download Edited File"
          >
            <Download className="w-4 h-4" />
            <span className="hidden sm:inline">Download</span>
          </button>

          <button
            type="button"
            onClick={handleSaveToVault}
            className="px-4 py-2 min-h-[40px] rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold text-xs inline-flex items-center gap-2 transition-colors"
          >
            <Save className="w-4 h-4" />
            Save Copy
          </button>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close photo editor"
            className="p-2 min-w-[40px] min-h-[40px] flex items-center justify-center rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 transition-colors"
            title="Close Editor (Esc)"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {exportError && (
        <div
          role="alert"
          onClick={(e) => e.stopPropagation()}
          className="w-full max-w-7xl mt-2 px-4 py-2 rounded-lg bg-rose-950/70 border border-rose-400/40 text-xs text-rose-200"
        >
          {exportError}
        </div>
      )}

      {/* Editor Body: Canvas & Sidebar Controls */}
      <div
        className="w-full max-w-7xl flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-6 my-3 sm:my-4 lg:overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Canvas Display Area */}
        <div className="lg:col-span-8 flex flex-col items-center justify-center relative rounded-2xl bg-slate-950/80 border border-white/10 p-4 overflow-auto min-h-[260px]">
          <div className="relative max-h-[65vh] flex items-center justify-center">
            <canvas
              ref={canvasRef}
              className="max-h-[60vh] max-w-full object-contain rounded-lg shadow-2xl"
            />
          </div>

          <div className="absolute bottom-4 left-4 flex items-center gap-2">
            <button
              type="button"
              onMouseDown={() => setShowBeforeAfter(true)}
              onMouseUp={() => setShowBeforeAfter(false)}
              onMouseLeave={() => setShowBeforeAfter(false)}
              onTouchStart={() => setShowBeforeAfter(true)}
              onTouchEnd={() => setShowBeforeAfter(false)}
              onKeyDown={(e) => {
                if (e.key === " " || e.key === "Enter") setShowBeforeAfter(true);
              }}
              onKeyUp={() => setShowBeforeAfter(false)}
              className="px-3 py-2 min-h-[38px] rounded-lg bg-slate-900/90 border border-white/20 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 backdrop-blur-md shadow-lg select-none"
              title="Hold to view original"
            >
              <Eye className="w-3.5 h-3.5 text-sky-400" />
              {showBeforeAfter ? "Viewing Original" : "Hold Before / After"}
            </button>
          </div>
        </div>

        {/* Editor Sidebar Controls */}
        <div className="lg:col-span-4 rounded-2xl bg-slate-900/90 border border-white/10 backdrop-blur-md flex flex-col overflow-hidden">
          {/* Tabs */}
          <div className="grid grid-cols-3 border-b border-white/10 text-xs font-medium" role="tablist">
            {(
              [
                { id: "adjust", label: "Adjust" },
                { id: "filters", label: "Filters" },
                { id: "transform", label: "Transform" },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={activeTab === tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-3 min-h-[42px] text-center transition-colors ${
                  activeTab === tab.id
                    ? "bg-sky-500/15 text-sky-300 border-b-2 border-sky-400"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="flex-1 p-5 space-y-5 overflow-y-auto">
            {activeTab === "adjust" && (
              <div className="space-y-4">
                {[
                  { label: "Brightness", key: "brightness" as const, min: 0, max: 200, unit: "%" },
                  { label: "Contrast", key: "contrast" as const, min: 0, max: 200, unit: "%" },
                  { label: "Saturation", key: "saturation" as const, min: 0, max: 200, unit: "%" },
                  { label: "Exposure", key: "exposure" as const, min: -100, max: 100, unit: "" },
                  { label: "Sharpness", key: "sharpness" as const, min: 0, max: 100, unit: "%" },
                ].map((item) => (
                  <div key={item.key} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <label htmlFor={`editor-${item.key}`} className="text-slate-300 font-medium">
                        {item.label}
                      </label>
                      <span className="font-mono text-sky-300 tabular-nums">
                        {state[item.key]}
                        {item.unit}
                      </span>
                    </div>
                    <input
                      id={`editor-${item.key}`}
                      type="range"
                      min={item.min}
                      max={item.max}
                      value={state[item.key]}
                      onChange={(e) => {
                        const val = Number(e.target.value);
                        updateState((prev) => ({ ...prev, [item.key]: val }));
                      }}
                      className="w-full accent-sky-400 bg-slate-950 rounded-lg h-2 cursor-pointer"
                    />
                  </div>
                ))}
              </div>
            )}

            {activeTab === "filters" && (
              <div className="grid grid-cols-2 gap-3">
                {FILTER_PRESETS.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    onClick={() => updateState((prev) => ({ ...prev, filter: f.id }))}
                    className={`p-3 min-h-[56px] rounded-xl border text-left transition-all ${
                      state.filter === f.id
                        ? "bg-sky-500/20 border-sky-400 text-sky-200"
                        : "bg-slate-950/60 border-white/10 text-slate-300 hover:border-white/30"
                    }`}
                  >
                    <div className="text-xs font-bold flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-sky-400" aria-hidden="true" />
                      {f.label}
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">Preset profile</div>
                  </button>
                ))}
              </div>
            )}

            {activeTab === "transform" && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <span className="text-xs font-medium text-slate-300">
                    Orientation &amp; Rotation ({state.rotation}°)
                  </span>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      type="button"
                      onClick={() =>
                        updateState((prev) => ({ ...prev, rotation: (prev.rotation + 90) % 360 }))
                      }
                      className="px-4 py-2.5 min-h-[42px] rounded-xl bg-slate-950 border border-white/10 hover:border-sky-400 text-xs font-medium text-slate-200 inline-flex items-center justify-center gap-2 transition-colors"
                    >
                      <RotateCw className="w-4 h-4 text-sky-400" />
                      Rotate 90°
                    </button>
                    <button
                      type="button"
                      onClick={() => updateState((prev) => ({ ...prev, flipH: !prev.flipH }))}
                      className="px-4 py-2.5 min-h-[42px] rounded-xl bg-slate-950 border border-white/10 hover:border-sky-400 text-xs font-medium text-slate-200 inline-flex items-center justify-center gap-2 transition-colors"
                    >
                      <FlipHorizontal className="w-4 h-4 text-sky-400" />
                      Flip Horizontal
                    </button>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-950/60 border border-white/10 text-xs text-slate-400 space-y-2">
                  <div className="font-semibold text-slate-200">Non-Destructive Pipeline</div>
                  <p>
                    Original image bytes remain untouched in IndexedDB. Saving creates a separate
                    edited copy under &ldquo;Edited Photos&rdquo;.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
