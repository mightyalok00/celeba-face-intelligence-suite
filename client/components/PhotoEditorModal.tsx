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
  Check,
  Eye,
  Crop,
  Maximize,
} from "lucide-react";

interface PhotoEditorModalProps {
  imageUrl: string;
  filename: string;
  onClose: () => void;
  onSaveEdited: (newUrl: string, newFilename: string) => void;
}

interface EditState {
  brightness: number; // 0 to 200, default 100
  contrast: number; // 0 to 200, default 100
  saturation: number; // 0 to 200, default 100
  exposure: number; // -100 to 100, default 0
  sharpness: number; // 0 to 100, default 0
  rotation: number; // 0, 90, 180, 270
  flipH: boolean;
  filter: "normal" | "vivid" | "cinematic" | "bw" | "warm" | "cool";
}

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

export function PhotoEditorModal({ imageUrl, filename, onClose, onSaveEdited }: PhotoEditorModalProps) {
  const [state, setState] = useState<EditState>(DEFAULT_STATE);
  const [history, setHistory] = useState<EditState[]>([DEFAULT_STATE]);
  const [historyIndex, setHistoryIndex] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<"adjust" | "filters" | "transform">("adjust");
  const [showBeforeAfter, setShowBeforeAfter] = useState(false);
  const [exportFormat, setExportFormat] = useState<"image/jpeg" | "image/png" | "image/webp">("image/jpeg");
  const [exportQuality, setExportQuality] = useState<number>(0.92);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageElement, setImageElement] = useState<HTMLImageElement | null>(null);

  // Load source image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      setImageElement(img);
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Push state to history
  const updateState = useCallback(
    (updater: (prev: EditState) => EditState) => {
      setState((prev) => {
        const next = updater(prev);
        const newHistory = history.slice(0, historyIndex + 1);
        newHistory.push(next);
        setHistory(newHistory);
        setHistoryIndex(newHistory.length - 1);
        return next;
      });
    },
    [history, historyIndex]
  );

  const handleUndo = () => {
    if (historyIndex > 0) {
      const prevIdx = historyIndex - 1;
      setHistoryIndex(prevIdx);
      setState(history[prevIdx]);
    }
  };

  const handleRedo = () => {
    if (historyIndex < history.length - 1) {
      const nextIdx = historyIndex + 1;
      setHistoryIndex(nextIdx);
      setState(history[nextIdx]);
    }
  };

  const handleReset = () => {
    setState(DEFAULT_STATE);
    setHistory([DEFAULT_STATE]);
    setHistoryIndex(0);
  };

  // Render to canvas with filters & adjustments
  const renderCanvas = useCallback(
    (targetCanvas: HTMLCanvasElement | null, overrideState?: EditState) => {
      if (!targetCanvas || !imageElement) return;
      const ctx = targetCanvas.getContext("2d");
      if (!ctx) return;

      const current = overrideState || state;
      const rot = current.rotation;
      const isRotated90or270 = rot === 90 || rot === 270;

      const w = isRotated90or270 ? imageElement.height : imageElement.width;
      const h = isRotated90or270 ? imageElement.width : imageElement.height;

      targetCanvas.width = w;
      targetCanvas.height = h;

      ctx.save();
      ctx.clearRect(0, 0, w, h);

      // Center transformations
      ctx.translate(w / 2, h / 2);
      ctx.rotate((rot * Math.PI) / 180);
      if (current.flipH) {
        ctx.scale(-1, 1);
      }

      // Filter settings
      let filterString = `brightness(${current.brightness}%) contrast(${current.contrast}%) saturate(${current.saturation}%)`;
      if (current.exposure !== 0) {
        const expBrightness = current.brightness + current.exposure;
        filterString = `brightness(${expBrightness}%) contrast(${current.contrast}%) saturate(${current.saturation}%)`;
      }

      if (current.filter === "vivid") {
        filterString += " saturate(140%) contrast(110%";
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
    },
    [imageElement, state]
  );

  useEffect(() => {
    renderCanvas(canvasRef.current);
  }, [imageElement, state, renderCanvas]);

  const handleExport = () => {
    if (!canvasRef.current) return;
    const dataUrl = canvasRef.current.toDataURL(exportFormat, exportQuality);
    const ext = exportFormat === "image/png" ? "png" : exportFormat === "image/webp" ? "webp" : "jpg";
    const editedName = filename.replace(/\.[^/.]+$/, "") + `_edited.${ext}`;
    onSaveEdited(dataUrl, editedName);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-black/95 backdrop-blur-xl flex flex-col items-center justify-between p-4 sm:p-6 animate-fade-in"
      onClick={onClose}
    >
      {/* Editor Header */}
      <div
        className="w-full max-w-7xl flex items-center justify-between text-slate-100 border-b border-white/10 pb-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3">
          <Sliders className="w-5 h-5 text-sky-400" />
          <div>
            <h2 className="text-base font-bold font-mono">{filename}</h2>
            <p className="text-xs text-slate-400">Google Photos Intelligence Studio Editor</p>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={handleUndo}
            disabled={historyIndex <= 0}
            className="p-2 rounded-lg bg-white/10 hover:bg-white/20 disabled:opacity-40 text-slate-200 transition-colors"
            title="Undo"
          >
            <Undo2 className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={handleRedo}
            disabled={historyIndex >= history.length - 1}
            className="p-2 rounded-lg bg-white/10 hover:bg-white/20 disabled:opacity-40 text-slate-200 transition-colors"
            title="Redo"
          >
            <Redo2 className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-slate-200 transition-colors"
            title="Reset All"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <div className="h-5 w-[1px] bg-white/15 mx-1" />

          <select
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value as any)}
            className="px-3 py-1.5 rounded-lg bg-slate-900 border border-white/15 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400"
          >
            <option value="image/jpeg">JPEG</option>
            <option value="image/png">PNG</option>
            <option value="image/webp">WebP</option>
          </select>

          <button
            type="button"
            onClick={handleExport}
            className="px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold text-xs inline-flex items-center gap-2 transition-colors"
          >
            <Download className="w-4 h-4" />
            Save & Export
          </button>

          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 transition-colors"
            title="Close Editor"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Editor Body: Canvas & Sidebar Controls */}
      <div
        className="w-full max-w-7xl flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 my-4 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Canvas Display Area */}
        <div className="lg:col-span-8 flex flex-col items-center justify-center relative rounded-2xl bg-slate-950/80 border border-white/10 p-4 overflow-auto">
          <div className="relative max-h-[65vh] flex items-center justify-center">
            <canvas ref={canvasRef} className="max-h-[60vh] max-w-full object-contain rounded-lg shadow-2xl" />
          </div>

          <div className="absolute bottom-4 left-4 flex items-center gap-2">
            <button
              type="button"
              onMouseDown={() => setShowBeforeAfter(true)}
              onMouseUp={() => setShowBeforeAfter(false)}
              onTouchStart={() => setShowBeforeAfter(true)}
              onTouchEnd={() => setShowBeforeAfter(false)}
              className="px-3 py-1.5 rounded-lg bg-slate-900/90 border border-white/20 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 backdrop-blur-md shadow-lg"
              title="Hold to view original"
            >
              <Eye className="w-3.5 h-3.5 text-sky-400" />
              Hold Before / After
            </button>
          </div>
        </div>

        {/* Editor Sidebar Controls */}
        <div className="lg:col-span-4 rounded-2xl bg-slate-900/90 border border-white/10 backdrop-blur-md flex flex-col overflow-hidden">
          {/* Tabs */}
          <div className="grid grid-cols-3 border-b border-white/10 text-xs font-medium">
            <button
              type="button"
              onClick={() => setActiveTab("adjust")}
              className={`py-3 text-center transition-colors ${
                activeTab === "adjust" ? "bg-sky-500/15 text-sky-300 border-b-2 border-sky-400" : "text-slate-400 hover:text-white"
              }`}
            >
              Adjust
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("filters")}
              className={`py-3 text-center transition-colors ${
                activeTab === "filters" ? "bg-sky-500/15 text-sky-300 border-b-2 border-sky-400" : "text-slate-400 hover:text-white"
              }`}
            >
              Filters
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("transform")}
              className={`py-3 text-center transition-colors ${
                activeTab === "transform" ? "bg-sky-500/15 text-sky-300 border-b-2 border-sky-400" : "text-slate-400 hover:text-white"
              }`}
            >
              Transform
            </button>
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
                      <span className="text-slate-300 font-medium">{item.label}</span>
                      <span className="font-mono text-sky-300 tabular-nums">
                        {state[item.key]}
                        {item.unit}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={item.min}
                      max={item.max}
                      value={state[item.key]}
                      onChange={(e) => {
                        const val = Number(e.target.value);
                        updateState((prev) => ({ ...prev, [item.key]: val }));
                      }}
                      className="w-full accent-sky-400 bg-slate-950 rounded-lg h-2"
                    />
                  </div>
                ))}
              </div>
            )}

            {activeTab === "filters" && (
              <div className="grid grid-cols-2 gap-3">
                {[
                  { id: "normal", label: "Natural", icon: Sparkles },
                  { id: "vivid", label: "Vivid", icon: Sparkles },
                  { id: "cinematic", label: "Cinematic", icon: Sparkles },
                  { id: "bw", label: "B & W", icon: Sparkles },
                  { id: "warm", label: "Warm Glow", icon: Sparkles },
                  { id: "cool", label: "Cool Tone", icon: Sparkles },
                ].map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    onClick={() => updateState((prev) => ({ ...prev, filter: f.id as any }))}
                    className={`p-3 rounded-xl border text-left transition-all ${
                      state.filter === f.id
                        ? "bg-sky-500/20 border-sky-400 text-sky-200"
                        : "bg-slate-950/60 border-white/10 text-slate-300 hover:border-white/30"
                    }`}
                  >
                    <div className="text-xs font-bold">{f.label}</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">Preset profile</div>
                  </button>
                ))}
              </div>
            )}

            {activeTab === "transform" && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <span className="text-xs font-medium text-slate-300">Orientation & Rotation</span>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      type="button"
                      onClick={() => updateState((prev) => ({ ...prev, rotation: (prev.rotation + 90) % 360 }))}
                      className="px-4 py-2.5 rounded-xl bg-slate-950 border border-white/10 hover:border-sky-400 text-xs font-medium text-slate-200 inline-flex items-center justify-center gap-2 transition-colors"
                    >
                      <RotateCw className="w-4 h-4 text-sky-400" />
                      Rotate 90°
                    </button>
                    <button
                      type="button"
                      onClick={() => updateState((prev) => ({ ...prev, flipH: !prev.flipH }))}
                      className="px-4 py-2.5 rounded-xl bg-slate-950 border border-white/10 hover:border-sky-400 text-xs font-medium text-slate-200 inline-flex items-center justify-center gap-2 transition-colors"
                    >
                      <FlipHorizontal className="w-4 h-4 text-sky-400" />
                      Flip Horizontal
                    </button>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-950/60 border border-white/10 text-xs text-slate-400 space-y-2">
                  <div className="font-semibold text-slate-200">Non-Destructive Pipeline</div>
                  <p>Original image bytes are preserved in memory. Edits apply in real-time via HTML5 Canvas rendering.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
