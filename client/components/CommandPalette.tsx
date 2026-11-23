import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Download, FileBarChart2, Layers, Search, X } from "lucide-react";
import type { BenchmarkSuitesResponse, DatasetAnalyticsResponse, PageId } from "../types";

interface CommandEntry {
  id: string;
  group: "Pages" | "Models" | "Figures" | "Quick Actions";
  title: string;
  subtitle: string;
  keywords: string;
  run: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  onNavigate: (page: PageId) => void;
  benchmarks: BenchmarkSuitesResponse | null;
  analytics: DatasetAnalyticsResponse | null;
}

const PAGE_COMMANDS: { id: PageId; title: string; subtitle: string }[] = [
  { id: "dashboard", title: "Intelligence Dashboard", subtitle: "KPIs, showcase slider & benchmark threshold chart" },
  { id: "image-analysis", title: "Facial Attribute Image Analysis", subtitle: "Upload image, inspection sliders, ML consensus & Gemini" },
  { id: "camera-analysis", title: "Live Camera Studio", subtitle: "Consent-gated real-time webcam frame evaluation" },
  { id: "model-comparison", title: "Deep Learning Model Comparison", subtitle: "Radar comparison, SLSQP weight simulator, model cards & YAML" },
  { id: "dataset-analytics", title: "Dataset & Split Analytics", subtitle: "202,599 images, 0% overlap StratifiedGroupKFold & gallery" },
  { id: "performance-reports", title: "Performance Reports & Exports", subtitle: "Confusion matrices, 5-fold CV, CSV/JSON export & lightbox" },
  { id: "ai-chat", title: "AI Research & Engineering Assistant", subtitle: "Streaming Gemini assistant grounded with repository metrics" },
  { id: "system-health", title: "System Health, Privacy & API Docs", subtitle: "Service probes, Docker Compose & GitHub README banners" },
];

export function CommandPalette({
  open,
  onClose,
  onNavigate,
  benchmarks,
  analytics,
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) {
      setQuery("");
      return;
    }
    const timer = window.setTimeout(() => inputRef.current?.focus(), 20);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  const entries = useMemo<CommandEntry[]>(() => {
    const items: CommandEntry[] = PAGE_COMMANDS.map((p) => ({
      id: `page-${p.id}`,
      group: "Pages",
      title: p.title,
      subtitle: p.subtitle,
      keywords: `${p.id} ${p.title} ${p.subtitle}`.toLowerCase(),
      run: () => {
        onNavigate(p.id);
        onClose();
      },
    }));

    items.push(
      {
        id: "action-export-prod-csv",
        group: "Quick Actions",
        title: "Download Production Benchmark CSV",
        subtitle: "GET /api/benchmarks/export?suite=production&format=csv",
        keywords: "export download csv benchmark production holdout",
        run: () => {
          window.location.assign("/api/benchmarks/export?suite=production&format=csv");
          onClose();
        },
      },
      {
        id: "action-export-prod-json",
        group: "Quick Actions",
        title: "Download Production Benchmark JSON",
        subtitle: "GET /api/benchmarks/export?suite=production&format=json",
        keywords: "export download json benchmark production holdout",
        run: () => {
          window.location.assign("/api/benchmarks/export?suite=production&format=json");
          onClose();
        },
      }
    );

    for (const row of benchmarks?.realImageProduction.rows ?? []) {
      items.push({
        id: `model-${row.model}`,
        group: "Models",
        title: row.model,
        subtitle: `Holdout Accuracy ${row.accuracyPct.toFixed(2)}% · F1 ${row.f1Score.toFixed(4)}`,
        keywords: `${row.model} ${row.strategy} accuracy model benchmark`.toLowerCase(),
        run: () => {
          onNavigate("model-comparison");
          onClose();
        },
      });
    }

    for (const fig of analytics?.figures ?? []) {
      items.push({
        id: `fig-${fig.filename}`,
        group: "Figures",
        title: fig.title,
        subtitle: `reports/figures/${fig.filename}`,
        keywords: `${fig.title} ${fig.filename} ${fig.category} figure chart`.toLowerCase(),
        run: () => {
          onNavigate(fig.category === "dataset" ? "dataset-analytics" : "performance-reports");
          onClose();
        },
      });
    }

    return items;
  }, [analytics, benchmarks, onClose, onNavigate]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries.slice(0, 14);
    return entries.filter((e) => e.keywords.includes(q)).slice(0, 14);
  }, [entries, query]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Global Command Palette"
      className="fixed inset-0 z-50 flex items-start justify-center pt-[12vh] px-4 bg-black/75 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl rounded-2xl border border-white/15 bg-[#0C1220] shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 px-4 py-3 border-b border-white/10">
          <Search className="w-4 h-4 text-sky-400 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && filtered.length > 0) {
                e.preventDefault();
                filtered[0].run();
              }
            }}
            placeholder="Search pages, models, benchmarks, exports, or figures..."
            className="flex-1 bg-transparent text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none"
          />
          <button
            type="button"
            aria-label="Close command palette"
            onClick={onClose}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-white/10"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="max-h-[55vh] overflow-y-auto divide-y divide-white/5 p-2">
          {filtered.length === 0 ? (
            <div className="py-10 text-center text-xs text-slate-400">
              No matching pages, models, or figures found.
            </div>
          ) : (
            filtered.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={item.run}
                className="w-full px-3 py-2.5 rounded-lg hover:bg-sky-500/10 flex items-center justify-between gap-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                <div className="flex items-center gap-3 min-w-0">
                  {item.group === "Quick Actions" ? (
                    <Download className="w-4 h-4 text-emerald-400 shrink-0" />
                  ) : item.group === "Models" ? (
                    <Layers className="w-4 h-4 text-sky-400 shrink-0" />
                  ) : (
                    <FileBarChart2 className="w-4 h-4 text-slate-400 shrink-0" />
                  )}
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-slate-100 truncate">{item.title}</div>
                    <div className="text-xs text-slate-400 truncate">{item.subtitle}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-[11px] font-mono text-slate-400">{item.group}</span>
                  <ArrowRight className="w-3.5 h-3.5 text-slate-500" />
                </div>
              </button>
            ))
          )}
        </div>

        <div className="px-4 py-2 border-t border-white/10 bg-slate-950/60 flex items-center justify-between text-[11px] text-slate-400">
          <span>Enter to jump to top match · Esc to close</span>
          <span className="font-mono">Ctrl/Cmd + K</span>
        </div>
      </div>
    </div>
  );
}
