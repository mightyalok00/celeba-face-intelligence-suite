import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Download, Maximize2, ShieldCheck, Database, Calendar, FileCode, Layers } from "lucide-react";
import { FigureLightbox, type LightboxFigure } from "../components/FigureLightbox";
import { GlassCard } from "../components/GlassCard";
import type { BenchmarkRow, BenchmarkSuitesResponse, DatasetAnalyticsResponse } from "../types";

interface PerformanceReportsPageProps {
  benchmarks: BenchmarkSuitesResponse | null;
  analytics: DatasetAnalyticsResponse | null;
}

type SuiteKey = "production" | "latest" | "synthetic";

export function PerformanceReportsPage({ benchmarks, analytics }: PerformanceReportsPageProps) {
  const [suite, setSuite] = useState<SuiteKey>("production");
  const [selectedModelName, setSelectedModelName] = useState<string>("");
  const [activeFigure, setActiveFigure] = useState<LightboxFigure | null>(null);

  const activeRows: BenchmarkRow[] = useMemo(() => {
    if (!benchmarks) return [];
    if (suite === "production") return benchmarks.realImageProduction.rows;
    if (suite === "latest") return benchmarks.latestTrainingRun.rows;
    return benchmarks.ciSmokeTestSynthetic.rows;
  }, [benchmarks, suite]);

  const selectedRow = useMemo(() => {
    if (activeRows.length === 0) return null;
    return activeRows.find((r) => r.model === selectedModelName) ?? activeRows[0];
  }, [activeRows, selectedModelName]);

  // Provenance metadata derived from active suite
  const suiteMetadata = useMemo(() => {
    if (!benchmarks) {
      return {
        sourceFile: "Unavailable",
        splitName: "Unavailable",
        sampleSize: "Unavailable",
        evaluationDate: "Unavailable",
        modelVersion: "v2.5.0",
        isSynthetic: false,
      };
    }
    if (suite === "production") {
      return {
        sourceFile: benchmarks.realImageProduction.sourceFile,
        splitName: "Real-Image Holdout (Identity-Disjoint StratifiedGroupKFold Test)",
        sampleSize: `${benchmarks.realImageProduction.sampleCount} real images`,
        evaluationDate: "Unavailable",
        modelVersion: "v2.5.0",
        isSynthetic: false,
      };
    }
    if (suite === "latest") {
      const run = benchmarks.latestTrainingRun;
      return {
        sourceFile: run.sourceFile,
        splitName: "Latest Training Run (Train/Test Split)",
        sampleSize: `${run.testSamples} test samples (${run.trainSamples} train)`,
        evaluationDate: run.timestamp || "Unavailable",
        modelVersion: "v2.5.0",
        isSynthetic: false,
      };
    }
    return {
      sourceFile: benchmarks.ciSmokeTestSynthetic.sourceFile,
      splitName: "Synthetic CI Smoke-Test Pipeline Validation",
      sampleSize: `${benchmarks.ciSmokeTestSynthetic.sampleCount} synthetic samples`,
      evaluationDate: "Unavailable",
      modelVersion: "v2.5.0",
      isSynthetic: true,
    };
  }, [benchmarks, suite]);

  const performanceFigures = useMemo(
    () =>
      (analytics?.figures ?? []).filter(
        (f) => f.category === "performance" || f.category === "confusion_matrix"
      ),
    [analytics]
  );

  const cvChartData = useMemo(
    () =>
      (benchmarks?.crossValidation.folds ?? []).map((f) => ({
        fold: `Fold ${f.fold}`,
        accuracyPct: Number((f.valAccuracy * 100).toFixed(2)),
      })),
    [benchmarks]
  );

  const cm = selectedRow?.confusionMatrix ?? null;
  const cmStats = useMemo(() => {
    if (!cm) return null;
    const [[tn, fp], [fn, tp]] = cm;
    const total = tn + fp + fn + tp;
    const specificity = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;
    const sensitivity = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
    return { tn, fp, fn, tp, total, specificity, sensitivity };
  }, [cm]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-50">Model Performance &amp; Evaluation Reports</h2>
          <p className="text-sm text-slate-400 mt-1">
            Exact Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrices
            from repository evaluation reports.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <div className="inline-flex flex-wrap items-center p-1 rounded-lg bg-slate-900 border border-white/10">
            <button
              type="button"
              onClick={() => setSuite("production")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                suite === "production"
                  ? "bg-sky-500 text-slate-950 font-semibold"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              Real-Image Holdout (75)
            </button>
            <button
              type="button"
              onClick={() => setSuite("latest")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                suite === "latest"
                  ? "bg-sky-500 text-slate-950 font-semibold"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              Latest Training Run (30)
            </button>
            <button
              type="button"
              onClick={() => setSuite("synthetic")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                suite === "synthetic"
                  ? "bg-amber-400 text-slate-950 font-semibold"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              Synthetic CI Smoke-Test (30)
            </button>
          </div>

          <div className="inline-flex items-center gap-2">
            <a
              href={`/api/benchmarks/export?suite=${suite}&format=csv`}
              download={`celeba_${suite}_benchmarks.csv`}
              className="px-3 py-2 min-h-[36px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5 text-sky-400" />
              <span>Export CSV</span>
            </a>
            <a
              href={`/api/benchmarks/export?suite=${suite}&format=json`}
              download={`celeba_${suite}_benchmarks.json`}
              className="px-3 py-2 min-h-[36px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5 text-emerald-400" />
              <span>Export JSON</span>
            </a>
          </div>
        </div>
      </div>

      {suite === "synthetic" && benchmarks && (
        <div className="p-4 rounded-xl border border-amber-500/40 bg-amber-950/25 text-xs sm:text-sm text-amber-200 space-y-1">
          <div className="font-bold uppercase tracking-wider text-[11px] text-amber-300">
            Synthetic Pipeline Validation Notice
          </div>
          <div>{benchmarks.ciSmokeTestSynthetic.note}</div>
        </div>
      )}

      {/* Artifact Provenance & Metadata Strip */}
      <GlassCard className="p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="space-y-1">
          <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1.5">
            <FileCode className="w-3.5 h-3.5 text-sky-400" /> Source Artifact
          </div>
          <div className="text-xs font-mono font-semibold text-slate-100 truncate" title={suiteMetadata.sourceFile}>
            {suiteMetadata.sourceFile}
          </div>
        </div>

        <div className="space-y-1">
          <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-indigo-400" /> Evaluation Split
          </div>
          <div className="text-xs font-medium text-slate-100 truncate" title={suiteMetadata.splitName}>
            {suiteMetadata.splitName}
          </div>
        </div>

        <div className="space-y-1">
          <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-emerald-400" /> Sample Size
          </div>
          <div className="text-xs font-mono font-semibold text-slate-100 tabular-nums">
            {suiteMetadata.sampleSize}
          </div>
        </div>

        <div className="space-y-1">
          <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1.5">
            <Calendar className="w-3.5 h-3.5 text-amber-400" /> Evaluation Date
          </div>
          <div className="text-xs font-mono font-semibold text-slate-100 tabular-nums truncate">
            {suiteMetadata.evaluationDate}
          </div>
        </div>

        <div className="space-y-1">
          <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-sky-400" /> Model Version
          </div>
          <div className="text-xs font-mono font-semibold text-sky-300 tabular-nums">
            {suiteMetadata.modelVersion}
            <span className="ml-1 px-1.5 py-0.5 rounded bg-white/10 text-[10px] text-slate-300 font-normal">
              {suiteMetadata.isSynthetic ? "Synthetic CI" : "Real-Image"}
            </span>
          </div>
        </div>
      </GlassCard>

      {/* Full Metrics Table */}
      <GlassCard className="space-y-3">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span>
            Suite Dataset Type:{" "}
            <span className="font-mono text-slate-200">
              {suite === "production"
                ? "real_image_holdout (75 samples)"
                : suite === "latest"
                  ? "real_image_training_run (30 test holdout)"
                  : "synthetic_smoke_test (30 generated samples)"}
            </span>
          </span>
          <span>Click any row to inspect its confusion matrix</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs sm:text-sm">
            <thead>
              <tr className="border-b border-white/10 text-slate-400">
                <th className="py-2.5 pr-3 font-medium">Model Configuration</th>
                <th className="py-2.5 px-3 font-medium text-right">Test Loss</th>
                <th className="py-2.5 px-3 font-medium text-right">Accuracy</th>
                <th className="py-2.5 px-3 font-medium text-right">Precision</th>
                <th className="py-2.5 px-3 font-medium text-right">Recall</th>
                <th className="py-2.5 px-3 font-medium text-right">F1-Score</th>
                <th className="py-2.5 px-3 font-medium text-right">ROC-AUC</th>
                <th className="py-2.5 pl-3 font-medium text-right">PR-AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {activeRows.map((row) => {
                const isSelected = selectedRow?.model === row.model;
                return (
                  <tr
                    key={row.model}
                    onClick={() => setSelectedModelName(row.model)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? "bg-sky-500/10" : "hover:bg-white/[0.03]"
                    }`}
                  >
                    <td className="py-2.5 pr-3 font-medium text-slate-100">
                      <div>{row.model}</div>
                      <div className="text-[11px] text-slate-400">{row.strategy}</div>
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-300">
                      {row.testLoss.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-sky-300 font-semibold">
                      {row.accuracyPct.toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-200">
                      {row.precision.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-200">
                      {row.recall.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-200">
                      {row.f1Score.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-emerald-300">
                      {row.rocAuc !== null ? row.rocAuc.toFixed(4) : "Unavailable"}
                    </td>
                    <td className="py-2.5 pl-3 text-right font-mono tabular-nums text-slate-300">
                      {row.prAuc !== null ? row.prAuc.toFixed(4) : "Unavailable"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </GlassCard>

      {/* Confusion Matrix Inspector + 5-Fold Cross Validation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GlassCard className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-base font-semibold text-slate-100">
                Confusion Matrix Inspector
              </h3>
              <p className="text-xs text-slate-400">{selectedRow?.model ?? "Select a model"}</p>
            </div>
            <select
              aria-label="Select model for confusion matrix"
              value={selectedRow?.model ?? ""}
              onChange={(e) => setSelectedModelName(e.target.value)}
              className="rounded-lg border border-white/15 bg-slate-950 px-3 py-1.5 text-xs text-slate-100"
            >
              {activeRows.map((r) => (
                <option key={r.model} value={r.model}>
                  {r.model}
                </option>
              ))}
            </select>
          </div>

          {cmStats ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 text-center">
                <div className="p-4 rounded-lg bg-emerald-500/15 border border-emerald-400/30">
                  <div className="text-xs text-emerald-300">True Negative (Neutral)</div>
                  <div className="text-2xl font-bold text-slate-50 font-mono tabular-nums mt-1">
                    {cmStats.tn}
                  </div>
                </div>
                <div className="p-4 rounded-lg bg-amber-500/10 border border-amber-400/25">
                  <div className="text-xs text-amber-300">False Positive</div>
                  <div className="text-2xl font-bold text-slate-50 font-mono tabular-nums mt-1">
                    {cmStats.fp}
                  </div>
                </div>
                <div className="p-4 rounded-lg bg-rose-500/10 border border-rose-400/25">
                  <div className="text-xs text-rose-300">False Negative</div>
                  <div className="text-2xl font-bold text-slate-50 font-mono tabular-nums mt-1">
                    {cmStats.fn}
                  </div>
                </div>
                <div className="p-4 rounded-lg bg-sky-500/15 border border-sky-400/30">
                  <div className="text-xs text-sky-300">True Positive (Smiling)</div>
                  <div className="text-2xl font-bold text-slate-50 font-mono tabular-nums mt-1">
                    {cmStats.tp}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3 text-xs pt-2 border-t border-white/10">
                <div>
                  <span className="text-slate-400">Holdout N:</span>{" "}
                  <span className="font-mono text-slate-100 tabular-nums">{cmStats.total}</span>
                </div>
                <div>
                  <span className="text-slate-400">Specificity:</span>{" "}
                  <span className="font-mono text-slate-100 tabular-nums">
                    {cmStats.specificity.toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-slate-400">Sensitivity:</span>{" "}
                  <span className="font-mono text-slate-100 tabular-nums">
                    {cmStats.sensitivity.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-sm text-slate-400">No confusion matrix available.</div>
          )}
        </GlassCard>

        <GlassCard className="space-y-4">
          <div>
            <h3 className="text-base font-semibold text-slate-100">
              5-Fold Cross-Validation Accuracy
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Source: reports/benchmarks/cross_validation_results.csv (Mean:{" "}
              {benchmarks ? (benchmarks.crossValidation.meanAccuracy * 100).toFixed(2) : "—"}%)
            </p>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={cvChartData} margin={{ top: 8, right: 16, left: -12, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                <XAxis dataKey="fold" tick={{ fill: "#CBD5E1", fontSize: 12 }} />
                <YAxis domain={[50, 100]} tick={{ fill: "#94A3B8", fontSize: 11 }} />
                <Tooltip
                  formatter={(v) => `${v}%`}
                  contentStyle={{
                    backgroundColor: "#0F172A",
                    borderColor: "rgba(255,255,255,0.12)",
                    borderRadius: "8px",
                    color: "#F8FAFC",
                    fontSize: "12px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="accuracyPct"
                  name="Validation Accuracy (%)"
                  stroke="#38BDF8"
                  strokeWidth={2.5}
                  dot={{ r: 4, fill: "#38BDF8" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>
      </div>

      {/* Repository Figures Gallery */}
      {performanceFigures.length > 0 && (
        <GlassCard className="space-y-4">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-base font-semibold text-slate-100">
              Generated Benchmark &amp; Confusion Matrix Figures (reports/figures/)
            </h3>
            <span className="text-xs text-slate-400">Click any figure to inspect full size</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {performanceFigures.map((fig) => (
              <button
                key={fig.filename}
                type="button"
                onClick={() =>
                  setActiveFigure({
                    title: fig.title,
                    url: fig.url,
                    filename: fig.filename,
                    subtitle: `reports/figures/${fig.filename}`,
                  })
                }
                className="group text-left rounded-lg border border-white/10 hover:border-sky-400/50 bg-slate-950/70 overflow-hidden transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                <div className="relative">
                  <img
                    src={fig.url}
                    alt={fig.title}
                    loading="lazy"
                    decoding="async"
                    referrerPolicy="no-referrer"
                    className="w-full h-48 object-contain bg-white"
                  />
                  <span className="absolute top-2 right-2 p-1.5 rounded-md bg-slate-900/80 text-slate-200 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Maximize2 className="w-3.5 h-3.5" />
                  </span>
                </div>
                <div className="p-3 text-xs font-medium text-slate-200 truncate">{fig.title}</div>
              </button>
            ))}
          </div>
        </GlassCard>
      )}

      <FigureLightbox figure={activeFigure} onClose={() => setActiveFigure(null)} />
    </div>
  );
}
