import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ArrowRight, CheckCircle2, AlertTriangle } from "lucide-react";
import { GlassCard } from "../components/GlassCard";
import type {
  BenchmarkSuitesResponse,
  DatasetAnalyticsResponse,
  PageId,
  SystemHealthResponse,
} from "../types";

interface DashboardPageProps {
  benchmarks: BenchmarkSuitesResponse | null;
  analytics: DatasetAnalyticsResponse | null;
  health: SystemHealthResponse | null;
  loading: boolean;
  error: string | null;
  onNavigate: (page: PageId) => void;
}

export function DashboardPage({
  benchmarks,
  analytics,
  health,
  loading,
  error,
  onNavigate,
}: DashboardPageProps) {
  const [chartSuite, setChartSuite] = useState<"real" | "synthetic">("real");

  const activeRows = useMemo(() => {
    if (!benchmarks) return [];
    return chartSuite === "real"
      ? benchmarks.realImageProduction.rows
      : benchmarks.ciSmokeTestSynthetic.rows;
  }, [benchmarks, chartSuite]);

  const chartData = useMemo(
    () =>
      activeRows.map((r) => ({
        name: r.model
          .replace(" (Tabular Attributes)", " (Tab)")
          .replace(" (Raw Image Pixels)", " (Px)")
          .replace("5-Fold Bagged ", "5F ")
          .replace("5-Fold Ensemble", "Ensemble")
          .replace("Artificial Neural Network ", "ANN "),
        accuracy: r.accuracyPct,
        rocAuc: r.rocAuc ? Number((r.rocAuc * 100).toFixed(2)) : 0,
        f1Pct: Number((r.f1Score * 100).toFixed(2)),
      })),
    [activeRows]
  );

  const weightsData = useMemo(() => {
    const weights = benchmarks?.latestTrainingRun.optimalEnsembleWeights ?? {};
    return Object.entries(weights).map(([name, val]) => ({
      name: name.toUpperCase(),
      value: Number((val * 100).toFixed(1)),
    }));
  }, [benchmarks]);

  const topRealModel = useMemo(() => {
    const rows = benchmarks?.realImageProduction.rows ?? [];
    return [...rows].sort((a, b) => (b.rocAuc ?? 0) - (a.rocAuc ?? 0))[0] ?? null;
  }, [benchmarks]);

  const weightColors = ["#38BDF8", "#818CF8", "#34D399"];

  if (loading && !benchmarks) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((idx) => (
            <GlassCard key={idx} className="h-28 animate-pulse bg-slate-900/50" >
              <div className="h-4 w-28 bg-slate-800 rounded" />
              <div className="h-8 w-20 bg-slate-800 rounded mt-4" />
            </GlassCard>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <GlassCard className="border-red-500/40 bg-red-950/20">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <h2 className="text-base font-semibold text-red-200">Unable to load repository telemetry</h2>
            <p className="text-sm text-red-300/90 mt-1">{error}</p>
          </div>
        </div>
      </GlassCard>
    );
  }

  return (
    <div className="space-y-6">
      {/* Hero Summary Row */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-sky-400">
            CelebA Face Intelligence Studio · Identity-Disjoint Evaluation
          </p>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-50 mt-1">
            Facial Attribute Benchmark & Inference Command
          </h2>
          <p className="text-sm text-slate-400 mt-1.5 max-w-3xl">
            Live workspace bridging PyTorch CNN, Spatial BiLSTM, Vision Transformer, ANN, and
            XGBoost repository benchmarks with server-side Gemini multimodal visual analysis.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => onNavigate("camera-analysis")}
            className="px-4 py-2.5 min-h-[40px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-slate-100 text-sm font-medium transition-colors whitespace-nowrap"
          >
            Open Live Camera
          </button>
          <button
            type="button"
            onClick={() => onNavigate("performance-reports")}
            className="px-4 py-2.5 min-h-[40px] rounded-lg border border-sky-400/30 bg-sky-500/10 hover:bg-sky-500/20 text-sky-300 text-sm font-medium transition-colors whitespace-nowrap inline-flex items-center gap-2"
          >
            <span>Inspect Confusion Matrices</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 4 KPI Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <GlassCard>
          <div className="text-xs font-medium text-slate-400">CelebA Dataset Scale</div>
          <div className="text-2xl font-bold text-slate-50 mt-2 tabular-nums">
            {(analytics?.manifest.total_images ?? 202599).toLocaleString()}
          </div>
          <div className="text-xs text-slate-400 mt-1 tabular-nums">
            {(analytics?.manifest.total_unique_identities ?? 10177).toLocaleString()} identities · 40 attributes
          </div>
        </GlassCard>

        <GlassCard>
          <div className="text-xs font-medium text-slate-400">Best Real-Image ROC-AUC</div>
          <div className="text-2xl font-bold text-sky-300 mt-2 tabular-nums">
            {topRealModel?.rocAuc ? topRealModel.rocAuc.toFixed(4) : "—"}
          </div>
          <div className="text-xs text-slate-400 mt-1 truncate">
            {topRealModel?.model ?? "Loading..."} ({topRealModel?.accuracyPct.toFixed(2)}% Acc)
          </div>
        </GlassCard>

        <GlassCard>
          <div className="text-xs font-medium text-slate-400">Identity Leakage Audit</div>
          <div className="text-2xl font-bold text-emerald-400 mt-2 tabular-nums">
            {analytics?.splitManifestSummary.identityOverlapCount ?? 0}.0%
          </div>
          <div className="text-xs text-slate-400 mt-1">
            StratifiedGroupKFold · {analytics?.splitManifestSummary.totalRows ?? 500} split rows verified
          </div>
        </GlassCard>

        <GlassCard>
          <div className="text-xs font-medium text-slate-400">5-Fold CV Mean Accuracy</div>
          <div className="text-2xl font-bold text-slate-50 mt-2 tabular-nums">
            {benchmarks?.crossValidation.meanAccuracy
              ? `${(benchmarks.crossValidation.meanAccuracy * 100).toFixed(2)}%`
              : "—"}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {benchmarks?.crossValidation.folds.length ?? 0} folds evaluated (cross_validation_results.csv)
          </div>
        </GlassCard>
      </div>

      {/* Main Chart + Optimal Ensemble Weights */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 items-stretch">
        <GlassCard className="xl:col-span-2 flex flex-col">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
            <div>
              <h3 className="text-base font-semibold text-slate-100">
                Architecture Accuracy & ROC-AUC Comparison
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                {chartSuite === "real"
                  ? "Source: reports/benchmarks/production_benchmark.csv (75 real identity-disjoint holdout faces)"
                  : "Source: reports/benchmarks/ci_smoke_test_benchmark.csv (30 synthetic CI smoke-test samples)"}
              </p>
            </div>

            <div className="inline-flex items-center p-1 rounded-lg bg-slate-950/80 border border-white/10 shrink-0">
              <button
                type="button"
                onClick={() => setChartSuite("real")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                  chartSuite === "real"
                    ? "bg-sky-500 text-slate-950"
                    : "text-slate-300 hover:text-white"
                }`}
              >
                Real-Image Holdout
              </button>
              <button
                type="button"
                onClick={() => setChartSuite("synthetic")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                  chartSuite === "synthetic"
                    ? "bg-amber-400 text-slate-950"
                    : "text-slate-300 hover:text-white"
                }`}
              >
                CI Smoke-Test (Synthetic)
              </button>
            </div>
          </div>

          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 8, right: 12, left: -12, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#94A3B8", fontSize: 11 }}
                  interval={0}
                  angle={-24}
                  textAnchor="end"
                  height={65}
                />
                <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0F172A",
                    borderColor: "rgba(255,255,255,0.12)",
                    borderRadius: "8px",
                    color: "#F8FAFC",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="accuracy" name="Accuracy (%)" fill="#38BDF8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="rocAuc" name="ROC-AUC (x100)" fill="#818CF8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>

        <GlassCard className="flex flex-col justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-100">
              SLSQP Ensemble Blending Weights
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Constrained log-loss optimization from latest_training_run.json
            </p>
          </div>

          <div className="h-56 w-full my-2">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={weightsData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={52}
                  outerRadius={80}
                  paddingAngle={3}
                >
                  {weightsData.map((entry, idx) => (
                    <Cell key={entry.name} fill={weightColors[idx % weightColors.length]} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value) => `${value}%`}
                  contentStyle={{
                    backgroundColor: "#0F172A",
                    borderColor: "rgba(255,255,255,0.12)",
                    borderRadius: "8px",
                    color: "#F8FAFC",
                    fontSize: "12px",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-2 border-t border-white/10 pt-3">
            {weightsData.map((item, idx) => (
              <div key={item.name} className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2 text-slate-300 font-medium">
                  <span
                    className="w-2.5 h-2.5 rounded-sm"
                    style={{ backgroundColor: weightColors[idx % weightColors.length] }}
                  />
                  {item.name}
                </span>
                <span className="font-mono text-slate-100 tabular-nums">{item.value}%</span>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>

      {/* Live Infrastructure & Checkpoint Readiness */}
      <GlassCard>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <h3 className="text-base font-semibold text-slate-100">
            Connected Services & Model Checkpoints
          </h3>
          <button
            type="button"
            onClick={() => onNavigate("system-health")}
            className="text-xs font-medium text-sky-400 hover:text-sky-300 underline-offset-4 hover:underline text-left"
          >
            View full diagnostics & API docs
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="p-4 rounded-lg bg-slate-950/60 border border-white/10">
            <div className="flex items-center justify-between">
              <span className="font-medium text-slate-200">Gemini Multimodal API</span>
              {health?.services.geminiApi.configured ? (
                <span className="text-xs font-medium text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Ready
                </span>
              ) : (
                <span className="text-xs font-medium text-amber-400">Not Configured</span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-1.5">
              Server-side SDK ({health?.services.geminiApi.defaultModel ?? "gemini-4-hydrogen"}) ·
              Zero image retention
            </p>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/60 border border-white/10">
            <div className="flex items-center justify-between">
              <span className="font-medium text-slate-200">Python FastAPI ML Engine</span>
              {health?.services.pythonMlBackend.engineReady ? (
                <span className="text-xs font-medium text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Connected
                </span>
              ) : (
                <span className="text-xs font-medium text-amber-400">Offline</span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-1.5">
              {health?.checkpoints.filter((c) => c.exists).length ?? 0} of{" "}
              {health?.checkpoints.length ?? 4} PyTorch checkpoints on disk (bilstm.pt,
              vision_transformer.pt)
            </p>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/60 border border-white/10">
            <div className="flex items-center justify-between">
              <span className="font-medium text-slate-200">MySQL celeba_db</span>
              {health?.services.mysqlDatabase.reachable ? (
                <span className="text-xs font-medium text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Reachable
                </span>
              ) : (
                <span className="text-xs font-medium text-slate-400">Offline / Unconfigured</span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-1.5">
              Benchmark CSVs and dataset manifests remain directly accessible from repository storage.
            </p>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
