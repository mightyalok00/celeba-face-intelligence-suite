import { Fragment, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertTriangle,
  ArrowUpDown,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Download,
  FileCode2,
  FlaskConical,
  RotateCcw,
  Search,
  SlidersHorizontal,
  XCircle,
} from "lucide-react";
import { GlassCard } from "../components/GlassCard";
import { ScrollReveal } from "../components/ScrollReveal";
import type { BenchmarkRow, BenchmarkSuitesResponse, CheckpointInfo, ModelsResponse } from "../types";

interface ModelComparisonPageProps {
  benchmarks: BenchmarkSuitesResponse | null;
  models: ModelsResponse | null;
}

const DEFAULT_SLSQP_WEIGHTS = {
  xgb: 0.3457,
  cnn: 0.3959,
  vit: 0.2584,
};

type ArchFamily = "all" | "deep" | "tree" | "ensemble";
type CheckpointFilter = "all" | "available" | "missing";
type SortField = "accuracyPct" | "precision" | "recall" | "f1Score" | "rocAuc" | "prAuc" | "testLoss" | "model";

function classifyFamily(modelName: string): Exclude<ArchFamily, "all"> {
  const lower = modelName.toLowerCase();
  if (lower.includes("ensemble")) return "ensemble";
  if (lower.includes("xgboost")) return "tree";
  return "deep";
}

function matchCheckpointForModel(
  modelName: string,
  checkpoints: CheckpointInfo[]
): CheckpointInfo | null {
  const lower = modelName.toLowerCase();
  if (lower.includes("vision transformer") || lower.includes("vit")) {
    return checkpoints.find((c) => c.architecture === "Vision Transformer") ?? null;
  }
  if (lower.includes("bilstm")) {
    return checkpoints.find((c) => c.architecture === "Spatial BiLSTM") ?? null;
  }
  if (lower.includes("residual cnn")) {
    return checkpoints.find((c) => c.architecture === "Residual CNN") ?? null;
  }
  if (lower.includes("ann") || lower.includes("artificial neural network")) {
    return checkpoints.find((c) => c.architecture === "ANN") ?? null;
  }
  return null;
}

export function ModelComparisonPage({ benchmarks, models }: ModelComparisonPageProps) {
  const [suite, setSuite] = useState<"real" | "synthetic">("real");
  const [selectedCardId, setSelectedCardId] = useState<string>("ensemble");

  // Leaderboard filters & sorting
  const [searchQuery, setSearchQuery] = useState("");
  const [familyFilter, setFamilyFilter] = useState<ArchFamily>("all");
  const [ckptFilter, setCkptFilter] = useState<CheckpointFilter>("all");
  const [sortField, setSortField] = useState<SortField>("rocAuc");
  const [sortDir, setSortDir] = useState<"desc" | "asc">("desc");
  const [expandedModel, setExpandedModel] = useState<string | null>(null);

  // SLSQP Ensemble Explorer state (hypothetical unless matching verified run)
  const [weightXgb, setWeightXgb] = useState<number>(Math.round(DEFAULT_SLSQP_WEIGHTS.xgb * 100));
  const [weightCnn, setWeightCnn] = useState<number>(Math.round(DEFAULT_SLSQP_WEIGHTS.cnn * 100));
  const [weightVit, setWeightVit] = useState<number>(Math.round(DEFAULT_SLSQP_WEIGHTS.vit * 100));

  const verifiedWeights = useMemo(() => {
    const raw = benchmarks?.latestTrainingRun.optimalEnsembleWeights ?? {};
    const findWeight = (needle: string, fallback: number) => {
      for (const [k, v] of Object.entries(raw)) {
        if (k.toLowerCase().includes(needle) && typeof v === "number") return v;
      }
      return fallback;
    };
    return {
      xgb: findWeight("xgb", DEFAULT_SLSQP_WEIGHTS.xgb),
      cnn: findWeight("cnn", DEFAULT_SLSQP_WEIGHTS.cnn),
      vit: findWeight("vit", DEFAULT_SLSQP_WEIGHTS.vit),
    };
  }, [benchmarks]);

  const normalizedBlend = useMemo(() => {
    const total = weightXgb + weightCnn + weightVit;
    const valid = total > 0 && weightXgb >= 0 && weightCnn >= 0 && weightVit >= 0;
    if (!valid) {
      return { xgb: 0, cnn: 0, vit: 0, total, valid: false };
    }
    const xgb = Number(((weightXgb / total) * 100).toFixed(2));
    const cnn = Number(((weightCnn / total) * 100).toFixed(2));
    const vit = Number((100 - xgb - cnn).toFixed(2));
    return { xgb, cnn, vit, total, valid: true };
  }, [weightXgb, weightCnn, weightVit]);

  const isVerifiedConfig = useMemo(() => {
    return (
      Math.abs(normalizedBlend.xgb - verifiedWeights.xgb * 100) < 0.8 &&
      Math.abs(normalizedBlend.cnn - verifiedWeights.cnn * 100) < 0.8 &&
      Math.abs(normalizedBlend.vit - verifiedWeights.vit * 100) < 0.8
    );
  }, [normalizedBlend, verifiedWeights]);

  const slsqpComparisonChartData = useMemo(
    () => [
      {
        estimator: "5-Fold XGBoost",
        Verified: Number((verifiedWeights.xgb * 100).toFixed(2)),
        Current: normalizedBlend.xgb,
      },
      {
        estimator: "Residual CNN",
        Verified: Number((verifiedWeights.cnn * 100).toFixed(2)),
        Current: normalizedBlend.cnn,
      },
      {
        estimator: "Vision Transformer",
        Verified: Number((verifiedWeights.vit * 100).toFixed(2)),
        Current: normalizedBlend.vit,
      },
    ],
    [verifiedWeights, normalizedBlend]
  );

  const resetWeightsToVerified = () => {
    setWeightXgb(Math.round(verifiedWeights.xgb * 100));
    setWeightCnn(Math.round(verifiedWeights.cnn * 100));
    setWeightVit(Math.round(verifiedWeights.vit * 100));
  };

  const rows = useMemo(() => {
    if (!benchmarks) return [];
    return suite === "real"
      ? benchmarks.realImageProduction.rows
      : benchmarks.ciSmokeTestSynthetic.rows;
  }, [benchmarks, suite]);

  const checkpoints = models?.checkpoints ?? [];
  const liveVerifiedSet = useMemo(
    () => new Set(models?.mlBackend.verifiedCheckpoints ?? []),
    [models?.mlBackend.verifiedCheckpoints]
  );

  const enrichedRows = useMemo(() => {
    return rows.map((row) => {
      const family = classifyFamily(row.model);
      const ckpt = matchCheckpointForModel(row.model, checkpoints);
      const liveReady = ckpt ? Boolean(models?.mlBackend.online && liveVerifiedSet.has(ckpt.architecture)) : false;
      const checkpointStatus = liveReady
        ? "ready"
        : ckpt?.exists
          ? "on_disk"
          : ckpt
            ? "missing"
            : "artifact_unavailable";
      return {
        row,
        family,
        ckpt,
        liveReady,
        checkpointStatus,
      };
    });
  }, [rows, checkpoints, liveVerifiedSet, models?.mlBackend.online]);

  const filteredSortedLeaderboard = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    const filtered = enrichedRows.filter((item) => {
      if (familyFilter !== "all" && item.family !== familyFilter) return false;
      if (ckptFilter === "available" && item.checkpointStatus !== "ready" && item.checkpointStatus !== "on_disk") {
        return false;
      }
      if (ckptFilter === "missing" && (item.checkpointStatus === "ready" || item.checkpointStatus === "on_disk")) {
        return false;
      }
      if (q) {
        const hay = `${item.row.model} ${item.row.strategy} ${item.ckpt?.filename ?? ""}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    const getSortValue = (r: BenchmarkRow, field: SortField): number | string => {
      switch (field) {
        case "model":
          return r.model.toLowerCase();
        case "accuracyPct":
          return r.accuracyPct;
        case "precision":
          return r.precision;
        case "recall":
          return r.recall;
        case "f1Score":
          return r.f1Score;
        case "rocAuc":
          return r.rocAuc ?? -1;
        case "prAuc":
          return r.prAuc ?? -1;
        case "testLoss":
          return r.testLoss;
      }
    };

    return [...filtered].sort((a, b) => {
      const av = getSortValue(a.row, sortField);
      const bv = getSortValue(b.row, sortField);
      if (typeof av === "string" && typeof bv === "string") {
        return sortDir === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      const diff = Number(av) - Number(bv);
      return sortDir === "asc" ? diff : -diff;
    });
  }, [enrichedRows, searchQuery, familyFilter, ckptFilter, sortField, sortDir]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((prev) => (prev === "desc" ? "asc" : "desc"));
    } else {
      setSortField(field);
      setSortDir(field === "testLoss" || field === "model" ? "asc" : "desc");
    }
  };

  const barData = useMemo(
    () =>
      rows.map((r) => ({
        name: r.model
          .replace(" (Tabular Attributes)", " Tab")
          .replace(" (Raw Image Pixels)", " Px")
          .replace("Artificial Neural Network ", "ANN "),
        Accuracy: r.accuracyPct,
        Precision: Number((r.precision * 100).toFixed(2)),
        Recall: Number((r.recall * 100).toFixed(2)),
        F1: Number((r.f1Score * 100).toFixed(2)),
      })),
    [rows]
  );

  const rocPrData = useMemo(
    () =>
      rows
        .filter((r) => r.rocAuc !== null || r.prAuc !== null)
        .map((r) => ({
          name: r.model
            .replace(" (Tabular Attributes)", " Tab")
            .replace(" (Raw Image Pixels)", " Px")
            .replace("Artificial Neural Network ", "ANN "),
          "ROC-AUC": r.rocAuc !== null ? Number((r.rocAuc * 100).toFixed(2)) : null,
          "PR-AUC": r.prAuc !== null ? Number((r.prAuc * 100).toFixed(2)) : null,
        })),
    [rows]
  );

  const radarData = useMemo(() => {
    const pick = (needle: string) => rows.find((r) => r.model.includes(needle));
    const vit = pick("Vision Transformer");
    const cnn = pick("Residual CNN");
    const bilstm = pick("BiLSTM");
    const ann = pick("ANN");
    const xgb = pick("Single-Fold XGBoost");
    const ens = pick("Ensemble + TTA");

    return [
      {
        metric: "Accuracy",
        ViT: vit?.accuracyPct ?? 0,
        CNN: cnn?.accuracyPct ?? 0,
        BiLSTM: bilstm?.accuracyPct ?? 0,
        ANN: ann?.accuracyPct ?? 0,
        XGBoost: xgb?.accuracyPct ?? 0,
        Ensemble: ens?.accuracyPct ?? 0,
      },
      {
        metric: "Precision",
        ViT: (vit?.precision ?? 0) * 100,
        CNN: (cnn?.precision ?? 0) * 100,
        BiLSTM: (bilstm?.precision ?? 0) * 100,
        ANN: (ann?.precision ?? 0) * 100,
        XGBoost: (xgb?.precision ?? 0) * 100,
        Ensemble: (ens?.precision ?? 0) * 100,
      },
      {
        metric: "Recall",
        ViT: (vit?.recall ?? 0) * 100,
        CNN: (cnn?.recall ?? 0) * 100,
        BiLSTM: (bilstm?.recall ?? 0) * 100,
        ANN: (ann?.recall ?? 0) * 100,
        XGBoost: (xgb?.recall ?? 0) * 100,
        Ensemble: (ens?.recall ?? 0) * 100,
      },
      {
        metric: "F1-Score",
        ViT: (vit?.f1Score ?? 0) * 100,
        CNN: (cnn?.f1Score ?? 0) * 100,
        BiLSTM: (bilstm?.f1Score ?? 0) * 100,
        ANN: (ann?.f1Score ?? 0) * 100,
        XGBoost: (xgb?.f1Score ?? 0) * 100,
        Ensemble: (ens?.f1Score ?? 0) * 100,
      },
      {
        metric: "ROC-AUC",
        ViT: (vit?.rocAuc ?? 0) * 100,
        CNN: (cnn?.rocAuc ?? 0) * 100,
        BiLSTM: (bilstm?.rocAuc ?? 0) * 100,
        ANN: (ann?.rocAuc ?? 0) * 100,
        XGBoost: (xgb?.rocAuc ?? 0) * 100,
        Ensemble: (ens?.rocAuc ?? 0) * 100,
      },
    ];
  }, [rows]);

  // Verified research insights derived strictly from production_benchmark.csv
  const researchInsights = useMemo(() => {
    const prod = benchmarks?.realImageProduction.rows ?? [];
    if (prod.length === 0) return null;
    const accLeader = [...prod].sort((a, b) => b.accuracyPct - a.accuracyPct)[0];
    const aucLeader = [...prod].sort((a, b) => (b.rocAuc ?? 0) - (a.rocAuc ?? 0))[0];
    const deepRows = prod.filter((r) => classifyFamily(r.model) === "deep");
    const bestDeep = [...deepRows].sort((a, b) => b.accuracyPct - a.accuracyPct)[0] ?? null;
    return { accLeader, aucLeader, bestDeep };
  }, [benchmarks]);

  const activeCard =
    models?.modelCards.find((c) => c.id === selectedCardId) ?? models?.modelCards[0] ?? null;

  const exportSuiteParam = suite === "real" ? "production" : "synthetic";

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-50">
            Model Comparison, Leaderboard & SLSQP Optimization
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Verified holdout benchmark leaderboard, ROC/PR profile charts, expandable model details,
            and constrained SLSQP ensemble weight exploration.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <div className="inline-flex items-center p-1 rounded-lg bg-slate-900 border border-white/10">
            <button
              type="button"
              onClick={() => setSuite("real")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                suite === "real" ? "bg-sky-500 text-slate-950" : "text-slate-300 hover:text-white"
              }`}
            >
              Real-Image Holdout ({benchmarks?.realImageProduction.sampleCount ?? 75})
            </button>
            <button
              type="button"
              onClick={() => setSuite("synthetic")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                suite === "synthetic"
                  ? "bg-amber-400 text-slate-950"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              Synthetic CI ({benchmarks?.ciSmokeTestSynthetic.sampleCount ?? 30})
            </button>
          </div>

          <a
            href={`/api/benchmarks/export?suite=${exportSuiteParam}&format=csv`}
            download={`celeba_${exportSuiteParam}_benchmarks.csv`}
            className="px-3 py-2 rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 whitespace-nowrap"
          >
            <Download className="w-3.5 h-3.5 text-sky-400" />
            <span>Export CSV</span>
          </a>
          <a
            href={`/api/benchmarks/export?suite=${exportSuiteParam}&format=json`}
            download={`celeba_${exportSuiteParam}_benchmarks.json`}
            className="px-3 py-2 rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 whitespace-nowrap"
          >
            <Download className="w-3.5 h-3.5 text-emerald-400" />
            <span>Export JSON</span>
          </a>
        </div>
      </div>

      {suite === "synthetic" && (
        <div className="p-3.5 rounded-xl border border-amber-500/40 bg-amber-950/25 text-xs text-amber-200 flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <strong className="text-amber-300">Synthetic CI Smoke-Test Mode:</strong> Displaying{" "}
            <span className="font-mono">reports/benchmarks/ci_smoke_test_benchmark.csv</span> (30
            generated validation samples). Strictly separated from verified real-image CelebA
            holdout benchmarks.
          </div>
        </div>
      )}

      {/* Sortable & Filterable Performance Leaderboard with Expandable Model Details */}
      <ScrollReveal>
        <GlassCard className="space-y-4">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
            <div>
              <h3 className="text-base font-semibold text-slate-100">
                Verified Performance Leaderboard
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Source:{" "}
                <span className="font-mono text-slate-300">
                  {suite === "real"
                    ? benchmarks?.realImageProduction.sourceFile
                    : benchmarks?.ciSmokeTestSynthetic.sourceFile}
                </span>{" "}
                · Click any row to expand confusion matrix, checkpoint status, and evaluation details.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <div className="relative">
                <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Filter models..."
                  aria-label="Filter models in leaderboard"
                  className="pl-8 pr-3 py-1.5 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-400 w-44"
                />
              </div>

              <select
                value={familyFilter}
                onChange={(e) => setFamilyFilter(e.target.value as ArchFamily)}
                aria-label="Filter by architecture family"
                className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400"
              >
                <option value="all">All Architectures</option>
                <option value="deep">Deep Learning (PyTorch)</option>
                <option value="tree">Tree / XGBoost</option>
                <option value="ensemble">Ensembles</option>
              </select>

              <select
                value={ckptFilter}
                onChange={(e) => setCkptFilter(e.target.value as CheckpointFilter)}
                aria-label="Filter by checkpoint availability"
                className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400"
              >
                <option value="all">All Checkpoints</option>
                <option value="available">Checkpoint On Disk / Ready</option>
                <option value="missing">Missing / No Artifact</option>
              </select>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-white/10 text-slate-400">
                  <th className="py-2.5 pr-3 font-medium">
                    <button
                      type="button"
                      onClick={() => handleSort("model")}
                      className="inline-flex items-center gap-1 hover:text-slate-100"
                    >
                      <span>Model</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </button>
                  </th>
                  <th className="py-2.5 px-2.5 font-medium text-right">
                    <button
                      type="button"
                      onClick={() => handleSort("accuracyPct")}
                      className="inline-flex items-center gap-1 hover:text-slate-100"
                    >
                      <span>Accuracy</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </button>
                  </th>
                  <th className="py-2.5 px-2.5 font-medium text-right">
                    <button
                      type="button"
                      onClick={() => handleSort("precision")}
                      className="inline-flex items-center gap-1 hover:text-slate-100"
                    >
                      <span>Precision</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </button>
                  </th>
                  <th className="py-2.5 px-2.5 font-medium text-right">
                    <button
                      type="button"
                      onClick={() => handleSort("recall")}
                      className="inline-flex items-center gap-1 hover:text-slate-100"
                    >
                      <span>Recall</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </button>
                  </th>
                  <th className="py-2.5 px-2.5 font-medium text-right">
                    <button
                      type="button"
                      onClick={() => handleSort("f1Score")}
                      className="inline-flex items-center gap-1 hover:text-slate-100"
                    >
                      <span>F1-Score</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </button>
                  </th>
                  <th className="py-2.5 px-2.5 font-medium text-right">
                    <button
                      type="button"
                      onClick={() => handleSort("rocAuc")}
                      className="inline-flex items-center gap-1 hover:text-slate-100"
                    >
                      <span>ROC-AUC</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </button>
                  </th>
                  <th className="py-2.5 px-2.5 font-medium text-right">
                    <button
                      type="button"
                      onClick={() => handleSort("prAuc")}
                      className="inline-flex items-center gap-1 hover:text-slate-100"
                    >
                      <span>PR-AUC</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </button>
                  </th>
                  <th className="py-2.5 px-2.5 font-medium text-right">
                    <button
                      type="button"
                      onClick={() => handleSort("testLoss")}
                      className="inline-flex items-center gap-1 hover:text-slate-100"
                    >
                      <span>Loss</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </button>
                  </th>
                  <th className="py-2.5 pl-2.5 font-medium text-right">Checkpoint</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {filteredSortedLeaderboard.map(({ row, ckpt, checkpointStatus }) => {
                  const isExpanded = expandedModel === row.model;
                  return (
                    <Fragment key={row.model}>
                      <tr
                        onClick={() => setExpandedModel(isExpanded ? null : row.model)}
                        className="hover:bg-white/[0.04] cursor-pointer transition-colors"
                      >
                        <td className="py-3 pr-3 font-medium text-slate-100">
                          <div className="flex items-center gap-2">
                            {isExpanded ? (
                              <ChevronUp className="w-3.5 h-3.5 text-sky-400 shrink-0" />
                            ) : (
                              <ChevronDown className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                            )}
                            <div>
                              <div>{row.model}</div>
                              <div className="text-[11px] text-slate-400 font-normal">
                                {row.strategy || "Evaluated holdout configuration"}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="py-3 px-2.5 text-right font-mono tabular-nums text-sky-300 font-semibold">
                          {row.accuracyPct.toFixed(2)}%
                        </td>
                        <td className="py-3 px-2.5 text-right font-mono tabular-nums text-slate-200">
                          {row.precision.toFixed(4)}
                        </td>
                        <td className="py-3 px-2.5 text-right font-mono tabular-nums text-slate-200">
                          {row.recall.toFixed(4)}
                        </td>
                        <td className="py-3 px-2.5 text-right font-mono tabular-nums text-emerald-300">
                          {row.f1Score.toFixed(4)}
                        </td>
                        <td className="py-3 px-2.5 text-right font-mono tabular-nums text-slate-100">
                          {row.rocAuc !== null ? row.rocAuc.toFixed(4) : <span className="text-slate-500">N/A</span>}
                        </td>
                        <td className="py-3 px-2.5 text-right font-mono tabular-nums text-slate-300">
                          {row.prAuc !== null ? row.prAuc.toFixed(4) : <span className="text-slate-500">N/A</span>}
                        </td>
                        <td className="py-3 px-2.5 text-right font-mono tabular-nums text-slate-400">
                          {row.testLoss.toFixed(4)}
                        </td>
                        <td className="py-3 pl-2.5 text-right whitespace-nowrap">
                          {checkpointStatus === "ready" ? (
                            <span className="text-emerald-400 font-medium inline-flex items-center gap-1">
                              <CheckCircle2 className="w-3.5 h-3.5" /> Ready
                            </span>
                          ) : checkpointStatus === "on_disk" ? (
                            <span className="text-sky-300 font-medium">
                              On Disk ({ckpt?.sizeMb} MB)
                            </span>
                          ) : checkpointStatus === "missing" ? (
                            <span className="text-amber-400 inline-flex items-center gap-1">
                              <XCircle className="w-3.5 h-3.5" /> Missing
                            </span>
                          ) : (
                            <span className="text-slate-500">Unavailable</span>
                          )}
                        </td>
                      </tr>

                      {isExpanded && (
                        <tr className="bg-slate-950/70">
                          <td colSpan={9} className="p-4">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                              <div className="space-y-1.5">
                                <div className="font-semibold text-slate-200">
                                  Checkpoint & Runtime Details
                                </div>
                                <div className="text-slate-400">
                                  Artifact:{" "}
                                  <span className="font-mono text-slate-200">
                                    {ckpt?.filename ?? "No serialized binary in models/checkpoints/"}
                                  </span>
                                </div>
                                <div className="text-slate-400">
                                  Size:{" "}
                                  <span className="font-mono text-slate-200">
                                    {ckpt?.sizeMb ? `${ckpt.sizeMb} MB` : "Unavailable"}
                                  </span>
                                </div>
                                <div className="text-slate-400">
                                  Inference Latency:{" "}
                                  <span className="font-mono text-slate-300">
                                    {models?.mlBackend.online
                                      ? "Measured live in Image Analysis"
                                      : "Unavailable (FastAPI offline)"}
                                  </span>
                                </div>
                              </div>

                              <div className="space-y-1.5">
                                <div className="font-semibold text-slate-200">
                                  Evaluation Strategy & Metrics
                                </div>
                                <div className="text-slate-400">
                                  Protocol:{" "}
                                  <span className="text-slate-200">
                                    {row.strategy || "Identity-disjoint holdout evaluation"}
                                  </span>
                                </div>
                                <div className="text-slate-400 font-mono tabular-nums">
                                  Test Loss: {row.testLoss.toFixed(4)} · ROC-AUC:{" "}
                                  {row.rocAuc !== null ? row.rocAuc.toFixed(4) : "N/A"} · PR-AUC:{" "}
                                  {row.prAuc !== null ? row.prAuc.toFixed(4) : "N/A"}
                                </div>
                              </div>

                              <div className="space-y-1.5">
                                <div className="font-semibold text-slate-200">
                                  Confusion Matrix (Holdout)
                                </div>
                                {row.confusionMatrix ? (
                                  <div className="grid grid-cols-2 gap-1.5 max-w-[220px] font-mono tabular-nums text-[11px]">
                                    <div className="p-2 rounded bg-emerald-500/15 border border-emerald-400/25 text-emerald-200">
                                      TN: {row.confusionMatrix[0][0]}
                                    </div>
                                    <div className="p-2 rounded bg-amber-500/15 border border-amber-400/25 text-amber-200">
                                      FP: {row.confusionMatrix[0][1]}
                                    </div>
                                    <div className="p-2 rounded bg-amber-500/15 border border-amber-400/25 text-amber-200">
                                      FN: {row.confusionMatrix[1][0]}
                                    </div>
                                    <div className="p-2 rounded bg-sky-500/15 border border-sky-400/25 text-sky-200">
                                      TP: {row.confusionMatrix[1][1]}
                                    </div>
                                  </div>
                                ) : (
                                  <div className="text-slate-500">
                                    Confusion matrix not recorded in this benchmark artifact.
                                  </div>
                                )}
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
                {filteredSortedLeaderboard.length === 0 && (
                  <tr>
                    <td colSpan={9} className="py-8 text-center text-slate-400">
                      No benchmarked models match the active search and checkpoint filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </GlassCard>
      </ScrollReveal>

      {/* Scroll-Triggered Animated Charts Grid */}
      <ScrollReveal>
        {(isVisible) => (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            <GlassCard className="xl:col-span-2">
              <h3 className="text-base font-semibold text-slate-100 mb-1">
                Accuracy, Precision, Recall & F1 Comparison
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                {suite === "real"
                  ? "Verified real-image holdout metrics from production_benchmark.csv"
                  : "Synthetic CI smoke-test validation metrics from ci_smoke_test_benchmark.csv"}
              </p>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} margin={{ top: 8, right: 12, left: -12, bottom: 48 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: "#94A3B8", fontSize: 10 }}
                      interval={0}
                      angle={-25}
                      textAnchor="end"
                      height={72}
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
                    <Legend wrapperStyle={{ fontSize: "12px" }} />
                    <Bar dataKey="Accuracy" fill="#38BDF8" radius={[3, 3, 0, 0]} isAnimationActive={isVisible} />
                    <Bar dataKey="Precision" fill="#34D399" radius={[3, 3, 0, 0]} isAnimationActive={isVisible} />
                    <Bar dataKey="Recall" fill="#818CF8" radius={[3, 3, 0, 0]} isAnimationActive={isVisible} />
                    <Bar dataKey="F1" fill="#FBBF24" radius={[3, 3, 0, 0]} isAnimationActive={isVisible} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>

            <GlassCard>
              <h3 className="text-base font-semibold text-slate-100 mb-1">
                Multi-Metric Radar Profile
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                ViT vs Residual CNN vs XGBoost vs Ensemble
              </p>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(148,163,184,0.2)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: "#CBD5E1", fontSize: 11 }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={{ fill: "#64748B", fontSize: 10 }} />
                    <Radar
                      name="Ensemble+TTA"
                      dataKey="Ensemble"
                      stroke="#38BDF8"
                      fill="#38BDF8"
                      fillOpacity={0.25}
                      isAnimationActive={isVisible}
                    />
                    <Radar
                      name="XGBoost"
                      dataKey="XGBoost"
                      stroke="#34D399"
                      fill="#34D399"
                      fillOpacity={0.2}
                      isAnimationActive={isVisible}
                    />
                    <Radar
                      name="Residual CNN"
                      dataKey="CNN"
                      stroke="#818CF8"
                      fill="#818CF8"
                      fillOpacity={0.18}
                      isAnimationActive={isVisible}
                    />
                    <Radar
                      name="ViT"
                      dataKey="ViT"
                      stroke="#FBBF24"
                      fill="#FBBF24"
                      fillOpacity={0.15}
                      isAnimationActive={isVisible}
                    />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>
          </div>
        )}
      </ScrollReveal>

      {/* ROC-AUC & PR-AUC Comparison + Research Insights */}
      <ScrollReveal>
        {(isVisible) => (
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
            <GlassCard className="xl:col-span-7 space-y-3">
              <div>
                <h3 className="text-base font-semibold text-slate-100">
                  ROC-AUC & Precision-Recall AUC Profile
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Discrimination & ranking quality across threshold-independent verified metrics (×100 scale)
                </p>
              </div>
              {rocPrData.length > 0 ? (
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={rocPrData} margin={{ top: 8, right: 12, left: -12, bottom: 42 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                      <XAxis
                        dataKey="name"
                        tick={{ fill: "#94A3B8", fontSize: 10 }}
                        interval={0}
                        angle={-22}
                        textAnchor="end"
                        height={64}
                      />
                      <YAxis domain={[0, 100]} tick={{ fill: "#94A3B8", fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0F172A",
                          borderColor: "rgba(255,255,255,0.12)",
                          borderRadius: "8px",
                          color: "#F8FAFC",
                          fontSize: "12px",
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: "11px" }} />
                      <Bar dataKey="ROC-AUC" fill="#38BDF8" radius={[4, 4, 0, 0]} isAnimationActive={isVisible} />
                      <Bar dataKey="PR-AUC" fill="#34D399" radius={[4, 4, 0, 0]} isAnimationActive={isVisible} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="py-10 text-xs text-slate-400 text-center">
                  ROC-AUC and PR-AUC values are unavailable in this benchmark artifact.
                </div>
              )}
            </GlassCard>

            <GlassCard className="xl:col-span-5 space-y-3">
              <div className="flex items-center gap-2">
                <FlaskConical className="w-4 h-4 text-sky-400" />
                <h3 className="text-base font-semibold text-slate-100">
                  Verified Research Insights & Trade-Offs
                </h3>
              </div>

              <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
                {researchInsights && (
                  <>
                    <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
                      <div className="font-semibold text-sky-300">
                        Holdout Accuracy vs Ranking Leader
                      </div>
                      <p className="mt-1 text-slate-400">
                        <strong className="text-slate-200">{researchInsights.accLeader.model}</strong>{" "}
                        leads real-image holdout accuracy at{" "}
                        <span className="font-mono text-slate-100">
                          {researchInsights.accLeader.accuracyPct.toFixed(2)}%
                        </span>
                        , while{" "}
                        <strong className="text-slate-200">{researchInsights.aucLeader.model}</strong>{" "}
                        achieves top probability ranking (ROC-AUC{" "}
                        <span className="font-mono text-slate-100">
                          {researchInsights.aucLeader.rocAuc?.toFixed(4) ?? "N/A"}
                        </span>
                        ).
                      </p>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
                      <div className="font-semibold text-emerald-300">
                        Small-Sample Deep Learning Trade-Off
                      </div>
                      <p className="mt-1 text-slate-400">
                        On the 75-face identity-disjoint holdout, gradient-boosted tabular features
                        outperform end-to-end neural architectures (best deep model:{" "}
                        <strong className="text-slate-200">
                          {researchInsights.bestDeep?.model ?? "N/A"}
                        </strong>{" "}
                        at{" "}
                        <span className="font-mono text-slate-100">
                          {researchInsights.bestDeep
                            ? `${researchInsights.bestDeep.accuracyPct.toFixed(2)}%`
                            : "N/A"}
                        </span>
                        ), illustrating higher sample complexity for ViT/BiLSTM without large-scale
                        pretraining.
                      </p>
                    </div>
                  </>
                )}

                <div className="p-3 rounded-lg bg-amber-950/20 border border-amber-500/30">
                  <div className="font-semibold text-amber-300">Evaluation Limitations & Registry Caveat</div>
                  <p className="mt-1 text-amber-100/80">
                    Legacy metrics in <span className="font-mono">model_registry.json</span> reflect
                    an earlier reference run and differ from the verified 75-sample{" "}
                    <span className="font-mono">production_benchmark.csv</span> holdout. Missing
                    checkpoints (<span className="font-mono">residual_cnn.pt</span>,{" "}
                    <span className="font-mono">ann.pt</span>) remain disabled for live inference.
                  </p>
                </div>
              </div>
            </GlassCard>
          </div>
        )}
      </ScrollReveal>

      {/* Interactive SLSQP Ensemble Weight Explorer */}
      <ScrollReveal>
        {(isVisible) => (
          <GlassCard className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="text-base font-semibold text-slate-100 inline-flex items-center gap-2">
                  <SlidersHorizontal className="w-4 h-4 text-sky-400" />
                  SLSQP Ensemble Weight Allocation Explorer
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Objective: <span className="font-mono text-slate-300">min_w LogLoss(OOF_val)</span>{" "}
                  s.t. <span className="font-mono text-slate-300">w_i &ge; 0, &Sigma;w_i = 1</span>{" "}
                  calibrated on 5-fold <span className="font-mono">StratifiedGroupKFold</span>{" "}
                  validation predictions (held-out test set untouched).
                </p>
              </div>

              <div className="flex items-center gap-2 self-start">
                <span
                  className={`px-2.5 py-1 rounded-md text-[11px] font-medium border ${
                    !normalizedBlend.valid
                      ? "bg-red-950/40 border-red-500/40 text-red-300"
                      : isVerifiedConfig
                        ? "bg-emerald-950/40 border-emerald-500/40 text-emerald-300"
                        : "bg-amber-950/40 border-amber-500/40 text-amber-300"
                  }`}
                >
                  {!normalizedBlend.valid
                    ? "Invalid Constraints (Sum = 0)"
                    : isVerifiedConfig
                      ? "Verified SLSQP Configuration"
                      : "Hypothetical Configuration (Test Metrics Unchanged)"}
                </span>
                <button
                  type="button"
                  onClick={resetWeightsToVerified}
                  className="px-3 py-1.5 min-h-[34px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 whitespace-nowrap"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Reset Verified</span>
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-center">
              <div className="lg:col-span-7 space-y-3">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div className="p-3.5 rounded-lg bg-slate-950/60 border border-white/10 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <label htmlFor="weightXgbSlider" className="font-medium text-emerald-300">
                        5-Fold XGBoost
                      </label>
                      <span className="font-mono font-semibold text-slate-100 tabular-nums">
                        {normalizedBlend.xgb.toFixed(1)}%
                      </span>
                    </div>
                    <input
                      id="weightXgbSlider"
                      type="range"
                      min={0}
                      max={100}
                      step={1}
                      value={weightXgb}
                      onChange={(e) => setWeightXgb(Number(e.target.value))}
                      className="w-full accent-emerald-400 cursor-pointer"
                    />
                    <div className="text-[11px] font-mono text-slate-400 tabular-nums">
                      Verified: {(verifiedWeights.xgb * 100).toFixed(2)}%
                    </div>
                  </div>

                  <div className="p-3.5 rounded-lg bg-slate-950/60 border border-white/10 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <label htmlFor="weightCnnSlider" className="font-medium text-indigo-300">
                        Residual CNN
                      </label>
                      <span className="font-mono font-semibold text-slate-100 tabular-nums">
                        {normalizedBlend.cnn.toFixed(1)}%
                      </span>
                    </div>
                    <input
                      id="weightCnnSlider"
                      type="range"
                      min={0}
                      max={100}
                      step={1}
                      value={weightCnn}
                      onChange={(e) => setWeightCnn(Number(e.target.value))}
                      className="w-full accent-indigo-400 cursor-pointer"
                    />
                    <div className="text-[11px] font-mono text-slate-400 tabular-nums">
                      Verified: {(verifiedWeights.cnn * 100).toFixed(2)}%
                    </div>
                  </div>

                  <div className="p-3.5 rounded-lg bg-slate-950/60 border border-white/10 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <label htmlFor="weightVitSlider" className="font-medium text-amber-300">
                        Vision Transformer
                      </label>
                      <span className="font-mono font-semibold text-slate-100 tabular-nums">
                        {normalizedBlend.vit.toFixed(1)}%
                      </span>
                    </div>
                    <input
                      id="weightVitSlider"
                      type="range"
                      min={0}
                      max={100}
                      step={1}
                      value={weightVit}
                      onChange={(e) => setWeightVit(Number(e.target.value))}
                      className="w-full accent-amber-400 cursor-pointer"
                    />
                    <div className="text-[11px] font-mono text-slate-400 tabular-nums">
                      Verified: {(verifiedWeights.vit * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>

                {/* Simplex Allocation Proportion Bar */}
                <div className="space-y-1.5">
                  <div className="h-3 w-full rounded-full overflow-hidden bg-slate-950 border border-white/10 flex">
                    <div
                      style={{ width: `${normalizedBlend.xgb}%` }}
                      className="bg-emerald-400 transition-all duration-200"
                    />
                    <div
                      style={{ width: `${normalizedBlend.cnn}%` }}
                      className="bg-indigo-400 transition-all duration-200"
                    />
                    <div
                      style={{ width: `${normalizedBlend.vit}%` }}
                      className="bg-amber-400 transition-all duration-200"
                    />
                  </div>
                  <div className="text-[11px] text-slate-400 flex flex-wrap items-center justify-between gap-2">
                    <span>
                      Changing weights does not fabricate or alter held-out test metrics. Run OOF
                      re-evaluation offline to measure new blends.
                    </span>
                    <span className="font-mono text-slate-300 tabular-nums">
                      Simplex Sum: {normalizedBlend.valid ? "1.0000" : "0.0000"} · Run:{" "}
                      {benchmarks?.latestTrainingRun.elapsedSeconds
                        ? `${benchmarks.latestTrainingRun.elapsedSeconds.toFixed(1)}s`
                        : "recorded"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Verified vs Current Weight Bar Chart */}
              <div className="lg:col-span-5 h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={slsqpComparisonChartData}
                    margin={{ top: 6, right: 8, left: -16, bottom: 4 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                    <XAxis dataKey="estimator" tick={{ fill: "#CBD5E1", fontSize: 10 }} />
                    <YAxis domain={[0, 100]} unit="%" tick={{ fill: "#94A3B8", fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0F172A",
                        borderColor: "rgba(255,255,255,0.12)",
                        borderRadius: "8px",
                        color: "#F8FAFC",
                        fontSize: "12px",
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Bar
                      dataKey="Verified"
                      name="Verified SLSQP (%)"
                      fill="#38BDF8"
                      radius={[3, 3, 0, 0]}
                      isAnimationActive={isVisible}
                    />
                    <Bar
                      dataKey="Current"
                      name="Explorer (%)"
                      fill="#34D399"
                      radius={[3, 3, 0, 0]}
                      isAnimationActive={isVisible}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </GlassCard>
        )}
      </ScrollReveal>

      {/* Checkpoint Inventory + Model Registry */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <GlassCard className="lg:col-span-5 space-y-4">
          <div>
            <h3 className="text-base font-semibold text-slate-100">
              PyTorch Checkpoint Files (models/checkpoints/)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              FastAPI service refuses inference when a checkpoint file is missing.
            </p>
          </div>

          <div className="divide-y divide-white/10">
            {checkpoints.map((ckpt) => (
              <div key={ckpt.filename} className="py-3 flex items-center justify-between text-sm">
                <div>
                  <div className="font-medium text-slate-100">{ckpt.architecture}</div>
                  <div className="text-xs font-mono text-slate-400">{ckpt.filename}</div>
                </div>
                {ckpt.exists ? (
                  <div className="text-right">
                    <span className="text-xs font-medium text-emerald-400 inline-flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" /> On Disk
                    </span>
                    <div className="text-xs font-mono text-slate-400 tabular-nums">
                      {ckpt.sizeMb} MB
                    </div>
                  </div>
                ) : (
                  <span className="text-xs font-medium text-slate-400 inline-flex items-center gap-1">
                    <XCircle className="w-3.5 h-3.5 text-amber-400" /> Not Present
                  </span>
                )}
              </div>
            ))}
          </div>
        </GlassCard>

        <GlassCard className="lg:col-span-7 space-y-4">
          <div>
            <h3 className="text-base font-semibold text-slate-100">
              Legacy Model Registry Reference (models/model_registry.json)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Historical registry metadata · Strategy: {models?.registry.identity_isolation_strategy}{" "}
              (kept separate from verified 75-sample holdout above)
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-white/10 text-slate-400">
                  <th className="py-2 pr-3 font-medium">Architecture</th>
                  <th className="py-2 px-3 font-medium">Strategy</th>
                  <th className="py-2 px-3 font-medium text-right">Accuracy</th>
                  <th className="py-2 px-3 font-medium text-right">F1-Score</th>
                  <th className="py-2 pl-3 font-medium">Role</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {(models?.registry.evaluated_models ?? []).map((entry) => (
                  <tr key={entry.id} className="hover:bg-white/[0.03]">
                    <td className="py-2.5 pr-3 font-medium text-slate-100">{entry.name}</td>
                    <td className="py-2.5 px-3 text-slate-300">{entry.strategy}</td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-sky-300">
                      {entry.accuracy}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-200">
                      {entry.f1_score.toFixed(4)}
                    </td>
                    <td className="py-2.5 pl-3 text-slate-400">{entry.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </GlassCard>
      </div>

      {/* Model Cards & YAML Hyperparameters */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GlassCard className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <h3 className="text-base font-semibold text-slate-100">Model Cards</h3>
            <div className="flex flex-wrap gap-1.5">
              {(models?.modelCards ?? []).map((card) => (
                <button
                  key={card.id}
                  type="button"
                  onClick={() => setSelectedCardId(card.id)}
                  className={`px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    activeCard?.id === card.id
                      ? "bg-sky-500 text-slate-950"
                      : "bg-slate-950 text-slate-300 border border-white/10 hover:text-white"
                  }`}
                >
                  {card.filename.replace("_card.md", "")}
                </button>
              ))}
            </div>
          </div>
          <pre className="p-4 rounded-lg bg-slate-950/80 border border-white/10 text-xs font-mono text-slate-300 whitespace-pre-wrap max-h-80 overflow-y-auto leading-relaxed">
            {activeCard?.content ?? "No model card loaded."}
          </pre>
        </GlassCard>

        <GlassCard className="space-y-4">
          <div className="flex items-center gap-2">
            <FileCode2 className="w-4 h-4 text-sky-400" />
            <h3 className="text-base font-semibold text-slate-100">
              Reproducibility Hyperparameters (reproducibility/config.yaml)
            </h3>
          </div>
          <pre className="p-4 rounded-lg bg-slate-950/80 border border-white/10 text-xs font-mono text-slate-300 whitespace-pre-wrap max-h-80 overflow-y-auto leading-relaxed">
            {models?.reproducibilityConfigYaml ?? "config.yaml unavailable."}
          </pre>
        </GlassCard>
      </div>
    </div>
  );
}
