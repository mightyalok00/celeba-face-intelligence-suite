import { useMemo, useState } from "react";
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
import { CheckCircle2, FileCode2, XCircle } from "lucide-react";
import { GlassCard } from "../components/GlassCard";
import type { BenchmarkSuitesResponse, ModelsResponse } from "../types";

interface ModelComparisonPageProps {
  benchmarks: BenchmarkSuitesResponse | null;
  models: ModelsResponse | null;
}

export function ModelComparisonPage({ benchmarks, models }: ModelComparisonPageProps) {
  const [suite, setSuite] = useState<"real" | "synthetic">("real");
  const [selectedCardId, setSelectedCardId] = useState<string>("ensemble");

  const rows = useMemo(() => {
    if (!benchmarks) return [];
    return suite === "real"
      ? benchmarks.realImageProduction.rows
      : benchmarks.ciSmokeTestSynthetic.rows;
  }, [benchmarks, suite]);

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

  const activeCard =
    models?.modelCards.find((c) => c.id === selectedCardId) ?? models?.modelCards[0] ?? null;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-50">Deep Learning Model Comparison</h2>
          <p className="text-sm text-slate-400 mt-1">
            Compare PyTorch ANN, Spatial BiLSTM, Residual CNN, Vision Transformer, XGBoost, and
            SLSQP Ensemble + TTA across real-image and synthetic smoke-test benchmarks.
          </p>
        </div>

        <div className="inline-flex items-center p-1 rounded-lg bg-slate-900 border border-white/10 shrink-0">
          <button
            type="button"
            onClick={() => setSuite("real")}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
              suite === "real" ? "bg-sky-500 text-slate-950" : "text-slate-300 hover:text-white"
            }`}
          >
            Real-Image Benchmark (75 samples)
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
            Synthetic CI Smoke-Test (30 samples)
          </button>
        </div>
      </div>

      {/* Charts Grid */}
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
                <Bar dataKey="Accuracy" fill="#38BDF8" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Precision" fill="#34D399" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Recall" fill="#818CF8" radius={[3, 3, 0, 0]} />
                <Bar dataKey="F1" fill="#FBBF24" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>

        <GlassCard>
          <h3 className="text-base font-semibold text-slate-100 mb-1">Multi-Metric Radar Profile</h3>
          <p className="text-xs text-slate-400 mb-4">ViT vs Residual CNN vs XGBoost vs Ensemble</p>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(148,163,184,0.2)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "#CBD5E1", fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fill: "#64748B", fontSize: 10 }} />
                <Radar name="Ensemble+TTA" dataKey="Ensemble" stroke="#38BDF8" fill="#38BDF8" fillOpacity={0.25} />
                <Radar name="XGBoost" dataKey="XGBoost" stroke="#34D399" fill="#34D399" fillOpacity={0.2} />
                <Radar name="Residual CNN" dataKey="CNN" stroke="#818CF8" fill="#818CF8" fillOpacity={0.18} />
                <Radar name="ViT" dataKey="ViT" stroke="#FBBF24" fill="#FBBF24" fillOpacity={0.15} />
                <Legend wrapperStyle={{ fontSize: "11px" }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>
      </div>

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
            {(models?.checkpoints ?? []).map((ckpt) => (
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
              Model Registry Reference (models/model_registry.json)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Preserved registry metadata · Strategy: {models?.registry.identity_isolation_strategy}
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
