import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Maximize2, Search } from "lucide-react";
import { FigureLightbox, type LightboxFigure } from "../components/FigureLightbox";
import { GlassCard } from "../components/GlassCard";
import type { DatasetAnalyticsResponse, GallerySample } from "../types";

interface DatasetAnalyticsPageProps {
  analytics: DatasetAnalyticsResponse | null;
}

export function DatasetAnalyticsPage({ analytics }: DatasetAnalyticsPageProps) {
  const [samples, setSamples] = useState<GallerySample[]>([]);
  const [query, setQuery] = useState("");
  const [attrFilter, setAttrFilter] = useState<"all" | "smiling" | "eyeglasses" | "young" | "male">(
    "all"
  );
  const [activeFigure, setActiveFigure] = useState<LightboxFigure | null>(null);

  useEffect(() => {
    fetch("/api/gallery")
      .then((r) => (r.ok ? r.json() : []))
      .then((data: GallerySample[]) => {
        if (Array.isArray(data)) setSamples(data);
      })
      .catch(() => {});
  }, []);

  const filteredSamples = useMemo(() => {
    const q = query.trim().toLowerCase();
    return samples.filter((s) => {
      if (attrFilter === "smiling" && !s.smiling) return false;
      if (attrFilter === "eyeglasses" && !s.eyeglasses) return false;
      if (attrFilter === "young" && !s.young) return false;
      if (attrFilter === "male" && !s.male) return false;
      if (!q) return true;
      return (
        s.filename.toLowerCase().includes(q) ||
        String(s.cluster_id).includes(q) ||
        s.attributes.some((a) => a.toLowerCase().includes(q))
      );
    });
  }, [samples, query, attrFilter]);

  const datasetFigures = useMemo(
    () => (analytics?.figures ?? []).filter((f) => f.category === "dataset"),
    [analytics]
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-50">CelebA Dataset & Split Analytics</h2>
        <p className="text-sm text-slate-400 mt-1">
          Identity-disjoint partitioning, attribute distributions, and curated reference samples from
          repository metadata.
        </p>
      </div>

      {/* Manifest KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <GlassCard>
          <div className="text-xs text-slate-400">Total CelebA Images</div>
          <div className="text-2xl font-bold text-slate-50 mt-1.5 tabular-nums">
            {(analytics?.identityFileStats.totalMappings || analytics?.manifest.total_images || 0).toLocaleString()}
          </div>
          <div className="text-xs text-slate-400 mt-1">identity_CelebA.txt verified lines</div>
        </GlassCard>

        <GlassCard>
          <div className="text-xs text-slate-400">Unique Identities</div>
          <div className="text-2xl font-bold text-sky-300 mt-1.5 tabular-nums">
            {(analytics?.identityFileStats.uniqueIdentities || analytics?.manifest.total_unique_identities || 0).toLocaleString()}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {analytics?.identityFileStats.celebritiesCatalogCount.toLocaleString() ?? "1,090"} cataloged in celebrities.csv
          </div>
        </GlassCard>

        <GlassCard>
          <div className="text-xs text-slate-400">Canonical Input Tensor</div>
          <div className="text-2xl font-bold text-slate-50 mt-1.5 font-mono tabular-nums">
            {analytics?.manifest.canonical_target_size?.join("×") ?? "64×64×3"}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Source aligned: {analytics?.manifest.resolution?.join("×") ?? "218×178×3"}
          </div>
        </GlassCard>

        <GlassCard>
          <div className="text-xs text-slate-400">Split Identity Leakage</div>
          <div className="text-2xl font-bold text-emerald-400 mt-1.5 tabular-nums">
            {analytics?.splitManifestSummary.identityOverlapCount ?? 0} overlaps
          </div>
          <div className="text-xs text-slate-400 mt-1">
            StratifiedGroupKFold across {analytics?.splitManifestSummary.totalRows ?? 0} manifest samples
          </div>
        </GlassCard>
      </div>

      {/* Split & Attribute Distribution Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <GlassCard className="lg:col-span-5 space-y-4">
          <div>
            <h3 className="text-base font-semibold text-slate-100">
              Identity-Disjoint Split Distribution
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Source: reproducibility/split_manifest.csv
            </p>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={analytics?.splitManifestSummary.splits ?? []}
                margin={{ top: 8, right: 12, left: -12, bottom: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                <XAxis dataKey="split" tick={{ fill: "#CBD5E1", fontSize: 12 }} />
                <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0F172A",
                    borderColor: "rgba(255,255,255,0.12)",
                    borderRadius: "8px",
                    color: "#F8FAFC",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="images" name="Images" fill="#38BDF8" radius={[4, 4, 0, 0]} />
                <Bar
                  dataKey="uniqueIdentities"
                  name="Unique Identities"
                  fill="#818CF8"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>

        <GlassCard className="lg:col-span-7 space-y-4">
          <div>
            <h3 className="text-base font-semibold text-slate-100">
              Top Attribute Frequencies in Reference Subset ({analytics?.galleryStats.totalSamples ?? 0} samples)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Source: frontend/gallery_data.json
            </p>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={analytics?.galleryStats.attributeCounts ?? []}
                layout="vertical"
                margin={{ top: 4, right: 16, left: 24, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                <XAxis type="number" tick={{ fill: "#94A3B8", fontSize: 11 }} />
                <YAxis
                  dataKey="attribute"
                  type="category"
                  width={130}
                  tick={{ fill: "#CBD5E1", fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0F172A",
                    borderColor: "rgba(255,255,255,0.12)",
                    borderRadius: "8px",
                    color: "#F8FAFC",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="count" name="Sample Count" fill="#34D399" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>
      </div>

      {/* Repository Dataset Visual Figures */}
      {datasetFigures.length > 0 && (
        <GlassCard className="space-y-4">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-base font-semibold text-slate-100">
              Repository Clustering & Demographic Figures
            </h3>
            <span className="text-xs text-slate-400">Click any figure to inspect full size</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {datasetFigures.map((fig) => (
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
                    referrerPolicy="no-referrer"
                    className="w-full h-52 object-contain bg-white"
                  />
                  <span className="absolute top-2 right-2 p-1.5 rounded-md bg-slate-900/80 text-slate-200 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Maximize2 className="w-3.5 h-3.5" />
                  </span>
                </div>
                <div className="p-3 text-xs font-medium text-slate-200">{fig.title}</div>
              </button>
            ))}
          </div>
        </GlassCard>
      )}

      {/* Curated Attribute Sample Explorer */}
      <GlassCard className="space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-slate-100">
              Curated Reference Sample Explorer
            </h3>
            <p className="text-xs text-slate-400">
              Showing {filteredSamples.length} of {samples.length} anonymized CelebA validation
              samples
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter attributes..."
                className="pl-9 pr-3 py-1.5 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-400"
              />
            </div>

            {(["all", "smiling", "eyeglasses", "young", "male"] as const).map((tag) => (
              <button
                key={tag}
                type="button"
                onClick={() => setAttrFilter(tag)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium capitalize transition-colors ${
                  attrFilter === tag
                    ? "bg-sky-500 text-slate-950"
                    : "bg-slate-950 text-slate-300 border border-white/10 hover:text-white"
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
          {filteredSamples.slice(0, 24).map((sample) => (
            <div
              key={sample.id}
              className="rounded-lg border border-white/10 bg-slate-950/70 overflow-hidden"
            >
              <img
                src={sample.thumbnail}
                alt={`Reference face ${sample.id}`}
                referrerPolicy="no-referrer"
                className="w-full aspect-square object-cover"
              />
              <div className="p-2 text-[11px] space-y-0.5">
                <div className="font-mono text-slate-200 truncate">{sample.filename}</div>
                <div className="text-slate-400 truncate">
                  {sample.smiling ? "Smiling" : "Neutral"} ·{" "}
                  {sample.eyeglasses ? "Glasses" : "No Glasses"}
                </div>
              </div>
            </div>
          ))}
        </div>
      </GlassCard>

      <FigureLightbox figure={activeFigure} onClose={() => setActiveFigure(null)} />
    </div>
  );
}
