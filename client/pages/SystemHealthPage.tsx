import { useEffect, useState, type FormEvent } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Download,
  HardDrive,
  Link2,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import { GlassCard } from "../components/GlassCard";
import type { ApiEndpointDoc, SystemHealthResponse } from "../types";

interface SystemHealthPageProps {
  health: SystemHealthResponse | null;
  onRefresh: () => void;
}

export function SystemHealthPage({ health, onRefresh }: SystemHealthPageProps) {
  const [endpoints, setEndpoints] = useState<ApiEndpointDoc[]>([]);
  const [backendUrlInput, setBackendUrlInput] = useState("");
  const [backendSaving, setBackendSaving] = useState(false);
  const [backendFeedback, setBackendFeedback] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  useEffect(() => {
    if (health?.services.pythonMlBackend.url && !backendUrlInput) {
      setBackendUrlInput(health.services.pythonMlBackend.url);
    }
  }, [health?.services.pythonMlBackend.url, backendUrlInput]);

  useEffect(() => {
    fetch("/api/docs")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data && Array.isArray(data.endpoints)) {
          setEndpoints(data.endpoints);
        }
      })
      .catch(() => {});
  }, []);

  const handleSaveBackendUrl = async (e: FormEvent) => {
    e.preventDefault();
    setBackendSaving(true);
    setBackendFeedback(null);
    try {
      const resp = await fetch("/api/system/backend-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: backendUrlInput.trim() }),
      });
      const data = (await resp.json().catch(() => ({}))) as {
        message?: string;
        pythonMlBackend?: { verifiedCheckpoints?: string[]; latencyMs?: number | null };
      };
      if (!resp.ok) {
        setBackendFeedback({
          type: "error",
          message: data.message || `Failed to connect to FastAPI backend (HTTP ${resp.status}).`,
        });
      } else {
        const verifiedCount = data.pythonMlBackend?.verifiedCheckpoints?.length ?? 0;
        const latency = data.pythonMlBackend?.latencyMs;
        setBackendFeedback({
          type: "success",
          message: `Connected to CelebA FastAPI ML service (${verifiedCount} verified checkpoints${latency !== null && latency !== undefined ? ` · ${latency} ms` : ""}).`,
        });
        onRefresh();
      }
    } catch (err) {
      setBackendFeedback({
        type: "error",
        message: err instanceof Error ? err.message : "Network error while updating backend URL.",
      });
    } finally {
      setBackendSaving(false);
    }
  };

  const handleResetBackendUrl = async () => {
    setBackendSaving(true);
    setBackendFeedback(null);
    try {
      const resp = await fetch("/api/system/backend-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reset: true }),
      });
      const data = (await resp.json().catch(() => ({}))) as {
        pythonMlBackend?: { url?: string };
      };
      if (data.pythonMlBackend?.url) {
        setBackendUrlInput(data.pythonMlBackend.url);
      }
      setBackendFeedback({
        type: "success",
        message: `Reset FastAPI backend URL to server default (${data.pythonMlBackend?.url ?? "http://127.0.0.1:8001"}).`,
      });
      onRefresh();
    } catch (err) {
      setBackendFeedback({
        type: "error",
        message: err instanceof Error ? err.message : "Failed to reset backend URL.",
      });
    } finally {
      setBackendSaving(false);
    }
  };

  const mlBackend = health?.services.pythonMlBackend;
  const mlStateLabel =
    mlBackend?.connectionState === "connected_ready"
      ? "Connected & Ready"
      : mlBackend?.connectionState === "connected_no_checkpoints"
        ? "Connected (No Checkpoints)"
        : mlBackend?.connectionState === "connected_degraded"
          ? "Connected (Degraded)"
          : "Disconnected";

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-slate-50">System Health, Privacy & API Docs</h2>
          <p className="text-sm text-slate-400 mt-1">
            Live service telemetry, configurable FastAPI inference URL, checkpoint inventory, and
            documented backend endpoints.
          </p>
        </div>

        <button
          type="button"
          onClick={onRefresh}
          className="px-4 py-2 min-h-[40px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs sm:text-sm font-medium text-slate-100 inline-flex items-center gap-2 self-start"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh Health</span>
        </button>
      </div>

      {/* Live Services Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <GlassCard className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">Gemini Multimodal API</h3>
            {health?.services.geminiApi.configured ? (
              <span className="text-xs text-emerald-400 inline-flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> Configured
              </span>
            ) : (
              <span className="text-xs text-amber-400 inline-flex items-center gap-1">
                <XCircle className="w-3.5 h-3.5" /> Unconfigured
              </span>
            )}
          </div>
          <div className="text-xs text-slate-400 space-y-1 font-mono">
            <div>Model: {health?.services.geminiApi.defaultModel ?? "gemini-4-hydrogen"}</div>
            <div>Key Scope: Server environment only</div>
          </div>
        </GlassCard>

        <GlassCard className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-slate-100">Python FastAPI ML Backend</h3>
            {mlBackend?.connectionState === "connected_ready" ? (
              <span className="text-xs text-emerald-400 inline-flex items-center gap-1 shrink-0">
                <CheckCircle2 className="w-3.5 h-3.5" /> {mlStateLabel}
              </span>
            ) : (
              <span className="text-xs text-amber-400 inline-flex items-center gap-1 shrink-0">
                <XCircle className="w-3.5 h-3.5" /> {mlStateLabel}
              </span>
            )}
          </div>
          <div className="text-xs text-slate-400 space-y-1">
            <div className="font-mono truncate">{mlBackend?.url}</div>
            <div>
              {mlBackend?.error ||
                `Verified checkpoints: ${mlBackend?.verifiedCheckpoints.join(", ") || "None"}`}
            </div>
          </div>
        </GlassCard>

        <GlassCard className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">MySQL (celeba_db)</h3>
            {health?.services.mysqlDatabase.reachable ? (
              <span className="text-xs text-emerald-400 inline-flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> Connected
              </span>
            ) : (
              <span className="text-xs text-slate-400 inline-flex items-center gap-1">
                <XCircle className="w-3.5 h-3.5" /> Offline
              </span>
            )}
          </div>
          <div className="text-xs text-slate-400 space-y-1">
            <div>
              Host:{" "}
              <span className="font-mono">
                {health?.services.mysqlDatabase.host || "Unconfigured"}:
                {health?.services.mysqlDatabase.port ?? 3306}
              </span>
            </div>
            <div>{health?.services.mysqlDatabase.error || "Connected via environment credentials."}</div>
          </div>
        </GlassCard>
      </div>

      {/* Configurable FastAPI Inference Backend URL + Checkpoint Readiness */}
      <GlassCard className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Link2 className="w-5 h-5 text-sky-400" />
            <div>
              <h3 className="text-base font-semibold text-slate-100">
                FastAPI PyTorch Backend Configuration & Checkpoints
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Connect to a local FastAPI instance (<span className="font-mono">http://127.0.0.1:8001</span>) or a deployed public HTTPS FastAPI URL (Cloud Run, Hugging Face Spaces, or Cloudflare Tunnel).
              </p>
            </div>
          </div>
          {mlBackend?.latencyMs !== null && mlBackend?.latencyMs !== undefined && (
            <span className="text-xs font-mono text-emerald-400 tabular-nums shrink-0">
              Health latency: {mlBackend.latencyMs} ms
            </span>
          )}
        </div>

        <form onSubmit={handleSaveBackendUrl} className="flex flex-col sm:flex-row gap-2.5">
          <input
            type="url"
            value={backendUrlInput}
            onChange={(e) => setBackendUrlInput(e.target.value)}
            placeholder="https://your-celeba-fastapi.run.app or http://127.0.0.1:8001"
            aria-label="Python FastAPI ML Backend Base URL"
            className="flex-1 rounded-lg border border-white/15 bg-slate-950 px-3.5 py-2.5 text-xs sm:text-sm font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-400"
          />
          <button
            type="submit"
            disabled={backendSaving || !backendUrlInput.trim()}
            className="px-4 py-2.5 min-h-[42px] rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-40 text-slate-950 text-xs sm:text-sm font-semibold transition-colors whitespace-nowrap"
          >
            {backendSaving ? "Verifying /health..." : "Test & Connect"}
          </button>
          <button
            type="button"
            disabled={backendSaving}
            onClick={handleResetBackendUrl}
            className="px-3.5 py-2.5 min-h-[42px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 disabled:opacity-40 text-slate-200 text-xs font-medium inline-flex items-center justify-center gap-1.5 whitespace-nowrap"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Default</span>
          </button>
        </form>

        {backendFeedback && (
          <div
            className={`p-3 rounded-lg border text-xs flex items-start gap-2 ${
              backendFeedback.type === "success"
                ? "border-emerald-500/40 bg-emerald-950/30 text-emerald-200"
                : "border-amber-500/40 bg-amber-950/30 text-amber-200"
            }`}
          >
            {backendFeedback.type === "success" ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            ) : (
              <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            )}
            <span>{backendFeedback.message}</span>
          </div>
        )}

        {/* Checkpoint Availability Inventory */}
        <div className="pt-2 border-t border-white/10 space-y-2.5">
          <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
            <HardDrive className="w-4 h-4 text-sky-400" />
            <span>PyTorch Checkpoint Inventory (models/checkpoints/)</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {(health?.checkpoints ?? []).map((ckpt) => {
              const liveVerified = mlBackend?.verifiedCheckpoints.includes(ckpt.architecture) ?? false;
              return (
                <div
                  key={ckpt.filename}
                  className="p-3 rounded-lg bg-slate-950/60 border border-white/10 space-y-1.5 text-xs"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-slate-100 truncate">{ckpt.architecture}</span>
                    {liveVerified ? (
                      <span className="text-emerald-400 font-medium shrink-0">Ready</span>
                    ) : ckpt.exists ? (
                      <span className="text-sky-300 font-medium shrink-0">On Disk</span>
                    ) : (
                      <span className="text-amber-400 font-medium shrink-0">Missing</span>
                    )}
                  </div>
                  <div className="font-mono text-[11px] text-slate-400 truncate">
                    {ckpt.filename} {ckpt.sizeMb ? `· ${ckpt.sizeMb} MB` : "· Not found"}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    {liveVerified
                      ? "Loaded in live FastAPI engine"
                      : ckpt.exists
                        ? "Requires live FastAPI service"
                        : "Inference disabled (no weights)"}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </GlassCard>

      {/* Privacy & Security Policies */}
      <GlassCard className="space-y-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          <h3 className="text-base font-semibold text-slate-100">
            Security, Privacy & Rate-Limiting Safeguards
          </h3>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs text-slate-300">
          <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
            <div className="font-semibold text-slate-100">Zero Image Retention</div>
            <div className="text-slate-400 mt-1">
              {health?.privacyAndSecurity.imageRetention ??
                "In-memory single-request processing only."}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
            <div className="font-semibold text-slate-100">Upload & Magic-Byte Validation</div>
            <div className="text-slate-400 mt-1">
              Accepts authentic JPEG, PNG, and WebP signatures up to{" "}
              {health?.privacyAndSecurity.maxUploadMb ?? 10} MB.
            </div>
          </div>
          <div className="p-3 rounded-lg bg-slate-950/60 border border-white/10">
            <div className="font-semibold text-slate-100">API Rate Limiting</div>
            <div className="text-slate-400 mt-1">
              Read: {health?.privacyAndSecurity.rateLimiting.readWindowLimit} · Inference:{" "}
              {health?.privacyAndSecurity.rateLimiting.inferenceWindowLimit}
            </div>
          </div>
        </div>
      </GlassCard>

      {/* API Endpoint Documentation */}
      <GlassCard className="space-y-3">
        <h3 className="text-base font-semibold text-slate-100">Backend API Endpoint Catalog</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-white/10 text-slate-400">
                <th className="py-2 pr-3 font-medium">Method</th>
                <th className="py-2 px-3 font-medium">Endpoint</th>
                <th className="py-2 px-3 font-medium">Description</th>
                <th className="py-2 pl-3 font-medium">Policy</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {endpoints.map((ep) => (
                <tr key={`${ep.method}-${ep.path}`}>
                  <td className="py-2.5 pr-3 font-mono font-semibold text-sky-300">{ep.method}</td>
                  <td className="py-2.5 px-3 font-mono text-slate-100">{ep.path}</td>
                  <td className="py-2.5 px-3 text-slate-300">{ep.summary}</td>
                  <td className="py-2.5 pl-3 text-slate-400">{ep.authOrPolicy}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </GlassCard>

      {/* GitHub README Coordinated Header & Footer Banner Set */}
      <GlassCard className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-base font-semibold text-slate-100">
              GitHub Repository Header & Footer Banner Set
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              High-resolution minimalist off-white & navy SVG banners in{" "}
              <span className="font-mono text-slate-300">docs/assets/</span> (1600×400 px header
              and 1600×200 px footer).
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-xl border border-white/10 bg-slate-950/70 p-4 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div className="text-xs font-mono text-slate-300">
                docs/assets/celeba_header_banner.svg · 1600 × 400 px
              </div>
              <a
                href="/docs/assets/celeba_header_banner.svg"
                download="celeba_header_banner.svg"
                className="px-3 py-1.5 rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 whitespace-nowrap"
              >
                <Download className="w-3.5 h-3.5 text-sky-400" />
                <span>Download Header SVG</span>
              </a>
            </div>
            <img
              src="/docs/assets/celeba_header_banner.svg"
              alt="CelebA Face Intelligence Suite 1600x400 Header Banner"
              className="w-full rounded-lg border border-slate-200/20"
            />
          </div>

          <div className="rounded-xl border border-white/10 bg-slate-950/70 p-4 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div className="text-xs font-mono text-slate-300">
                docs/assets/celeba_footer_banner.svg · 1600 × 200 px
              </div>
              <a
                href="/docs/assets/celeba_footer_banner.svg"
                download="celeba_footer_banner.svg"
                className="px-3 py-1.5 rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 whitespace-nowrap"
              >
                <Download className="w-3.5 h-3.5 text-sky-400" />
                <span>Download Footer SVG</span>
              </a>
            </div>
            <img
              src="/docs/assets/celeba_footer_banner.svg"
              alt="CelebA Face Intelligence Suite 1600x200 Footer Banner"
              className="w-full rounded-lg border border-slate-200/20"
            />
          </div>
        </div>
      </GlassCard>

      {/* Reproducible Setup Instructions */}
      <GlassCard className="space-y-3">
        <h3 className="text-base font-semibold text-slate-100">Reproducible Setup Instructions</h3>
        <pre className="p-4 rounded-lg bg-slate-950/80 border border-white/10 text-xs font-mono text-slate-300 overflow-x-auto leading-relaxed">
{`# Option A: Full-Stack Multi-Container Deployment (Web + FastAPI ML + MySQL 8.0)
cp .env.example .env   # Configure GEMINI_API_KEY, MYSQL_PASSWORD, MYSQL_ROOT_PASSWORD
docker compose up --build

# Option B: Local Manual Services
# 1. Node.js + Vite Full-Stack Studio (Port 3000)
npm ci
npm run dev

# 2. Python FastAPI PyTorch/XGBoost Inference Microservice (Port 8001)
pip install -r requirements.txt
uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8001

# 3. MySQL celeba_db (see config/database.env.example)
mysql -u root -p < database/celeba_db_setup.sql`}
        </pre>
      </GlassCard>
    </div>
  );
}
