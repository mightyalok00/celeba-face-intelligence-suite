import { useEffect, useState } from "react";
import { CheckCircle2, RefreshCw, ShieldCheck, XCircle } from "lucide-react";
import { GlassCard } from "../components/GlassCard";
import type { ApiEndpointDoc, SystemHealthResponse } from "../types";

interface SystemHealthPageProps {
  health: SystemHealthResponse | null;
  onRefresh: () => void;
}

export function SystemHealthPage({ health, onRefresh }: SystemHealthPageProps) {
  const [endpoints, setEndpoints] = useState<ApiEndpointDoc[]>([]);

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

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-slate-50">System Health, Privacy & API Docs</h2>
          <p className="text-sm text-slate-400 mt-1">
            Live service telemetry, security & non-retention safeguards, and documented backend
            endpoints.
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
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">Python FastAPI ML Backend</h3>
            {health?.services.pythonMlBackend.engineReady ? (
              <span className="text-xs text-emerald-400 inline-flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> Online
              </span>
            ) : (
              <span className="text-xs text-amber-400 inline-flex items-center gap-1">
                <XCircle className="w-3.5 h-3.5" /> Offline
              </span>
            )}
          </div>
          <div className="text-xs text-slate-400 space-y-1">
            <div className="font-mono truncate">{health?.services.pythonMlBackend.url}</div>
            <div>
              {health?.services.pythonMlBackend.error ||
                `Verified checkpoints: ${health?.services.pythonMlBackend.verifiedCheckpoints.join(", ")}`}
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

      {/* Reproducible Setup Instructions */}
      <GlassCard className="space-y-3">
        <h3 className="text-base font-semibold text-slate-100">Reproducible Setup Instructions</h3>
        <pre className="p-4 rounded-lg bg-slate-950/80 border border-white/10 text-xs font-mono text-slate-300 overflow-x-auto leading-relaxed">
{`# 1. Start the Node.js + Vite Full-Stack Studio (Port 3000)
npm ci
npm run dev

# 2. (Optional) Start the Python FastAPI PyTorch/XGBoost Inference Microservice (Port 8001)
pip install -r requirements.txt
uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8001

# 3. (Optional) Initialize MySQL celeba_db (see config/database.env.example)
mysql -u root -p < database/celeba_db_setup.sql`}
        </pre>
      </GlassCard>
    </div>
  );
}
