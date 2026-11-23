import express from "express";
import type { Request, Response, NextFunction } from "express";
import crypto from "node:crypto";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import { GoogleGenAI, ThinkingLevel, Type } from "@google/genai";

dotenv.config();

process.on("uncaughtException", (err: Error) => {
  console.error("[Server] Uncaught exception:", err.message);
});

process.on("unhandledRejection", (reason: unknown) => {
  console.error("[Server] Unhandled rejection:", reason);
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = __dirname;

const PORT = Number(process.env.PORT) || 3000;
const HOST = "0.0.0.0";
const DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen";
const FALLBACK_GEMINI_MODEL = "gemini-3.1-flash-lite";
const PYTHON_ML_BACKEND_URL = (
  process.env.PYTHON_ML_BACKEND_URL || "http://127.0.0.1:8001"
).replace(/\/+$/, "");

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB
const ALLOWED_MIME_TYPES = new Set(["image/jpeg", "image/jpg", "image/png", "image/webp"]);

// ---------------------------------------------------------------------------
// Types (erasable-only for direct Node 22 TypeScript execution compatibility)
// ---------------------------------------------------------------------------

interface ParsedBenchmarkRow {
  model: string;
  strategy: string;
  testLoss: number;
  accuracyPct: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number | null;
  prAuc: number | null;
  confusionMatrix: [[number, number], [number, number]] | null;
}

interface GalleryItem {
  id: number;
  filename: string;
  path: string;
  thumbnail: string;
  cluster_id: number;
  ground_truth_id: string;
  person_name: string;
  cluster_label?: string;
  smiling: boolean;
  eyeglasses: boolean;
  male: boolean;
  young: boolean;
  attributes: string[];
  confidence?: {
    smiling: number;
    young: number;
    male: number;
    eyeglasses: number;
  };
}

interface RateBucket {
  timestamps: number[];
}

// ---------------------------------------------------------------------------
// In-Memory Sliding-Window Rate Limiter
// ---------------------------------------------------------------------------

function createRateLimiter(options: {
  windowMs: number;
  maxRequests: number;
  label: string;
}) {
  const store = new Map<string, RateBucket>();

  // Periodic cleanup to avoid unbounded memory growth
  const cleanupTimer = setInterval(() => {
    const cutoff = Date.now() - options.windowMs;
    for (const [key, bucket] of store.entries()) {
      const fresh = bucket.timestamps.filter((t) => t > cutoff);
      if (fresh.length === 0) {
        store.delete(key);
      } else {
        bucket.timestamps = fresh;
      }
    }
  }, Math.max(options.windowMs, 30_000));
  cleanupTimer.unref();

  return (req: Request, res: Response, next: NextFunction) => {
    const forwarded = req.headers["x-forwarded-for"];
    const clientIp = Array.isArray(forwarded)
      ? forwarded[0]
      : typeof forwarded === "string"
        ? forwarded.split(",")[0].trim()
        : req.ip || "unknown";
    const key = `${options.label}:${clientIp}`;
    const now = Date.now();
    const cutoff = now - options.windowMs;

    const bucket = store.get(key) ?? { timestamps: [] };
    bucket.timestamps = bucket.timestamps.filter((t) => t > cutoff);

    if (bucket.timestamps.length >= options.maxRequests) {
      const oldest = bucket.timestamps[0] ?? now;
      const retryAfterSec = Math.max(1, Math.ceil((oldest + options.windowMs - now) / 1000));
      res.setHeader("Retry-After", String(retryAfterSec));
      res.setHeader("X-RateLimit-Limit", String(options.maxRequests));
      res.setHeader("X-RateLimit-Remaining", "0");
      res.status(429).json({
        status: "error",
        error_code: "RATE_LIMIT_EXCEEDED",
        message: `Rate limit exceeded for ${options.label}. Please retry in ${retryAfterSec}s.`,
      });
      return;
    }

    bucket.timestamps.push(now);
    store.set(key, bucket);
    res.setHeader("X-RateLimit-Limit", String(options.maxRequests));
    res.setHeader(
      "X-RateLimit-Remaining",
      String(Math.max(0, options.maxRequests - bucket.timestamps.length))
    );
    next();
  };
}

const apiReadLimiter = createRateLimiter({
  windowMs: 60_000,
  maxRequests: 120,
  label: "api-read",
});

const aiInferenceLimiter = createRateLimiter({
  windowMs: 60_000,
  maxRequests: 25,
  label: "ai-inference",
});

// ---------------------------------------------------------------------------
// Image Validation (Magic-Byte + MIME + Size, Zero Disk Retention)
// ---------------------------------------------------------------------------

function validateImagePayload(payload: {
  image_base64?: unknown;
  mime_type?: unknown;
}): { ok: true; mimeType: string; base64Data: string; buffer: Buffer } | { ok: false; status: number; message: string } {
  if (typeof payload.image_base64 !== "string" || payload.image_base64.trim().length === 0) {
    return { ok: false, status: 400, message: "No image_base64 payload provided." };
  }

  let rawBase64 = payload.image_base64.trim();
  let declaredMime = typeof payload.mime_type === "string" ? payload.mime_type.trim().toLowerCase() : "";

  const dataUrlMatch = /^data:([^;]+);base64,(.+)$/s.exec(rawBase64);
  if (dataUrlMatch) {
    declaredMime = dataUrlMatch[1].trim().toLowerCase();
    rawBase64 = dataUrlMatch[2].replace(/\s+/g, "");
  } else {
    rawBase64 = rawBase64.replace(/\s+/g, "");
  }

  if (declaredMime && !ALLOWED_MIME_TYPES.has(declaredMime)) {
    return {
      ok: false,
      status: 415,
      message: `Unsupported image MIME type '${declaredMime}'. Allowed formats: JPEG, PNG, WebP.`,
    };
  }

  // Approximate decoded byte length check before full allocation
  const estimatedBytes = Math.floor((rawBase64.length * 3) / 4);
  if (estimatedBytes > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      status: 413,
      message: "Uploaded image exceeds the 10 MB maximum size limit.",
    };
  }

  const buffer = Buffer.from(rawBase64, "base64");
  if (buffer.length === 0 || buffer.length > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      status: 400,
      message: buffer.length === 0 ? "Decoded image buffer is empty." : "Image exceeds 10 MB limit.",
    };
  }

  // Magic-byte signature inspection
  const isJpeg = buffer.length >= 3 && buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff;
  const isPng =
    buffer.length >= 8 &&
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47 &&
    buffer[4] === 0x0d &&
    buffer[5] === 0x0a &&
    buffer[6] === 0x1a &&
    buffer[7] === 0x0a;
  const isWebp =
    buffer.length >= 12 &&
    buffer.subarray(0, 4).toString("ascii") === "RIFF" &&
    buffer.subarray(8, 12).toString("ascii") === "WEBP";

  if (!isJpeg && !isPng && !isWebp) {
    return {
      ok: false,
      status: 400,
      message: "Invalid image file signature. Only authentic JPEG, PNG, and WebP images are accepted.",
    };
  }

  const detectedMime = isJpeg ? "image/jpeg" : isPng ? "image/png" : "image/webp";
  return {
    ok: true,
    mimeType: detectedMime,
    base64Data: buffer.toString("base64"),
    buffer,
  };
}

// ---------------------------------------------------------------------------
// Repository File Parsers (100% Real Repo Artifacts)
// ---------------------------------------------------------------------------

function parseCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += ch;
    }
  }
  result.push(current.trim());
  return result;
}

function parseConfusionMatrix(raw: string | undefined): [[number, number], [number, number]] | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (
      Array.isArray(parsed) &&
      parsed.length === 2 &&
      Array.isArray(parsed[0]) &&
      parsed[0].length === 2 &&
      Array.isArray(parsed[1]) &&
      parsed[1].length === 2
    ) {
      return [
        [Number(parsed[0][0]), Number(parsed[0][1])],
        [Number(parsed[1][0]), Number(parsed[1][1])],
      ];
    }
  } catch {
    return null;
  }
  return null;
}

function parseAccuracyPct(raw: unknown): number {
  if (typeof raw === "number") return raw <= 1 ? Number((raw * 100).toFixed(2)) : raw;
  const cleaned = String(raw ?? "0").replace("%", "").trim();
  const num = Number(cleaned);
  return Number.isFinite(num) ? num : 0;
}

function parseBenchmarkCsv(relativePath: string): ParsedBenchmarkRow[] {
  const fullPath = path.join(PROJECT_ROOT, relativePath);
  if (!fs.existsSync(fullPath)) return [];
  const raw = fs.readFileSync(fullPath, "utf-8");
  const lines = raw
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length < 2) return [];

  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const vals = parseCsvLine(line);
    const map: Record<string, string> = {};
    headers.forEach((h, idx) => {
      map[h] = vals[idx] ?? "";
    });
    const roc = map["ROC-AUC"] ? Number(map["ROC-AUC"]) : NaN;
    const pr = map["PR-AUC"] ? Number(map["PR-AUC"]) : NaN;
    return {
      model: map["Model Configuration"] || "Unknown",
      strategy: map["Strategy"] || "",
      testLoss: Number(map["Test Loss"] || 0),
      accuracyPct: parseAccuracyPct(map["Accuracy"]),
      precision: Number(map["Precision"] || 0),
      recall: Number(map["Recall"] || 0),
      f1Score: Number(map["F1-Score"] || 0),
      rocAuc: Number.isFinite(roc) ? roc : null,
      prAuc: Number.isFinite(pr) ? pr : null,
      confusionMatrix: parseConfusionMatrix(map["Confusion Matrix"]),
    };
  });
}

function readJsonSafe<T>(relativePath: string, fallback: T): T {
  const fullPath = path.join(PROJECT_ROOT, relativePath);
  if (!fs.existsSync(fullPath)) return fallback;
  try {
    return JSON.parse(fs.readFileSync(fullPath, "utf-8")) as T;
  } catch {
    return fallback;
  }
}

function loadGalleryData(): GalleryItem[] {
  return readJsonSafe<GalleryItem[]>("frontend/gallery_data.json", []);
}

function getCheckpointInventory() {
  const expected = [
    { architecture: "Spatial BiLSTM", filename: "bilstm.pt" },
    { architecture: "Vision Transformer", filename: "vision_transformer.pt" },
    { architecture: "Residual CNN", filename: "residual_cnn.pt" },
    { architecture: "ANN", filename: "ann.pt" },
  ];
  return expected.map((item) => {
    const fullPath = path.join(PROJECT_ROOT, "models", "checkpoints", item.filename);
    const exists = fs.existsSync(fullPath);
    const sizeBytes = exists ? fs.statSync(fullPath).size : null;
    return {
      architecture: item.architecture,
      filename: item.filename,
      exists,
      sizeBytes,
      sizeMb: sizeBytes !== null ? (sizeBytes / (1024 * 1024)).toFixed(2) : null,
    };
  });
}

let cachedIdentityStats: { totalMappings: number; uniqueIdentities: number } | null = null;
function getIdentityFileStats() {
  if (cachedIdentityStats) return cachedIdentityStats;
  const idPath = path.join(PROJECT_ROOT, "data", "metadata", "identity_CelebA.txt");
  if (!fs.existsSync(idPath)) {
    return { totalMappings: 0, uniqueIdentities: 0 };
  }
  const raw = fs.readFileSync(idPath, "utf-8");
  const lines = raw.split(/\r?\n/);
  let count = 0;
  const ids = new Set<string>();
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    count++;
    const parts = trimmed.split(/\s+/);
    if (parts[1]) ids.add(parts[1]);
  }
  cachedIdentityStats = { totalMappings: count, uniqueIdentities: ids.size };
  return cachedIdentityStats;
}

function getSplitManifestSummary() {
  const splitPath = path.join(PROJECT_ROOT, "reproducibility", "split_manifest.csv");
  if (!fs.existsSync(splitPath)) {
    return { totalRows: 0, splits: [], identityOverlapCount: 0 };
  }
  const lines = fs
    .readFileSync(splitPath, "utf-8")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  const bySplit = new Map<string, { images: number; ids: Set<string> }>();
  for (const line of lines.slice(1)) {
    const [, personId, split] = parseCsvLine(line);
    if (!split) continue;
    const entry = bySplit.get(split) ?? { images: 0, ids: new Set<string>() };
    entry.images += 1;
    if (personId) entry.ids.add(personId);
    bySplit.set(split, entry);
  }

  const trainIds = bySplit.get("train")?.ids ?? new Set<string>();
  const valIds = bySplit.get("val")?.ids ?? new Set<string>();
  const testIds = bySplit.get("test")?.ids ?? new Set<string>();
  let overlap = 0;
  for (const id of testIds) {
    if (trainIds.has(id) || valIds.has(id)) overlap++;
  }

  return {
    totalRows: Math.max(0, lines.length - 1),
    splits: Array.from(bySplit.entries()).map(([split, data]) => ({
      split,
      images: data.images,
      uniqueIdentities: data.ids.size,
    })),
    identityOverlapCount: overlap,
  };
}

function getCelebritiesCount(): number {
  const celebPath = path.join(PROJECT_ROOT, "data", "metadata", "celebrities.csv");
  if (!fs.existsSync(celebPath)) return 0;
  const lines = fs
    .readFileSync(celebPath, "utf-8")
    .split(/\r?\n/)
    .filter((l) => l.trim().length > 0);
  return Math.max(0, lines.length - 1);
}

// ---------------------------------------------------------------------------
// Upstream Probes (Python FastAPI ML Backend & MySQL TCP Check)
// ---------------------------------------------------------------------------

async function probePythonMlBackend(): Promise<{
  url: string;
  reachable: boolean;
  engineReady: boolean;
  verifiedCheckpoints: string[];
  latencyMs: number | null;
  error: string | null;
}> {
  const t0 = performance.now();
  try {
    const resp = await fetch(`${PYTHON_ML_BACKEND_URL}/health`, {
      signal: AbortSignal.timeout(1200),
    });
    const latencyMs = Number((performance.now() - t0).toFixed(1));
    if (!resp.ok) {
      return {
        url: PYTHON_ML_BACKEND_URL,
        reachable: true,
        engineReady: false,
        verifiedCheckpoints: [],
        latencyMs,
        error: `FastAPI returned HTTP ${resp.status}`,
      };
    }
    const data = (await resp.json()) as {
      service?: string;
      engine_ready?: boolean;
      verified_checkpoints?: string[];
      error?: string | null;
    };
    if (data.service !== "celeba-fastapi-ml") {
      return {
        url: PYTHON_ML_BACKEND_URL,
        reachable: false,
        engineReady: false,
        verifiedCheckpoints: [],
        latencyMs,
        error: `Endpoint at ${PYTHON_ML_BACKEND_URL} is not the CelebA FastAPI ML service (service="${data.service ?? "unknown"}").`,
      };
    }
    return {
      url: PYTHON_ML_BACKEND_URL,
      reachable: true,
      engineReady: Boolean(data.engine_ready),
      verifiedCheckpoints: Array.isArray(data.verified_checkpoints) ? data.verified_checkpoints : [],
      latencyMs,
      error: data.error ?? null,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : "Connection failed";
    return {
      url: PYTHON_ML_BACKEND_URL,
      reachable: false,
      engineReady: false,
      verifiedCheckpoints: [],
      latencyMs: null,
      error: `Python FastAPI ML service offline at ${PYTHON_ML_BACKEND_URL} (${message})`,
    };
  }
}

async function probeMysqlStatus(): Promise<{
  configured: boolean;
  host: string | null;
  port: number;
  database: string | null;
  reachable: boolean;
  error: string | null;
}> {
  const host = process.env.MYSQL_HOST || null;
  const user = process.env.MYSQL_USER || null;
  const password = process.env.MYSQL_PASSWORD || null;
  const database = process.env.MYSQL_DATABASE || null;
  const port = Number(process.env.MYSQL_PORT) || 3306;

  if (!host || !user || !password || !database) {
    return {
      configured: false,
      host,
      port,
      database,
      reachable: false,
      error: "MySQL environment variables (MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE) are not configured.",
    };
  }

  return new Promise((resolve) => {
    const socket = net.createConnection({ host, port });
    const timer = setTimeout(() => {
      socket.destroy();
      resolve({
        configured: true,
        host,
        port,
        database,
        reachable: false,
        error: `TCP timeout connecting to MySQL at ${host}:${port}`,
      });
    }, 800);

    socket.on("connect", () => {
      clearTimeout(timer);
      socket.end();
      resolve({
        configured: true,
        host,
        port,
        database,
        reachable: true,
        error: null,
      });
    });

    socket.on("error", (err) => {
      clearTimeout(timer);
      resolve({
        configured: true,
        host,
        port,
        database,
        reachable: false,
        error: err.message,
      });
    });
  });
}

// ---------------------------------------------------------------------------
// Gemini Client (Server-Side Only, Privacy-Preserving)
// ---------------------------------------------------------------------------

function createGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return null;
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function isTransientGeminiError(err: unknown): boolean {
  if (err instanceof Error && (err.name === "TimeoutError" || err.name === "AbortError")) {
    return true;
  }
  const msg = err instanceof Error ? err.message : String(err);
  return (
    msg.includes("503") ||
    msg.includes("504") ||
    msg.includes("UNAVAILABLE") ||
    msg.includes("DEADLINE_EXCEEDED") ||
    msg.includes("429") ||
    msg.includes("RESOURCE_EXHAUSTED") ||
    msg.toLowerCase().includes("high demand") ||
    msg.toLowerCase().includes("timed out") ||
    msg.toLowerCase().includes("timeout") ||
    msg.toLowerCase().includes("aborted") ||
    msg.toLowerCase().includes("fetch failed")
  );
}

async function callGeminiWithFallback<T>(
  runner: (model: string, abortSignal: AbortSignal) => Promise<T>
): Promise<{ result: T; modelUsed: string }> {
  const attempts = [
    { model: DEFAULT_GEMINI_MODEL, timeoutMs: 20_000, backoffMs: 500 },
    { model: FALLBACK_GEMINI_MODEL, timeoutMs: 30_000, backoffMs: 900 },
    { model: FALLBACK_GEMINI_MODEL, timeoutMs: 35_000, backoffMs: 1400 },
    { model: DEFAULT_GEMINI_MODEL, timeoutMs: 35_000, backoffMs: 0 },
  ];
  let lastErr: unknown;

  for (let i = 0; i < attempts.length; i++) {
    const { model, timeoutMs, backoffMs } = attempts[i];
    try {
      const result = await runner(model, AbortSignal.timeout(timeoutMs));
      return { result, modelUsed: model };
    } catch (err) {
      lastErr = err;
      if (i === attempts.length - 1 || !isTransientGeminiError(err)) {
        throw err;
      }
      await new Promise((resolve) => setTimeout(resolve, backoffMs));
    }
  }
  throw lastErr;
}

const PRIVACY_ANALYZER_SYSTEM_INSTRUCTION = `
You are a Computer Vision and Facial Expression Quality Assistant for CelebA AI Studio.
Analyze ONLY visible, non-sensitive facial expression, visual accessories, hair color, and photographic quality cues (lighting, framing, occlusion).
STRICT PRIVACY RULES:
1. Do NOT attempt to identify any specific person or celebrity by name.
2. Do NOT infer or speculate on sensitive personal characteristics (race, ethnicity, nationality, religion, health conditions, socioeconomic status, or sexual orientation).
3. Return ONLY valid JSON matching the requested schema.
`.trim();

function buildChatSystemInstruction(): string {
  const prodRows = parseBenchmarkCsv("reports/benchmarks/production_benchmark.csv").map((r) => ({
    model: r.model,
    acc: r.accuracyPct,
    f1: r.f1Score,
    rocAuc: r.rocAuc,
  }));
  const smokeRows = parseBenchmarkCsv("reports/benchmarks/ci_smoke_test_benchmark.csv").map((r) => ({
    model: r.model,
    acc: r.accuracyPct,
    f1: r.f1Score,
    rocAuc: r.rocAuc,
  }));
  const latestRun = readJsonSafe<{ optimal_ensemble_weights?: Record<string, number> }>(
    "reports/benchmarks/latest_training_run.json",
    {}
  );
  const manifest = readJsonSafe<Record<string, unknown>>("reproducibility/dataset_manifest.json", {});

  return `
You are the CelebA AI — Face Intelligence Studio Assistant.
Help engineers and researchers understand the architectures, evaluation protocols, dataset analytics, fairness audit, and APIs in this repository.
Always cite exact numbers from the repository ground truth below and clearly distinguish real-image holdout benchmarks from synthetic CI smoke-test benchmarks. Never invent metrics or predictions.

Repository Ground Truth:
- Dataset: CelebA (${manifest.total_images ?? 202599} images, ${manifest.total_unique_identities ?? 10177} identities, 40 attributes, canonical 64x64x3, GroupShuffleSplit + 5-fold StratifiedGroupKFold with 0% identity leakage).
- Real-Image Production Holdout (75 samples, Target: Smiling, reports/benchmarks/production_benchmark.csv): ${JSON.stringify(prodRows)}
- Synthetic CI Smoke-Test (30 synthetic samples, reports/benchmarks/ci_smoke_test_benchmark.csv): ${JSON.stringify(smokeRows)}
- SLSQP Optimal Ensemble Weights (latest_training_run.json): ${JSON.stringify(latestRun.optimal_ensemble_weights ?? {})}
- Fairness Audit (docs/AUDIT_AND_EVALUATION.md): Male/Female Disparate Impact Ratio 0.9275 (EEOC 4/5ths compliant >= 0.80); Young/Senior Disparate Impact Ratio 0.8832 (EEOC 4/5ths compliant).
- Privacy & Security: Zero disk retention, magic-byte upload validation, explicit camera consent, rate limiting, no individual identification or sensitive characteristic inference.
`.trim();
}

// ---------------------------------------------------------------------------
// Express App Setup
// ---------------------------------------------------------------------------

const app = express();
app.disable("x-powered-by");
app.use(express.json({ limit: "15mb" }));

// Request Correlation ID & Structured API Audit Logging (No Sensitive Payloads)
app.use((req: Request, res: Response, next: NextFunction) => {
  const incomingId = req.headers["x-request-id"];
  const candidate = Array.isArray(incomingId) ? incomingId[0] : incomingId;
  const requestId =
    typeof candidate === "string" && /^[A-Za-z0-9_-]{8,64}$/.test(candidate.trim())
      ? candidate.trim()
      : crypto.randomUUID();

  res.setHeader("X-Request-Id", requestId);

  if (req.path.startsWith("/api/")) {
    const startMs = performance.now();
    res.on("finish", () => {
      const durationMs = Number((performance.now() - startMs).toFixed(1));
      console.log(
        JSON.stringify({
          ts: new Date().toISOString(),
          requestId,
          method: req.method,
          path: req.path,
          status: res.statusCode,
          durationMs,
        })
      );
    });
  }
  next();
});

// CORS & Security Headers
app.use((req: Request, res: Response, next: NextFunction) => {
  const isProd = process.env.NODE_ENV === "production";
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.setHeader("X-Permitted-Cross-Domain-Policies", "none");
  res.setHeader("Permissions-Policy", "camera=(self), microphone=(), geolocation=()");
  res.setHeader(
    "Content-Security-Policy",
    [
      "default-src 'self'",
      isProd
        ? "script-src 'self' 'unsafe-inline'"
        : "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
      "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
      "font-src 'self' https://fonts.gstatic.com data:",
      "img-src 'self' data: blob:",
      "media-src 'self' blob:",
      isProd ? "connect-src 'self'" : "connect-src 'self' ws: wss:",
      "object-src 'none'",
      "base-uri 'self'",
      "form-action 'self'",
    ].join("; ")
  );
  if (req.path.startsWith("/api/")) {
    res.setHeader("Cache-Control", "no-store, max-age=0");
  }
  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }
  next();
});

// Health & Readiness Probes
app.head(["/health", "/healthz", "/_ah/health", "/api/health", "/ping"], (_req: Request, res: Response) => {
  res.status(200).end();
});

app.get(["/health", "/healthz", "/_ah/health", "/api/health", "/ping"], (_req: Request, res: Response) => {
  res.json({
    status: "healthy",
    service: "CelebA AI — Face Intelligence Studio",
    provider: "Gemini API + Express Gateway",
    ready: true,
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
  });
});

// Documented API Endpoints Catalog
const API_DOCS = [
  {
    method: "GET",
    path: "/api/benchmarks",
    summary: "Returns real-image holdout benchmarks, latest training run, 5-fold CV results, and synthetic CI smoke-test metrics.",
    authOrPolicy: "Read rate limit (120 req/min)",
  },
  {
    method: "GET",
    path: "/api/benchmarks/export",
    summary: "Downloads allowlisted benchmark suites (production, latest, synthetic) as CSV or JSON attachments.",
    authOrPolicy: "Read rate limit (120 req/min), strict suite/format allowlist",
  },
  {
    method: "GET",
    path: "/api/models",
    summary: "Returns model registry, checkpoint file inventory, model cards, YAML hyperparameters, and live FastAPI status.",
    authOrPolicy: "Read rate limit (120 req/min)",
  },
  {
    method: "GET",
    path: "/api/dataset/analytics",
    summary: "Returns CelebA dataset manifest, identity-disjoint split counts, gallery attribute distributions, and figure URLs.",
    authOrPolicy: "Read rate limit (120 req/min)",
  },
  {
    method: "GET",
    path: "/api/gallery",
    summary: "Returns curated CelebA reference sample thumbnails and attributes for non-sensitive testing.",
    authOrPolicy: "Read rate limit (120 req/min)",
  },
  {
    method: "POST",
    path: "/api/ml/predict",
    summary: "Forwards validated facial image to Python FastAPI PyTorch/XGBoost inference backend. Returns 503 if backend is offline (no fake predictions).",
    authOrPolicy: "AI inference rate limit (25 req/min), magic-byte validation, zero disk retention",
  },
  {
    method: "POST",
    path: "/api/ml/consensus",
    summary: "Executes multi-architecture consensus across verified PyTorch checkpoints via FastAPI backend. Returns 503 if backend is offline.",
    authOrPolicy: "AI inference rate limit (25 req/min), magic-byte validation, zero disk retention",
  },
  {
    method: "POST",
    path: "/api/gemini/analyze",
    summary: "Executes server-side Gemini multimodal visual attribute & quality analysis without individual identification or sensitive profiling.",
    authOrPolicy: "AI inference rate limit (25 req/min), server-side GEMINI_API_KEY, zero disk retention",
  },
  {
    method: "POST",
    path: "/api/gemini/chat",
    summary: "Multi-turn Gemini AI assistant grounded with repository benchmarks, architectures, and fairness reports.",
    authOrPolicy: "AI inference rate limit (25 req/min), server-side GEMINI_API_KEY",
  },
  {
    method: "POST",
    path: "/api/gemini/chat/stream",
    summary: "Streaming Server-Sent Events (SSE) endpoint for real-time token responses from grounded Gemini assistant.",
    authOrPolicy: "AI inference rate limit (25 req/min), server-side GEMINI_API_KEY",
  },
  {
    method: "GET",
    path: "/api/system/health",
    summary: "Inspects live connectivity for Gemini API, Python FastAPI ML backend, MySQL celeba_db, checkpoints, and privacy controls.",
    authOrPolicy: "Read rate limit (120 req/min)",
  },
  {
    method: "GET",
    path: "/api/docs",
    summary: "Returns structured documentation of all CelebA AI Studio backend API routes.",
    authOrPolicy: "Public",
  },
];

app.get("/api/docs", (_req: Request, res: Response) => {
  res.json({
    service: "CelebA AI — Face Intelligence Studio API",
    version: "2.5.0",
    endpoints: API_DOCS,
  });
});

app.get("/api/benchmarks", apiReadLimiter, (_req: Request, res: Response) => {
  const productionRows = parseBenchmarkCsv("reports/benchmarks/production_benchmark.csv");
  const smokeRows = parseBenchmarkCsv("reports/benchmarks/ci_smoke_test_benchmark.csv");
  const latestRun = readJsonSafe<{
    timestamp?: string;
    elapsed_seconds?: number;
    train_samples?: number;
    test_samples?: number;
    optimal_ensemble_weights?: Record<string, number>;
    benchmarks?: Array<Record<string, unknown>>;
  }>("reports/benchmarks/latest_training_run.json", {});

  const latestRows: ParsedBenchmarkRow[] = (latestRun.benchmarks ?? []).map((b) => ({
    model: String(b["Model Configuration"] ?? "Unknown"),
    strategy: String(b["Strategy"] ?? ""),
    testLoss: Number(b["Test Loss"] ?? 0),
    accuracyPct: parseAccuracyPct(b["Accuracy"]),
    precision: Number(b["Precision"] ?? 0),
    recall: Number(b["Recall"] ?? 0),
    f1Score: Number(b["F1-Score"] ?? 0),
    rocAuc: b["ROC-AUC"] !== undefined ? Number(b["ROC-AUC"]) : null,
    prAuc: b["PR-AUC"] !== undefined ? Number(b["PR-AUC"]) : null,
    confusionMatrix: parseConfusionMatrix(
      b["Confusion Matrix"] !== undefined ? String(b["Confusion Matrix"]) : undefined
    ),
  }));

  const cvPath = path.join(PROJECT_ROOT, "reports", "benchmarks", "cross_validation_results.csv");
  const folds: { fold: number; valAccuracy: number }[] = [];
  if (fs.existsSync(cvPath)) {
    const cvLines = fs
      .readFileSync(cvPath, "utf-8")
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);
    for (const line of cvLines.slice(1)) {
      const [foldStr, accStr] = parseCsvLine(line);
      const fold = Number(foldStr);
      const valAccuracy = Number(accStr);
      if (Number.isFinite(fold) && Number.isFinite(valAccuracy)) {
        folds.push({ fold, valAccuracy });
      }
    }
  }
  const meanAccuracy =
    folds.length > 0
      ? Number((folds.reduce((sum, f) => sum + f.valAccuracy, 0) / folds.length).toFixed(4))
      : 0;

  res.json({
    realImageProduction: {
      sourceFile: "reports/benchmarks/production_benchmark.csv",
      datasetType: "real_image_holdout",
      sampleCount: 75,
      targetAttribute: "Smiling",
      rows: productionRows,
    },
    latestTrainingRun: {
      sourceFile: "reports/benchmarks/latest_training_run.json",
      datasetType: "real_image_training_run",
      timestamp: latestRun.timestamp ?? "",
      elapsedSeconds: latestRun.elapsed_seconds ?? 0,
      trainSamples: latestRun.train_samples ?? 0,
      testSamples: latestRun.test_samples ?? 0,
      optimalEnsembleWeights: latestRun.optimal_ensemble_weights ?? {},
      rows: latestRows,
    },
    ciSmokeTestSynthetic: {
      sourceFile: "reports/benchmarks/ci_smoke_test_benchmark.csv",
      datasetType: "synthetic_smoke_test",
      sampleCount: 30,
      note: "Synthetic CI pipeline validation dataset (30 generated samples). Kept strictly separate from real-image CelebA holdout benchmarks.",
      rows: smokeRows,
    },
    crossValidation: {
      sourceFile: "reports/benchmarks/cross_validation_results.csv",
      folds,
      meanAccuracy,
    },
  });
});

function escapeCsvField(val: unknown): string {
  const s = val === null || val === undefined ? "" : String(val);
  if (/[",\r\n]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

const EXPORT_SUITE_ALLOWLIST = new Set(["production", "latest", "synthetic"]);
const EXPORT_FORMAT_ALLOWLIST = new Set(["csv", "json"]);

app.get("/api/benchmarks/export", apiReadLimiter, (req: Request, res: Response) => {
  const rawSuite = typeof req.query.suite === "string" ? req.query.suite.trim().toLowerCase() : "production";
  const rawFormat = typeof req.query.format === "string" ? req.query.format.trim().toLowerCase() : "csv";

  if (!EXPORT_SUITE_ALLOWLIST.has(rawSuite)) {
    res.status(400).json({
      status: "error",
      error_code: "INVALID_EXPORT_SUITE",
      message: "Invalid suite parameter. Allowed values: production, latest, synthetic.",
    });
    return;
  }

  if (!EXPORT_FORMAT_ALLOWLIST.has(rawFormat)) {
    res.status(400).json({
      status: "error",
      error_code: "INVALID_EXPORT_FORMAT",
      message: "Invalid format parameter. Allowed values: csv, json.",
    });
    return;
  }

  let sourceFile = "reports/benchmarks/production_benchmark.csv";
  let datasetType = "real_image_holdout";
  let rows: ParsedBenchmarkRow[] = [];

  if (rawSuite === "production") {
    rows = parseBenchmarkCsv("reports/benchmarks/production_benchmark.csv");
  } else if (rawSuite === "synthetic") {
    sourceFile = "reports/benchmarks/ci_smoke_test_benchmark.csv";
    datasetType = "synthetic_smoke_test";
    rows = parseBenchmarkCsv("reports/benchmarks/ci_smoke_test_benchmark.csv");
  } else {
    sourceFile = "reports/benchmarks/latest_training_run.json";
    datasetType = "real_image_training_run";
    const latestRun = readJsonSafe<{ benchmarks?: Array<Record<string, unknown>> }>(
      "reports/benchmarks/latest_training_run.json",
      {}
    );
    rows = (latestRun.benchmarks ?? []).map((b) => ({
      model: String(b["Model Configuration"] ?? "Unknown"),
      strategy: String(b["Strategy"] ?? ""),
      testLoss: Number(b["Test Loss"] ?? 0),
      accuracyPct: parseAccuracyPct(b["Accuracy"]),
      precision: Number(b["Precision"] ?? 0),
      recall: Number(b["Recall"] ?? 0),
      f1Score: Number(b["F1-Score"] ?? 0),
      rocAuc: b["ROC-AUC"] !== undefined ? Number(b["ROC-AUC"]) : null,
      prAuc: b["PR-AUC"] !== undefined ? Number(b["PR-AUC"]) : null,
      confusionMatrix: parseConfusionMatrix(
        b["Confusion Matrix"] !== undefined ? String(b["Confusion Matrix"]) : undefined
      ),
    }));
  }

  const downloadFilename = `celeba_${rawSuite}_benchmarks.${rawFormat}`;
  res.setHeader("Content-Disposition", `attachment; filename="${downloadFilename}"`);

  if (rawFormat === "json") {
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.send(
      JSON.stringify(
        {
          suite: rawSuite,
          datasetType,
          sourceFile,
          exportedAt: new Date().toISOString(),
          rowCount: rows.length,
          rows,
        },
        null,
        2
      )
    );
    return;
  }

  const csvHeader = [
    "Model Configuration",
    "Strategy",
    "Test Loss",
    "Accuracy (%)",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC",
    "PR-AUC",
    "Confusion Matrix",
  ].join(",");

  const csvLines = rows.map((r) =>
    [
      escapeCsvField(r.model),
      escapeCsvField(r.strategy),
      escapeCsvField(r.testLoss.toFixed(4)),
      escapeCsvField(r.accuracyPct.toFixed(2)),
      escapeCsvField(r.precision.toFixed(4)),
      escapeCsvField(r.recall.toFixed(4)),
      escapeCsvField(r.f1Score.toFixed(4)),
      escapeCsvField(r.rocAuc !== null ? r.rocAuc.toFixed(4) : ""),
      escapeCsvField(r.prAuc !== null ? r.prAuc.toFixed(4) : ""),
      escapeCsvField(r.confusionMatrix ? JSON.stringify(r.confusionMatrix) : ""),
    ].join(",")
  );

  res.setHeader("Content-Type", "text/csv; charset=utf-8");
  res.send([csvHeader, ...csvLines].join("\n"));
});

app.get("/api/models", apiReadLimiter, async (_req: Request, res: Response) => {
  const registry = readJsonSafe("models/model_registry.json", {
    project: "CelebA Face AI & Intelligence System",
    version: "2.5.0",
    target_attribute: "Smiling",
    identity_isolation_strategy: "StratifiedGroupKFold",
    evaluated_models: [],
  });
  const checkpoints = getCheckpointInventory();

  const cardSpecs = [
    { id: "ensemble", title: "SLSQP Multi-Architecture Ensemble + TTA", filename: "ensemble_card.md" },
    { id: "residual_cnn", title: "4-Stage Residual Conv2D CNN", filename: "residual_cnn_card.md" },
    { id: "vision_transformer", title: "Vision Transformer (ViT)", filename: "vision_transformer_card.md" },
  ];
  const modelCards = cardSpecs
    .map((spec) => {
      const fullPath = path.join(PROJECT_ROOT, "reports", "model_cards", spec.filename);
      if (!fs.existsSync(fullPath)) return null;
      return {
        ...spec,
        content: fs.readFileSync(fullPath, "utf-8"),
      };
    })
    .filter((x): x is NonNullable<typeof x> => x !== null);

  const cfgPath = path.join(PROJECT_ROOT, "reproducibility", "config.yaml");
  const reproducibilityConfigYaml = fs.existsSync(cfgPath)
    ? fs.readFileSync(cfgPath, "utf-8")
    : "";

  const mlProbe = await probePythonMlBackend();

  res.json({
    registry,
    checkpoints,
    modelCards,
    reproducibilityConfigYaml,
    mlBackend: {
      url: mlProbe.url,
      online: mlProbe.reachable && mlProbe.engineReady,
      verifiedCheckpoints: mlProbe.verifiedCheckpoints,
      error: mlProbe.error,
    },
  });
});

app.get("/api/dataset/analytics", apiReadLimiter, (_req: Request, res: Response) => {
  const manifest = readJsonSafe("reproducibility/dataset_manifest.json", {});
  const experimentMetadata = readJsonSafe("reproducibility/experiment_metadata.json", {});
  const splitManifestSummary = getSplitManifestSummary();
  const identityStats = getIdentityFileStats();
  const celebritiesCatalogCount = getCelebritiesCount();
  const gallery = loadGalleryData();

  const attrMap = new Map<string, number>();
  const clusterSet = new Set<number>();
  let smilingCount = 0;
  let youngCount = 0;
  let maleCount = 0;
  let eyeglassesCount = 0;

  for (const item of gallery) {
    clusterSet.add(item.cluster_id);
    if (item.smiling) smilingCount++;
    if (item.young) youngCount++;
    if (item.male) maleCount++;
    if (item.eyeglasses) eyeglassesCount++;
    for (const attr of item.attributes ?? []) {
      attrMap.set(attr, (attrMap.get(attr) ?? 0) + 1);
    }
  }

  const totalSamples = gallery.length;
  const attributeCounts = Array.from(attrMap.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 14)
    .map(([attribute, count]) => ({
      attribute: attribute.replace(/_/g, " "),
      count,
      percentage: totalSamples > 0 ? Number(((count / totalSamples) * 100).toFixed(1)) : 0,
    }));

  const figuresDir = path.join(PROJECT_ROOT, "reports", "figures");
  const figures: Array<{
    filename: string;
    title: string;
    category: "dataset" | "performance" | "confusion_matrix";
    url: string;
  }> = [];

  if (fs.existsSync(figuresDir)) {
    const files = fs.readdirSync(figuresDir).filter((f) => f.endsWith(".png")).sort();
    for (const filename of files) {
      const category = filename.startsWith("cm_")
        ? "confusion_matrix"
        : filename.includes("attribute") || filename.includes("cluster") || filename.includes("cosine")
          ? "dataset"
          : "performance";
      const title = filename
        .replace(/\.png$/i, "")
        .replace(/^cm_/, "Confusion Matrix: ")
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase());
      figures.push({
        filename,
        title,
        category,
        url: `/reports/figures/${encodeURIComponent(filename)}`,
      });
    }
  }

  res.json({
    manifest,
    experimentMetadata,
    splitManifestSummary,
    identityFileStats: {
      ...identityStats,
      celebritiesCatalogCount,
    },
    galleryStats: {
      totalSamples,
      uniqueClusters: clusterSet.size,
      attributeCounts,
      summaryRates: {
        smiling: totalSamples > 0 ? Number(((smilingCount / totalSamples) * 100).toFixed(1)) : 0,
        young: totalSamples > 0 ? Number(((youngCount / totalSamples) * 100).toFixed(1)) : 0,
        male: totalSamples > 0 ? Number(((maleCount / totalSamples) * 100).toFixed(1)) : 0,
        eyeglasses: totalSamples > 0 ? Number(((eyeglassesCount / totalSamples) * 100).toFixed(1)) : 0,
      },
    },
    figures,
  });
});

app.get("/api/gallery", apiReadLimiter, (_req: Request, res: Response) => {
  // Strip ground-truth person names to avoid emphasizing individual identification in the studio UI
  const sanitized = loadGalleryData().map((item) => ({
    id: item.id,
    filename: item.filename,
    thumbnail: item.thumbnail,
    cluster_id: item.cluster_id,
    smiling: item.smiling,
    eyeglasses: item.eyeglasses,
    male: item.male,
    young: item.young,
    attributes: item.attributes,
  }));
  res.json(sanitized);
});

app.get("/api/system/health", apiReadLimiter, async (_req: Request, res: Response) => {
  const [mlProbe, mysqlProbe] = await Promise.all([probePythonMlBackend(), probeMysqlStatus()]);
  const checkpoints = getCheckpointInventory();
  const geminiConfigured = Boolean(process.env.GEMINI_API_KEY);

  res.json({
    status: geminiConfigured || mlProbe.engineReady ? "healthy" : "degraded",
    service: "CelebA AI — Face Intelligence Studio",
    nodeVersion: process.version,
    environment: process.env.NODE_ENV || "development",
    port: PORT,
    uptimeSeconds: Number(process.uptime().toFixed(1)),
    timestamp: new Date().toISOString(),
    services: {
      geminiApi: {
        configured: geminiConfigured,
        defaultModel: DEFAULT_GEMINI_MODEL,
        keyStorage: "server_env_only",
      },
      pythonMlBackend: mlProbe,
      mysqlDatabase: mysqlProbe,
    },
    checkpoints,
    privacyAndSecurity: {
      imageRetention: "Zero retention (in-memory single-request processing only)",
      identityInferenceBlocked: true,
      sensitiveAttributeInferenceBlocked: true,
      magicByteValidation: true,
      maxUploadMb: 10,
      rateLimiting: {
        readWindowLimit: "120 requests / 60s per IP",
        inferenceWindowLimit: "25 requests / 60s per IP",
      },
    },
  });
});

// Real ML Inference Bridge -> Python FastAPI (PyTorch / XGBoost)
// Strictly returns 503 if FastAPI ML backend is unavailable; never fabricates predictions.
app.post("/api/ml/predict", aiInferenceLimiter, async (req: Request, res: Response) => {
  const body = (req.body ?? {}) as {
    image_base64?: unknown;
    mime_type?: unknown;
    architecture?: unknown;
    use_tta?: unknown;
  };

  const validated = validateImagePayload(body);
  if (!validated.ok) {
    res.status(validated.status).json({
      status: "error",
      error_code: "INVALID_IMAGE_UPLOAD",
      message: validated.message,
    });
    return;
  }

  const architecture =
    typeof body.architecture === "string" && body.architecture.trim()
      ? body.architecture.trim()
      : "Vision Transformer";
  const useTta = body.use_tta !== false;

  try {
    const form = new FormData();
    const uint8 = new Uint8Array(validated.buffer);
    const blob = new Blob([uint8], { type: validated.mimeType });
    form.append("file", blob, "upload.jpg");
    form.append("architecture", architecture);
    form.append("use_tta", String(useTta));

    const upstream = await fetch(`${PYTHON_ML_BACKEND_URL}/api/predict`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(15_000),
    });

    const data = (await upstream.json().catch(() => ({}))) as Record<string, unknown>;
    if (!upstream.ok) {
      res.status(upstream.status).json({
        status: "error",
        error_code: "ML_INFERENCE_ERROR",
        message:
          (typeof data.detail === "string" && data.detail) ||
          (typeof data.message === "string" && data.message) ||
          `Python FastAPI inference backend returned HTTP ${upstream.status}.`,
        backend_url: PYTHON_ML_BACKEND_URL,
      });
      return;
    }

    res.json({
      status: "success",
      ...data,
      retained_on_disk: false,
    });
  } catch (err) {
    const reason = err instanceof Error ? err.message : "Connection refused";
    res.status(503).json({
      status: "error",
      error_code: "ML_BACKEND_UNAVAILABLE",
      message:
        `PyTorch/XGBoost FastAPI inference backend is unavailable at ${PYTHON_ML_BACKEND_URL} (${reason}). ` +
        `Start it with "uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8001" in a Python environment with requirements.txt installed. ` +
        `No synthetic or fallback predictions were generated.`,
      backend_url: PYTHON_ML_BACKEND_URL,
    });
  }
});

// Multi-Architecture Consensus Inference Bridge -> Python FastAPI
// Strictly returns 503 if FastAPI ML backend is unavailable; never fabricates predictions.
app.post("/api/ml/consensus", aiInferenceLimiter, async (req: Request, res: Response) => {
  const body = (req.body ?? {}) as {
    image_base64?: unknown;
    mime_type?: unknown;
    use_tta?: unknown;
  };

  const validated = validateImagePayload(body);
  if (!validated.ok) {
    res.status(validated.status).json({
      status: "error",
      error_code: "INVALID_IMAGE_UPLOAD",
      message: validated.message,
    });
    return;
  }

  const useTta = body.use_tta !== false;

  try {
    const form = new FormData();
    const uint8 = new Uint8Array(validated.buffer);
    const blob = new Blob([uint8], { type: validated.mimeType });
    form.append("file", blob, "upload.jpg");
    form.append("use_tta", String(useTta));

    const upstream = await fetch(`${PYTHON_ML_BACKEND_URL}/api/predict/consensus`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(25_000),
    });

    const data = (await upstream.json().catch(() => ({}))) as Record<string, unknown>;
    if (!upstream.ok) {
      res.status(upstream.status).json({
        status: "error",
        error_code: "ML_INFERENCE_ERROR",
        message:
          (typeof data.detail === "string" && data.detail) ||
          (typeof data.message === "string" && data.message) ||
          `Python FastAPI consensus endpoint returned HTTP ${upstream.status}.`,
        backend_url: PYTHON_ML_BACKEND_URL,
      });
      return;
    }

    res.json({
      status: "success",
      ...data,
      retained_on_disk: false,
    });
  } catch (err) {
    const reason = err instanceof Error ? err.message : "Connection refused";
    res.status(503).json({
      status: "error",
      error_code: "ML_BACKEND_UNAVAILABLE",
      message:
        `PyTorch/XGBoost FastAPI inference backend is unavailable at ${PYTHON_ML_BACKEND_URL} (${reason}). ` +
        `Start it with "uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8001" in a Python environment with requirements.txt installed. ` +
        `No synthetic consensus predictions were generated.`,
      backend_url: PYTHON_ML_BACKEND_URL,
    });
  }
});

// Server-Side Gemini Multimodal Face Visual Analysis (Privacy-Hardened)
app.post("/api/gemini/analyze", aiInferenceLimiter, async (req: Request, res: Response) => {
  const ai = createGeminiClient();
  if (!ai) {
    res.status(503).json({
      status: "error",
      error_code: "GEMINI_NOT_CONFIGURED",
      message: "GEMINI_API_KEY is not configured in the server environment.",
    });
    return;
  }

  const body = (req.body ?? {}) as {
    image_base64?: unknown;
    mime_type?: unknown;
    context_note?: unknown;
  };

  const validated = validateImagePayload(body);
  if (!validated.ok) {
    res.status(validated.status).json({
      status: "error",
      error_code: "INVALID_IMAGE_UPLOAD",
      message: validated.message,
    });
    return;
  }

  const t0 = performance.now();
  try {
    let promptText =
      "Analyze visible facial expression, hair color, eyeglasses/facial hair presence, and photographic quality (lighting, framing, occlusion) without identifying any individual or inferring sensitive personal characteristics.";
    if (typeof body.context_note === "string" && body.context_note.trim()) {
      promptText += ` Additional context: ${body.context_note.trim().slice(0, 300)}`;
    }

    const { result: response, modelUsed } = await callGeminiWithFallback((model, abortSignal) =>
      ai.models.generateContent({
        model,
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: validated.mimeType,
                data: validated.base64Data,
              },
            },
            { text: promptText },
          ],
        },
        config: {
          abortSignal,
          systemInstruction: PRIVACY_ANALYZER_SYSTEM_INSTRUCTION,
          thinkingConfig: {
            thinkingLevel:
              model === FALLBACK_GEMINI_MODEL ? ThinkingLevel.MINIMAL : ThinkingLevel.LOW,
          },
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              expression: {
                type: Type.OBJECT,
                properties: {
                  is_smiling: { type: Type.BOOLEAN },
                  confidence: { type: Type.NUMBER },
                  primary_emotion: { type: Type.STRING },
                },
                required: ["is_smiling", "confidence", "primary_emotion"],
              },
              visual_attributes: {
                type: Type.OBJECT,
                properties: {
                  hair_color: { type: Type.STRING },
                  wearing_eyeglasses: { type: Type.BOOLEAN },
                  facial_hair: { type: Type.BOOLEAN },
                  headwear_or_accessories: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  notable_visual_cues: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                },
                required: [
                  "hair_color",
                  "wearing_eyeglasses",
                  "facial_hair",
                  "headwear_or_accessories",
                  "notable_visual_cues",
                ],
              },
              image_quality: {
                type: Type.OBJECT,
                properties: {
                  lighting: { type: Type.STRING },
                  framing: { type: Type.STRING },
                  occlusion_level: { type: Type.STRING },
                },
                required: ["lighting", "framing", "occlusion_level"],
              },
              reasoning_summary: { type: Type.STRING },
            },
            required: ["expression", "visual_attributes", "image_quality", "reasoning_summary"],
          },
        },
      })
    );

    const parsed = JSON.parse((response.text ?? "{}").trim()) as unknown;
    const latencyMs = Number((performance.now() - t0).toFixed(1));

    res.json({
      status: "success",
      source: "gemini_multimodal_api",
      model: modelUsed,
      retained_on_disk: false,
      privacy_notice:
        "Analyzed transiently in memory. No facial image was persisted and no individual identity or sensitive personal characteristics were inferred.",
      latency_ms: latencyMs,
      analysis: parsed,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Gemini API request failed";
    console.error("[Gemini /api/gemini/analyze] Error:", message);
    res.status(502).json({
      status: "error",
      error_code: "GEMINI_UPSTREAM_ERROR",
      model: DEFAULT_GEMINI_MODEL,
      message,
    });
  }
});

// Server-Side Gemini AI Studio Chat Assistant
app.post("/api/gemini/chat", aiInferenceLimiter, async (req: Request, res: Response) => {
  const ai = createGeminiClient();
  if (!ai) {
    res.status(503).json({
      status: "error",
      error_code: "GEMINI_NOT_CONFIGURED",
      message: "GEMINI_API_KEY is not configured in the server environment.",
    });
    return;
  }

  const body = (req.body ?? {}) as {
    messages?: Array<{ role?: string; text?: string }>;
  };

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    res.status(400).json({
      status: "error",
      message: "Provide a non-empty messages array.",
    });
    return;
  }

  const sanitizedHistory = body.messages
    .slice(-14)
    .filter((m) => typeof m.text === "string" && m.text.trim().length > 0)
    .map((m) => ({
      role: m.role === "model" ? "model" : "user",
      parts: [{ text: String(m.text).trim().slice(0, 4000) }],
    }));

  if (sanitizedHistory.length === 0) {
    res.status(400).json({
      status: "error",
      message: "No valid message content provided.",
    });
    return;
  }

  try {
    const { result: response, modelUsed } = await callGeminiWithFallback((model, abortSignal) =>
      ai.models.generateContent({
        model,
        contents: sanitizedHistory,
        config: {
          abortSignal,
          systemInstruction: buildChatSystemInstruction(),
          thinkingConfig: {
            thinkingLevel:
              model === FALLBACK_GEMINI_MODEL ? ThinkingLevel.MINIMAL : ThinkingLevel.LOW,
          },
        },
      })
    );

    res.json({
      status: "success",
      model: modelUsed,
      reply: (response.text ?? "").trim(),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Gemini chat request failed";
    console.error("[Gemini /api/gemini/chat] Error:", message);
    res.status(502).json({
      status: "error",
      error_code: "GEMINI_UPSTREAM_ERROR",
      message,
    });
  }
});

// Server-Side Gemini AI Studio Streaming Chat Assistant (SSE)
app.post("/api/gemini/chat/stream", aiInferenceLimiter, async (req: Request, res: Response) => {
  const ai = createGeminiClient();
  if (!ai) {
    res.status(503).json({
      status: "error",
      error_code: "GEMINI_NOT_CONFIGURED",
      message: "GEMINI_API_KEY is not configured in the server environment.",
    });
    return;
  }

  const body = (req.body ?? {}) as {
    messages?: Array<{ role?: string; text?: string }>;
  };

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    res.status(400).json({
      status: "error",
      message: "Provide a non-empty messages array.",
    });
    return;
  }

  const sanitizedHistory = body.messages
    .slice(-14)
    .filter((m) => typeof m.text === "string" && m.text.trim().length > 0)
    .map((m) => ({
      role: m.role === "model" ? "model" : "user",
      parts: [{ text: String(m.text).trim().slice(0, 4000) }],
    }));

  if (sanitizedHistory.length === 0) {
    res.status(400).json({
      status: "error",
      message: "No valid message content provided.",
    });
    return;
  }

  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders?.();

  let clientClosed = false;
  const clientAbortController = new AbortController();
  res.on("close", () => {
    if (!res.writableEnded) {
      clientClosed = true;
      clientAbortController.abort();
    }
  });

  const sendSse = (payload: Record<string, unknown>) => {
    if (clientClosed || res.writableEnded) return;
    res.write(`data: ${JSON.stringify(payload)}\n\n`);
  };

  const attempts = [
    { model: DEFAULT_GEMINI_MODEL, timeoutMs: 22_000, backoffMs: 500 },
    { model: FALLBACK_GEMINI_MODEL, timeoutMs: 30_000, backoffMs: 900 },
    { model: FALLBACK_GEMINI_MODEL, timeoutMs: 35_000, backoffMs: 1400 },
    { model: DEFAULT_GEMINI_MODEL, timeoutMs: 35_000, backoffMs: 0 },
  ];

  const systemInstruction = buildChatSystemInstruction();
  let streamSucceeded = false;
  let lastErrorMessage = "Gemini streaming request failed.";

  for (let i = 0; i < attempts.length; i++) {
    if (clientClosed) break;
    const { model, timeoutMs, backoffMs } = attempts[i];
    const combinedSignal = AbortSignal.any([
      clientAbortController.signal,
      AbortSignal.timeout(timeoutMs),
    ]);

    let emittedAnyChunk = false;
    try {
      const stream = await ai.models.generateContentStream({
        model,
        contents: sanitizedHistory,
        config: {
          abortSignal: combinedSignal,
          systemInstruction,
          thinkingConfig: {
            thinkingLevel:
              model === FALLBACK_GEMINI_MODEL ? ThinkingLevel.MINIMAL : ThinkingLevel.LOW,
          },
        },
      });

      sendSse({ type: "meta", model });

      for await (const chunk of stream) {
        if (clientClosed) break;
        const text = chunk.text ?? "";
        if (text.length > 0) {
          emittedAnyChunk = true;
          sendSse({ type: "delta", text });
        }
      }

      if (!clientClosed) {
        sendSse({ type: "done", model });
      }
      streamSucceeded = true;
      break;
    } catch (err) {
      if (clientClosed) break;
      lastErrorMessage = err instanceof Error ? err.message : "Gemini streaming failed";
      const canRetry = !emittedAnyChunk && i < attempts.length - 1 && isTransientGeminiError(err);
      if (!canRetry) {
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, backoffMs));
    }
  }

  if (!streamSucceeded && !clientClosed) {
    console.error("[Gemini /api/gemini/chat/stream] Error:", lastErrorMessage);
    sendSse({ type: "error", message: lastErrorMessage });
  }

  if (!res.writableEnded) {
    res.end();
  }
});

// Static Repository Assets (Figures, Model Cards, Docs, Legacy UI)
app.use("/reports", express.static(path.join(PROJECT_ROOT, "reports")));
app.use("/models", express.static(path.join(PROJECT_ROOT, "models")));
app.use("/docs", express.static(path.join(PROJECT_ROOT, "docs")));
app.use("/legacy", express.static(path.join(PROJECT_ROOT, "frontend")));

// API 404 handler
app.use("/api", (_req: Request, res: Response) => {
  res.status(404).json({ error: "API endpoint not found" });
});

// ---------------------------------------------------------------------------
// Vite Dev Middleware / Production Static Serving + Server Boot
// ---------------------------------------------------------------------------

let serverInstance: ReturnType<typeof app.listen> | null = null;

async function startServer(): Promise<void> {
  const isProd = process.env.NODE_ENV === "production";
  const distDir = path.join(PROJECT_ROOT, "dist");

  if (!isProd) {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false, allowedHosts: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else if (fs.existsSync(distDir)) {
    app.use(express.static(distDir));
    app.get("/{*splat}", (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  } else {
    app.get("/", (_req: Request, res: Response) => {
      res.status(500).send("Production dist/ directory not built. Run npm run build.");
    });
  }

  serverInstance = app.listen(PORT, HOST, () => {
    console.log("======================================================================");
    console.log("  CELEBA AI — FACE INTELLIGENCE STUDIO");
    console.log(`  Listening on:          http://${HOST}:${PORT}`);
    console.log(`  Mode:                  ${isProd ? "production" : "development (Vite)"}`);
    console.log(`  Python ML Backend URL: ${PYTHON_ML_BACKEND_URL}`);
    console.log(`  Gemini API Key:        ${process.env.GEMINI_API_KEY ? "configured" : "not set"}`);
    console.log("======================================================================");
  });
}

const gracefulShutdown = (signal: string) => {
  console.log(`[Server] Received ${signal}. Shutting down...`);
  if (serverInstance) {
    serverInstance.close(() => process.exit(0));
  } else {
    process.exit(0);
  }
};

process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));

startServer().catch((err) => {
  console.error("[Server] Failed to start:", err);
  process.exit(1);
});
