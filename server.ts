import express from "express";
import type { Request, Response, NextFunction } from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GoogleGenAI, Type } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = __dirname;

const PORT = Number(process.env.PORT) || 3000;
const HOST = "0.0.0.0";
const DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen";

interface GalleryConfidence {
  smiling: number;
  young: number;
  male: number;
  eyeglasses: number;
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
  confidence?: GalleryConfidence;
}

const SYSTEM_INSTRUCTION = `
You are an expert Computer Vision and Facial Biometrics AI Assistant.
Analyze the provided face image according to the CelebA biometrics schema and return ONLY a valid JSON object matching the requested schema.
`.trim();

function loadGalleryData(): GalleryItem[] {
  const jsonPath = path.join(PROJECT_ROOT, "frontend", "gallery_data.json");
  if (!fs.existsSync(jsonPath)) {
    return [];
  }
  try {
    const raw = fs.readFileSync(jsonPath, "utf-8");
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as GalleryItem[]) : [];
  } catch (err) {
    console.error("Failed to load gallery_data.json:", err);
    return [];
  }
}

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

function loadBenchmarkMetrics(): Record<string, string | number>[] {
  const candidates = [
    path.join(PROJECT_ROOT, "reports", "benchmarks", "production_benchmark.csv"),
    path.join(PROJECT_ROOT, "reports", "benchmarks", "model_benchmark_results.csv"),
    path.join(PROJECT_ROOT, "reports", "model_benchmark_results.csv"),
  ];
  const csvPath = candidates.find((p) => fs.existsSync(p)) || "";
  if (!csvPath) {
    return [];
  }
  try {
    const raw = fs.readFileSync(csvPath, "utf-8");
    const lines = raw
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
    if (lines.length < 2) {
      return [];
    }
    const headers = parseCsvLine(lines[0]);
    return lines.slice(1).map((line) => {
      const values = parseCsvLine(line);
      const record: Record<string, string | number> = {};
      headers.forEach((header, idx) => {
        const val = values[idx] ?? "";
        const num = Number(val);
        record[header] = val !== "" && !Number.isNaN(num) ? num : val;
      });
      return record;
    });
  } catch (err) {
    console.error("Failed to load model_benchmark_results.csv:", err);
    return [];
  }
}

function loadModelRegistry(): Record<string, unknown> {
  const registryPath = path.join(PROJECT_ROOT, "models", "model_registry.json");
  if (!fs.existsSync(registryPath)) {
    return { evaluated_models: [] };
  }
  try {
    return JSON.parse(fs.readFileSync(registryPath, "utf-8")) as Record<string, unknown>;
  } catch (err) {
    console.error("Failed to load model_registry.json:", err);
    return { evaluated_models: [] };
  }
}

function executeMcpFaceIdentification(imageId: string) {
  const cleanId = imageId.replace(/\.png$/i, ".jpg").replace(/\.jpeg$/i, ".jpg");
  const reqId = Date.now() % 100000;
  const startTime = performance.now();

  const mcpRequest = {
    jsonrpc: "2.0",
    id: reqId,
    method: "tools/call",
    params: {
      name: "mysql_identify_face",
      arguments: { image_id: cleanId, database: "celeba_db" },
    },
  };

  const gallery = loadGalleryData();
  const stem = cleanId.replace(/\.jpg$/i, "");
  const matched = gallery.find((g) => g.filename.includes(stem));
  const execMs = Number((performance.now() - startTime).toFixed(2));

  if (!matched) {
    const mcpErrResponse = {
      jsonrpc: "2.0",
      id: reqId,
      error: {
        code: -32602,
        message: `Image ID ${cleanId} not found in celeba_db or gallery index`,
        details: "Curated gallery index fallback active (external MySQL unavailable in ephemeral runtime)",
      },
    };
    return { request: mcpRequest, response: mcpErrResponse, structured: null };
  }

  const pad = (val: string | number, width = 23) => String(val).padEnd(width, " ");
  const gender = matched.male ? "Male" : "Female";
  const expression = matched.smiling ? "Smiling" : "Neutral";
  const eyeglasses = matched.eyeglasses ? "Yes" : "No";
  const ageGroup = matched.young ? "Young" : "Senior";
  const clusterCount = gallery.filter((g) => g.cluster_id === matched.cluster_id).length;

  const formattedTable = [
    "+-----------------------------+-------------------------+",
    "| Metric                      | Value                   |",
    "+-----------------------------+-------------------------+",
    `| Image ID                    | ${pad(cleanId)} |`,
    `| Detected Person             | ${pad(matched.person_name)} |`,
    `| Cluster / Identity ID       | #${pad(matched.ground_truth_id, 22)} |`,
    `| Gender                      | ${pad(gender)} |`,
    `| Expression                  | ${pad(expression)} |`,
    `| Eyeglasses                  | ${pad(eyeglasses)} |`,
    `| Age Group                   | ${pad(ageGroup)} |`,
    `| Status                      | ${pad("Curated Fallback Index")} |`,
    `| Provenance                  | ${pad("Ground-Truth Reference")} |`,
    "+-----------------------------+-------------------------+",
  ].join("\n");

  const structuredFallback = {
    image_id: cleanId,
    person_name: matched.person_name,
    identity_cluster_id: matched.ground_truth_id,
    gender,
    age_group: ageGroup,
    expression,
    eyeglasses,
    total_photos: clusterCount,
    confidence: matched.confidence?.young ?? 0.98,
    is_fallback: true,
    source: "Curated Ground-Truth Reference (In-Memory Gallery Index)",
  };

  const mcpResponse = {
    jsonrpc: "2.0",
    id: reqId,
    result: {
      content: [{ type: "text", text: formattedTable }],
      structured: structuredFallback,
      source: structuredFallback.source,
      executionTimeMs: execMs,
    },
  };

  return { request: mcpRequest, response: mcpResponse, structured: structuredFallback };
}

function resolveGeminiModel(requestedModel?: string): string {
  const deprecated = new Set([
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
    "gemini-2.0-flash",
    "gemini-2.0-pro",
    "gemini-2.0-flash-thinking",
  ]);
  if (!requestedModel || deprecated.has(requestedModel.trim().toLowerCase())) {
    return DEFAULT_GEMINI_MODEL;
  }
  return requestedModel.trim();
}

function extractInlineImage(payload: {
  image?: string;
  image_path?: string;
  image_base64?: string;
  mime_type?: string;
}): { mimeType: string; data: string; label: string } | null {
  if (typeof payload.image_base64 === "string" && payload.image_base64.trim().length > 0) {
    const raw = payload.image_base64.trim();
    const dataUrlMatch = /^data:([^;]+);base64,(.+)$/s.exec(raw);
    if (dataUrlMatch) {
      return {
        mimeType: dataUrlMatch[1],
        data: dataUrlMatch[2],
        label: payload.image || "uploaded_image",
      };
    }
    return {
      mimeType: payload.mime_type || "image/jpeg",
      data: raw,
      label: payload.image || "uploaded_image",
    };
  }

  const target = (payload.image || payload.image_path || "").trim();
  const gallery = loadGalleryData();

  if (target) {
    const stem = path.basename(target).replace(/\.(png|jpe?g)$/i, "");
    const match = gallery.find((g) => g.filename.includes(stem));
    if (match?.thumbnail) {
      const m = /^data:([^;]+);base64,(.+)$/s.exec(match.thumbnail);
      if (m) {
        return { mimeType: m[1], data: m[2], label: match.filename };
      }
    }

    const safeResolved = path.resolve(PROJECT_ROOT, target);
    if (
      safeResolved.startsWith(PROJECT_ROOT + path.sep) &&
      fs.existsSync(safeResolved) &&
      fs.statSync(safeResolved).isFile()
    ) {
      const ext = path.extname(safeResolved).toLowerCase();
      const mimeType = ext === ".png" ? "image/png" : "image/jpeg";
      const data = fs.readFileSync(safeResolved).toString("base64");
      return { mimeType, data, label: path.basename(safeResolved) };
    }
  } else if (gallery.length > 0 && gallery[0].thumbnail) {
    const m = /^data:([^;]+);base64,(.+)$/s.exec(gallery[0].thumbnail);
    if (m) {
      return { mimeType: m[1], data: m[2], label: gallery[0].filename };
    }
  }

  return null;
}

const app = express();
app.use(express.json({ limit: "15mb" }));

// CORS headers matching original server
app.use((req: Request, res: Response, next: NextFunction) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }
  next();
});

// Primary web interface routes
app.get(["/", "/index.html"], (_req: Request, res: Response) => {
  const indexPath = path.join(PROJECT_ROOT, "frontend", "index.html");
  if (!fs.existsSync(indexPath)) {
    res.status(404).send("frontend/index.html not found");
    return;
  }
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.sendFile(indexPath);
});

// Gallery JSON endpoints
app.get(["/api/gallery", "/gallery_data.json"], (_req: Request, res: Response) => {
  res.json(loadGalleryData());
});

// Container pre-warming, load balancer, and health probe endpoints
app.get(["/health", "/healthz", "/_ah/health", "/api/health", "/ping"], (_req: Request, res: Response) => {
  res.json({
    status: "healthy",
    service: "CelebA Face Intelligence",
    provider: "Gemini API",
    models: [DEFAULT_GEMINI_MODEL],
    ready: true,
  });
});

// Benchmarks CSV -> JSON endpoint
app.get("/api/benchmarks", (_req: Request, res: Response) => {
  res.json(loadBenchmarkMetrics());
});

// Model registry JSON endpoint
app.get("/api/models", (_req: Request, res: Response) => {
  res.json(loadModelRegistry());
});

// Server-side Gemini multimodal face analysis endpoint
app.post("/api/analyze", async (req: Request, res: Response) => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    res.status(503).json({
      status: "error",
      message: "GEMINI_API_KEY is not configured in the server environment.",
    });
    return;
  }

  const body = (req.body ?? {}) as {
    image?: string;
    image_path?: string;
    image_base64?: string;
    mime_type?: string;
    model?: string;
    additional_context?: string;
  };

  const inlineImage = extractInlineImage(body);
  if (!inlineImage) {
    res.status(400).json({
      status: "error",
      message: "Target image could not be resolved. Provide image_base64 or a valid gallery filename.",
    });
    return;
  }

  const modelName = resolveGeminiModel(body.model);

  try {
    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });

    let promptText =
      "Analyze this face image and extract facial attributes according to CelebA standards.";
    if (body.additional_context) {
      promptText += `\nDatabase Context: ${body.additional_context}`;
    }

    const response = await ai.models.generateContent({
      model: modelName,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: inlineImage.mimeType,
              data: inlineImage.data,
            },
          },
          { text: promptText },
        ],
      },
      config: {
        systemInstruction: SYSTEM_INSTRUCTION,
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
            demographics: {
              type: Type.OBJECT,
              properties: {
                apparent_age_group: { type: Type.STRING },
                apparent_gender: { type: Type.STRING },
                confidence: { type: Type.NUMBER },
              },
              required: ["apparent_age_group", "apparent_gender", "confidence"],
            },
            facial_attributes: {
              type: Type.OBJECT,
              properties: {
                hair_color: { type: Type.STRING },
                wearing_eyeglasses: { type: Type.BOOLEAN },
                facial_hair: { type: Type.BOOLEAN },
                notable_features: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
              },
              required: [
                "hair_color",
                "wearing_eyeglasses",
                "facial_hair",
                "notable_features",
              ],
            },
            reasoning_summary: { type: Type.STRING },
          },
          required: [
            "expression",
            "demographics",
            "facial_attributes",
            "reasoning_summary",
          ],
        },
      },
    });

    const parsedAnalysis: unknown = JSON.parse((response.text ?? "{}").trim());

    res.json({
      status: "success",
      image: inlineImage.label,
      google_ai_studio: {
        status: "success",
        source: "gemini_api",
        model: modelName,
        image: inlineImage.label,
        analysis: parsedAnalysis,
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Gemini API request failed";
    console.error("Gemini /api/analyze error:", err);
    res.status(502).json({
      status: "error",
      model: modelName,
      message,
    });
  }
});

// MCP Face Identification JSON-RPC 2.0 endpoint
app.post(["/api/mcp", "/api/mcp/identify"], (req: Request, res: Response) => {
  const body = (req.body ?? {}) as { image_id?: string };
  const imageId = typeof body.image_id === "string" && body.image_id.trim()
    ? body.image_id.trim()
    : "001553.jpg";
  const result = executeMcpFaceIdentification(imageId);
  res.json(result);
});

// Static assets serving (safe subdirectories only)
app.use("/frontend", express.static(path.join(PROJECT_ROOT, "frontend")));
app.use("/reports", express.static(path.join(PROJECT_ROOT, "reports")));
app.use("/models", express.static(path.join(PROJECT_ROOT, "models")));
app.use("/data", express.static(path.join(PROJECT_ROOT, "data")));
app.use("/docs", express.static(path.join(PROJECT_ROOT, "docs")));

app.use((_req: Request, res: Response) => {
  res.status(404).json({ error: "Endpoint not found" });
});

app.listen(PORT, HOST, () => {
  console.log("======================================================================");
  console.log("  CELEBA FACE INTELLIGENCE - WEB APPLICATION SERVER");
  console.log(`  Server listening on: http://${HOST}:${PORT}`);
  console.log(`  Web Interface: http://localhost:${PORT}/`);
  console.log("======================================================================");
});
