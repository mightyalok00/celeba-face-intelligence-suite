import { useEffect, useRef, useState, type FormEvent } from "react";
import { AlertCircle, RotateCcw, Send, Square } from "lucide-react";
import { GlassCard } from "../components/GlassCard";
import { MarkdownMessage } from "../components/MarkdownMessage";

interface ChatMessage {
  role: "user" | "model";
  text: string;
  model?: string;
}

const STARTER_PROMPTS = [
  "Compare real-image holdout benchmarks vs synthetic CI smoke-test metrics.",
  "Explain how StratifiedGroupKFold prevents identity leakage in CelebA.",
  "How are the SLSQP ensemble weights computed across XGBoost, CNN, and ViT?",
  "Summarize the EEOC 4/5ths fairness audit findings for Male and Young subgroups.",
];

export function AiChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "model",
      text: "Ask me about the CelebA dataset partitioning, PyTorch/XGBoost model benchmarks, SLSQP ensemble optimization, fairness audit, or API endpoints.",
    },
  ]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, streaming]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const stopStreaming = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
  };

  const sendMessage = async (textToSend: string) => {
    const trimmed = textToSend.trim();
    if (!trimmed || streaming) return;

    const historyForApi = [...messages, { role: "user" as const, text: trimmed }];
    const assistantIndex = historyForApi.length;

    setMessages([...historyForApi, { role: "model", text: "" }]);
    setInput("");
    setStreaming(true);
    setError(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const resp = await fetch("/api/gemini/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: historyForApi.map((m) => ({ role: m.role, text: m.text })),
        }),
        signal: controller.signal,
      });

      const contentType = resp.headers.get("content-type") || "";
      if (!resp.ok || !contentType.includes("text/event-stream") || !resp.body) {
        const errPayload = (await resp.json().catch(() => ({}))) as { message?: string };
        throw new Error(errPayload.message || `Chat error (HTTP ${resp.status})`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const eventBlock = buffer.slice(0, boundary).trim();
          buffer = buffer.slice(boundary + 2);
          boundary = buffer.indexOf("\n\n");

          if (!eventBlock) continue;
          const dataLines = eventBlock
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim());

          for (const rawJson of dataLines) {
            if (!rawJson) continue;
            const evt = JSON.parse(rawJson) as {
              type?: string;
              text?: string;
              model?: string;
              message?: string;
            };

            if (evt.type === "meta" && evt.model) {
              setMessages((prev) =>
                prev.map((m, i) => (i === assistantIndex ? { ...m, model: evt.model } : m))
              );
            } else if (evt.type === "delta" && evt.text) {
              setMessages((prev) =>
                prev.map((m, i) =>
                  i === assistantIndex ? { ...m, text: `${m.text}${evt.text}` } : m
                )
              );
            } else if (evt.type === "error") {
              setError(evt.message || "Streaming error occurred.");
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setMessages((prev) =>
          prev.map((m, i) =>
            i === assistantIndex && !m.text.trim()
              ? { ...m, text: "_Response stopped by user._" }
              : m
          )
        );
      } else {
        setError(err instanceof Error ? err.message : "Failed to communicate with Gemini chat API.");
        setMessages((prev) =>
          prev.filter((m, i) => !(i === assistantIndex && m.role === "model" && !m.text.trim()))
        );
      }
    } finally {
      abortRef.current = null;
      setStreaming(false);
    }
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    void sendMessage(input);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-slate-50">AI Research & Engineering Assistant</h2>
          <p className="text-sm text-slate-400 mt-1">
            Real-time streaming Gemini assistant grounded with repository benchmarks, model
            registry, and fairness audit specs.
          </p>
        </div>

        <button
          type="button"
          onClick={() => {
            stopStreaming();
            setMessages([
              {
                role: "model",
                text: "Conversation reset. Ask anything about CelebA AI models, metrics, or APIs.",
              },
            ]);
            setError(null);
          }}
          className="px-3.5 py-2 min-h-[40px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-2 self-start whitespace-nowrap"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset Chat</span>
        </button>
      </div>

      {/* Starter Prompts */}
      <div className="flex flex-wrap gap-2">
        {STARTER_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            disabled={streaming}
            onClick={() => void sendMessage(prompt)}
            className="px-3 py-2 rounded-lg border border-white/10 bg-slate-900/70 hover:border-sky-400/40 hover:bg-slate-800/80 disabled:opacity-50 text-xs text-slate-300 hover:text-slate-100 transition-colors text-left"
          >
            {prompt}
          </button>
        ))}
      </div>

      <GlassCard className="flex flex-col h-[580px]">
        <div className="flex-1 overflow-y-auto space-y-4 pr-1">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[88%] rounded-xl px-4 py-3 text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-sky-500 text-slate-950 font-medium whitespace-pre-wrap"
                    : "bg-slate-950/85 border border-white/10 text-slate-200 space-y-2"
                }`}
              >
                {msg.role === "user" ? (
                  msg.text
                ) : (
                  <>
                    {msg.text ? (
                      <MarkdownMessage content={msg.text} />
                    ) : (
                      <span className="text-xs text-slate-400">Streaming response...</span>
                    )}
                    {msg.model && (
                      <div className="pt-1.5 border-t border-white/5 text-[11px] font-mono text-slate-400 tabular-nums">
                        Model: {msg.model}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {error && (
          <div className="mt-3 p-3 rounded-lg border border-red-500/40 bg-red-950/25 text-xs text-red-200 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="mt-4 pt-4 border-t border-white/10 flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about ViT vs XGBoost performance, checkpoints, or CelebA setup..."
            className="flex-1 rounded-lg border border-white/15 bg-slate-950 px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-400"
          />
          {streaming ? (
            <button
              type="button"
              onClick={stopStreaming}
              className="px-4 py-2.5 min-h-[42px] rounded-lg border border-amber-400/40 bg-amber-500/15 hover:bg-amber-500/25 text-amber-200 text-sm font-semibold inline-flex items-center gap-2 transition-colors whitespace-nowrap"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Stop</span>
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="px-4 py-2.5 min-h-[42px] rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-40 text-slate-950 text-sm font-semibold inline-flex items-center gap-2 transition-colors whitespace-nowrap"
            >
              <Send className="w-4 h-4" />
              <span>Send</span>
            </button>
          )}
        </form>
      </GlassCard>
    </div>
  );
}
