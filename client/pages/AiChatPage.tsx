import { useState, type FormEvent } from "react";
import { AlertCircle, RotateCcw, Send } from "lucide-react";
import { GlassCard } from "../components/GlassCard";

interface ChatMessage {
  role: "user" | "model";
  text: string;
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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = async (textToSend: string) => {
    const trimmed = textToSend.trim();
    if (!trimmed || loading) return;

    const nextMessages: ChatMessage[] = [...messages, { role: "user", text: trimmed }];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const resp = await fetch("/api/gemini/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: nextMessages }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.message || `Chat error (HTTP ${resp.status})`);
      } else {
        setMessages((prev) => [...prev, { role: "model", text: data.reply || "No reply returned." }]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to communicate with Gemini chat API.");
    } finally {
      setLoading(false);
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
            Server-side Gemini assistant grounded with repository benchmarks, model registry, and
            reproducibility specs.
          </p>
        </div>

        <button
          type="button"
          onClick={() => {
            setMessages([
              {
                role: "model",
                text: "Conversation reset. Ask anything about CelebA AI models, metrics, or APIs.",
              },
            ]);
            setError(null);
          }}
          className="px-3.5 py-2 min-h-[40px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-2 self-start"
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
            disabled={loading}
            onClick={() => void sendMessage(prompt)}
            className="px-3 py-2 rounded-lg border border-white/10 bg-slate-900/70 hover:border-sky-400/40 hover:bg-slate-800/80 text-xs text-slate-300 hover:text-slate-100 transition-colors text-left"
          >
            {prompt}
          </button>
        ))}
      </div>

      <GlassCard className="flex flex-col h-[560px]">
        <div className="flex-1 overflow-y-auto space-y-4 pr-1">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                  msg.role === "user"
                    ? "bg-sky-500 text-slate-950 font-medium"
                    : "bg-slate-950/80 border border-white/10 text-slate-200"
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="rounded-xl px-4 py-3 text-sm bg-slate-950/80 border border-white/10 text-slate-400">
                Generating grounded response...
              </div>
            </div>
          )}
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
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="px-4 py-2.5 min-h-[42px] rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-40 text-slate-950 text-sm font-semibold inline-flex items-center gap-2 transition-colors"
          >
            <Send className="w-4 h-4" />
            <span>Send</span>
          </button>
        </form>
      </GlassCard>
    </div>
  );
}
