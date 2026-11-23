import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";

// Suppress noisy Vite HMR WebSocket connection failures in AI Studio preview iframe
if (typeof window !== "undefined") {
  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason;
    const msg = String(reason?.message || reason || "");
    if (msg.includes("WebSocket") || msg.includes("@vite/client") || msg.includes("closed without opened")) {
      event.preventDefault();
    }
  });

  const originalConsoleError = console.error;
  console.error = (...args: unknown[]) => {
    const combined = args.map((a) => String(a)).join(" ");
    if (
      combined.includes("WebSocket") ||
      combined.includes("@vite/client") ||
      combined.includes("closed without opened") ||
      combined.includes("ERR_CONNECTION_REFUSED")
    ) {
      return;
    }
    originalConsoleError(...args);
  };
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);

