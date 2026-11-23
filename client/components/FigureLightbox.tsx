import { useEffect } from "react";
import { Download, X } from "lucide-react";

export interface LightboxFigure {
  title: string;
  url: string;
  filename: string;
  subtitle?: string;
}

interface FigureLightboxProps {
  figure: LightboxFigure | null;
  onClose: () => void;
}

export function FigureLightbox({ figure, onClose }: FigureLightboxProps) {
  useEffect(() => {
    if (!figure) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [figure, onClose]);

  if (!figure) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={figure.title}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-5xl rounded-2xl border border-white/15 bg-[#0C1220] shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-3.5 border-b border-white/10 flex items-center justify-between gap-4">
          <div className="min-w-0">
            <h4 className="text-sm sm:text-base font-semibold text-slate-100 truncate">
              {figure.title}
            </h4>
            <div className="text-xs font-mono text-slate-400 truncate">
              {figure.subtitle ?? figure.filename}
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <a
              href={figure.url}
              download={figure.filename}
              className="px-3 py-1.5 min-h-[36px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-medium text-slate-200 inline-flex items-center gap-1.5 whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download</span>
            </a>
            <button
              type="button"
              aria-label="Close lightbox"
              onClick={onClose}
              className="p-2 min-w-[36px] min-h-[36px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/15 text-slate-300 hover:text-white flex items-center justify-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="p-4 sm:p-6 bg-white flex items-center justify-center max-h-[78vh] overflow-auto">
          <img
            src={figure.url}
            alt={figure.title}
            referrerPolicy="no-referrer"
            className="max-w-full max-h-[70vh] object-contain"
          />
        </div>
      </div>
    </div>
  );
}
