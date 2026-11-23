import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import type { PageId } from "../types";

interface SlideItem {
  id: string;
  kicker: string;
  title: string;
  subtitle: string;
  highlights: { label: string; value: string }[];
  ctaLabel: string;
  targetPage: PageId;
}

interface HeroShowcaseSliderProps {
  totalImages: number;
  uniqueIdentities: number;
  topAccuracyPct: string;
  topRocAuc: string;
  identityLeakageCount: number;
  onNavigate: (page: PageId) => void;
}

export function HeroShowcaseSlider({
  totalImages,
  uniqueIdentities,
  topAccuracyPct,
  topRocAuc,
  identityLeakageCount,
  onNavigate,
}: HeroShowcaseSliderProps) {
  const slides: SlideItem[] = [
    {
      id: "architecture",
      kicker: "01. CelebA Face Intelligence Suite",
      title: "Multi-Architecture Facial Attribute Studio",
      subtitle:
        "Benchmarked across PyTorch Residual CNN, Spatial BiLSTM, Vision Transformer, ANN, XGBoost, and server-side Gemini multimodal assessment.",
      highlights: [
        { label: "Dataset Scale", value: `${totalImages.toLocaleString()} faces` },
        { label: "Identities", value: uniqueIdentities.toLocaleString() },
        { label: "Attributes", value: "40 binary traits" },
      ],
      ctaLabel: "Open Image Analysis",
      targetPage: "image-analysis",
    },
    {
      id: "benchmarks",
      kicker: "02. Identity-Disjoint Evaluation",
      title: "Verified Real-Image Holdout Benchmarks",
      subtitle:
        "Evaluated on 75 identity-disjoint CelebA holdout faces in production_benchmark.csv, strictly separated from synthetic CI smoke-test runs.",
      highlights: [
        { label: "Peak Holdout Acc", value: topAccuracyPct },
        { label: "Top ROC-AUC", value: topRocAuc },
        { label: "5-Fold Ensemble", value: "SLSQP + TTA" },
      ],
      ctaLabel: "Compare Models",
      targetPage: "model-comparison",
    },
    {
      id: "fairness",
      kicker: "03. Leakage & Fairness Audit",
      title: "Zero-Leakage Partitioning & EEOC 4/5ths Audit",
      subtitle:
        "GroupShuffleSplit and 5-fold StratifiedGroupKFold prevent person_id overlap across splits while auditing subgroup parity across attributes.",
      highlights: [
        { label: "Identity Overlap", value: `${identityLeakageCount}.0%` },
        { label: "Male / Female DI", value: "0.9275" },
        { label: "Young / Senior DI", value: "0.8832" },
      ],
      ctaLabel: "View Performance Reports",
      targetPage: "performance-reports",
    },
    {
      id: "privacy",
      kicker: "04. Privacy & Security Controls",
      title: "Transient In-Memory Vision & Camera Consent",
      subtitle:
        "Magic-byte JPEG/PNG/WebP verification, server-side Gemini key isolation, explicit browser camera consent, and zero disk retention.",
      highlights: [
        { label: "Disk Retention", value: "0 bytes stored" },
        { label: "Upload Cap", value: "10 MB verified" },
        { label: "Rate Limiter", value: "25 req / 60s" },
      ],
      ctaLabel: "Open Live Camera",
      targetPage: "camera-analysis",
    },
  ];

  const [activeIndex, setActiveIndex] = useState(0);
  const [autoPlay, setAutoPlay] = useState(() => {
    if (typeof window !== "undefined" && window.matchMedia) {
      return !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    }
    return true;
  });

  useEffect(() => {
    if (!autoPlay) return;
    const timer = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % slides.length);
    }, 6500);
    return () => clearInterval(timer);
  }, [autoPlay, slides.length]);

  const activeSlide = slides[activeIndex];

  const goPrev = () => {
    setActiveIndex((prev) => (prev - 1 + slides.length) % slides.length);
  };

  const goNext = () => {
    setActiveIndex((prev) => (prev + 1) % slides.length);
  };

  return (
    <section
      aria-roledescription="carousel"
      aria-label="CelebA Studio Highlights"
      onKeyDown={(e) => {
        if (e.key === "ArrowLeft") goPrev();
        if (e.key === "ArrowRight") goNext();
      }}
      tabIndex={0}
      className="relative overflow-hidden rounded-xl border border-white/10 bg-gradient-to-br from-[#0F172A] via-[#0C1324] to-[#0A101E] p-6 sm:p-8 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
    >
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* Left: Slide Copy & Verified Highlights */}
        <div className="lg:col-span-8 space-y-4">
          <div className="text-xs font-medium text-sky-400 tracking-wide">
            {activeSlide.kicker}
          </div>

          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-50 text-balance max-w-2xl">
            {activeSlide.title}
          </h2>

          <p className="text-sm text-slate-300 leading-relaxed max-w-2xl">
            {activeSlide.subtitle}
          </p>

          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 pt-1 text-xs sm:text-sm">
            {activeSlide.highlights.map((item) => (
              <div key={item.label} className="flex items-baseline gap-2">
                <span className="text-slate-400">{item.label}:</span>
                <span className="font-mono font-semibold text-slate-100 tabular-nums">
                  {item.value}
                </span>
              </div>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              type="button"
              onClick={() => onNavigate(activeSlide.targetPage)}
              className="px-4 py-2.5 min-h-[40px] rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs sm:text-sm font-semibold transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-300"
            >
              {activeSlide.ctaLabel}
            </button>
            <button
              type="button"
              onClick={() => onNavigate("dataset-analytics")}
              className="px-4 py-2.5 min-h-[40px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-slate-100 text-xs sm:text-sm font-medium transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
            >
              Explore Dataset
            </button>
          </div>
        </div>

        {/* Right: Geometric Facial Mesh & Neural Connections Diagram */}
        <div className="lg:col-span-4 flex items-center justify-center">
          <div className="w-full max-w-[260px] aspect-[4/3] rounded-lg border border-white/10 bg-slate-950/60 p-4 flex items-center justify-center">
            <svg
              viewBox="0 0 240 180"
              className="w-full h-full"
              role="img"
              aria-label="Geometric facial landmark and neural network wireframe illustration"
            >
              {/* Subtle Neural Connection Grid */}
              <g stroke="rgba(56, 189, 248, 0.22)" strokeWidth="1">
                <line x1="34" y1="42" x2="88" y2="56" />
                <line x1="34" y1="90" x2="78" y2="90" />
                <line x1="34" y1="138" x2="88" y2="124" />
                <line x1="206" y1="42" x2="152" y2="56" />
                <line x1="206" y1="90" x2="162" y2="90" />
                <line x1="206" y1="138" x2="152" y2="124" />
              </g>

              {/* Outer Face Contour Wireframe */}
              <path
                d="M120 26 C88 26 72 52 74 88 C76 122 96 150 120 156 C144 150 164 122 166 88 C168 52 152 26 120 26 Z"
                fill="rgba(15, 23, 42, 0.55)"
                stroke="rgba(148, 163, 184, 0.55)"
                strokeWidth="1.25"
              />

              {/* Internal Landmark Mesh */}
              <g stroke="rgba(56, 189, 248, 0.45)" strokeWidth="1" fill="none">
                <polyline points="92,74 108,70 120,96 132,70 148,74" />
                <polyline points="88,76 120,104 152,76" />
                <polyline points="100,122 120,128 140,122" />
                <line x1="120" y1="40" x2="120" y2="146" strokeDasharray="3 3" />
              </g>

              {/* Landmark & Input Layer Nodes */}
              <g fill="#38BDF8">
                <circle cx="96" cy="74" r="2.5" />
                <circle cx="144" cy="74" r="2.5" />
                <circle cx="120" cy="98" r="2.5" />
                <circle cx="102" cy="122" r="2.5" />
                <circle cx="138" cy="122" r="2.5" />
                <circle cx="120" cy="146" r="2.5" />
              </g>

              {/* Left / Right Neural Layer Nodes */}
              <g fill="#94A3B8">
                <circle cx="34" cy="42" r="3" />
                <circle cx="34" cy="90" r="3" />
                <circle cx="34" cy="138" r="3" />
                <circle cx="206" cy="42" r="3" />
                <circle cx="206" cy="90" r="3" />
                <circle cx="206" cy="138" r="3" />
              </g>
            </svg>
          </div>
        </div>
      </div>

      {/* Carousel Controls Bar */}
      <div className="mt-6 pt-4 border-t border-white/10 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2" role="tablist" aria-label="Slide selector">
          {slides.map((slide, idx) => {
            const isCurrent = idx === activeIndex;
            return (
              <button
                key={slide.id}
                type="button"
                role="tab"
                aria-selected={isCurrent}
                aria-label={`Show slide ${idx + 1}: ${slide.title}`}
                onClick={() => setActiveIndex(idx)}
                className={`h-2 rounded-full transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                  isCurrent ? "w-7 bg-sky-400" : "w-2.5 bg-slate-600 hover:bg-slate-400"
                }`}
              />
            );
          })}
          <span className="ml-2 text-xs font-mono text-slate-400 tabular-nums">
            {activeIndex + 1} / {slides.length}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setAutoPlay((v) => !v)}
            aria-label={autoPlay ? "Pause slide rotation" : "Resume slide rotation"}
            className="px-3 py-1.5 min-h-[34px] rounded-md border border-white/10 bg-white/5 hover:bg-white/10 text-xs text-slate-300 inline-flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            {autoPlay ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{autoPlay ? "Pause" : "Play"}</span>
          </button>

          <button
            type="button"
            onClick={goPrev}
            aria-label="Previous slide"
            className="p-2 min-w-[36px] min-h-[36px] rounded-md border border-white/10 bg-white/5 hover:bg-white/10 text-slate-200 flex items-center justify-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <button
            type="button"
            onClick={goNext}
            aria-label="Next slide"
            className="p-2 min-w-[36px] min-h-[36px] rounded-md border border-white/10 bg-white/5 hover:bg-white/10 text-slate-200 flex items-center justify-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </section>
  );
}
