import { useState, type ReactNode } from "react";
import {
  Activity,
  BarChart3,
  Bot,
  Camera,
  Database,
  FileBarChart2,
  Image as ImageIcon,
  Layers,
  Menu,
  Search,
  X,
} from "lucide-react";
import type { PageId } from "../types";

interface NavItem {
  id: PageId;
  label: string;
  icon: typeof Activity;
}

const NAV_ITEMS: NavItem[] = [
  { id: "dashboard", label: "Intelligence Dashboard", icon: Activity },
  { id: "image-analysis", label: "Image Analysis", icon: ImageIcon },
  { id: "camera-analysis", label: "Live Camera", icon: Camera },
  { id: "model-comparison", label: "Model Comparison", icon: Layers },
  { id: "dataset-analytics", label: "Dataset Analytics", icon: Database },
  { id: "performance-reports", label: "Performance Reports", icon: FileBarChart2 },
  { id: "ai-chat", label: "AI Chat Assistant", icon: Bot },
  { id: "system-health", label: "System Health & API", icon: BarChart3 },
];

const HEADER_QUICK_LINKS: { id: PageId; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "model-comparison", label: "Models" },
  { id: "dataset-analytics", label: "Dataset" },
  { id: "performance-reports", label: "Reports" },
];

const FOOTER_LINKS: { id: PageId; label: string }[] = [
  { id: "dashboard", label: "Overview" },
  { id: "image-analysis", label: "Image Analysis" },
  { id: "model-comparison", label: "Model Comparison" },
  { id: "performance-reports", label: "Benchmarks" },
  { id: "system-health", label: "System & API Docs" },
];

interface LayoutProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  onOpenSearch?: () => void;
  children: ReactNode;
}

export function Layout({ activePage, onNavigate, onOpenSearch, children }: LayoutProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const currentNav = NAV_ITEMS.find((item) => item.id === activePage) ?? NAV_ITEMS[0];

  const handleSelect = (id: PageId) => {
    onNavigate(id);
    setMobileMenuOpen(false);
  };

  return (
    <div className="min-h-screen bg-[#090D16] text-slate-100 flex">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex lg:w-64 lg:flex-col lg:fixed lg:inset-y-0 border-r border-white/10 bg-[#0C1220]/95 backdrop-blur-md">
        <div className="h-16 px-5 flex items-center border-b border-white/10">
          <button
            type="button"
            onClick={() => handleSelect("dashboard")}
            className="text-left text-base font-semibold tracking-tight text-slate-50 whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 rounded-md"
          >
            CelebA AI Studio
          </button>
        </div>

        <nav aria-label="Primary navigation" className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleSelect(item.id)}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 min-h-[42px] rounded-lg text-sm font-medium transition-colors whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                  isActive
                    ? "bg-sky-500/15 text-sky-300 border border-sky-400/30"
                    : "text-slate-300 hover:bg-white/5 hover:text-slate-100 border border-transparent"
                }`}
              >
                <Icon className="w-4 h-4 shrink-0" aria-hidden="true" />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t border-white/10 text-xs text-slate-400 space-y-1">
          <div className="font-medium text-slate-300">Privacy & Zero Retention</div>
          <div>Uploaded frames stay in memory for single-request inference only.</div>
        </div>
      </aside>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-black/70 backdrop-blur-sm"
            onClick={() => setMobileMenuOpen(false)}
          />
          <aside className="relative w-72 max-w-[85vw] bg-[#0C1220] border-r border-white/10 flex flex-col z-10">
            <div className="h-16 px-5 flex items-center justify-between border-b border-white/10">
              <span className="text-base font-semibold text-slate-50">CelebA AI Studio</span>
              <button
                type="button"
                aria-label="Close navigation menu"
                onClick={() => setMobileMenuOpen(false)}
                className="p-2 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-lg text-slate-300 hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <nav aria-label="Mobile navigation" className="flex-1 p-3 space-y-1 overflow-y-auto">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const isActive = activePage === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleSelect(item.id)}
                    className={`w-full flex items-center gap-3 px-3.5 py-3 min-h-[44px] rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${
                      isActive
                        ? "bg-sky-500/15 text-sky-300 border border-sky-400/30"
                        : "text-slate-300 hover:bg-white/5 border border-transparent"
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" aria-hidden="true" />
                    <span className="truncate">{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </aside>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 lg:pl-64 flex flex-col min-w-0 min-h-screen">
        {/* Top Application Header */}
        <header className="h-16 sticky top-0 z-30 border-b border-white/10 bg-[#090D16]/90 backdrop-blur-md px-4 sm:px-8 flex items-center justify-between gap-6">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              aria-label="Open navigation menu"
              onClick={() => setMobileMenuOpen(true)}
              className="lg:hidden p-2 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-lg text-slate-300 hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-2 text-sm text-slate-400 truncate">
              <button
                type="button"
                onClick={() => handleSelect("dashboard")}
                className="hidden sm:inline text-slate-400 hover:text-slate-100 transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 rounded"
              >
                CelebA AI
              </button>
              <span className="hidden sm:inline text-slate-600" aria-hidden="true">
                /
              </span>
              <h1 className="text-slate-100 font-semibold truncate">{currentNav.label}</h1>
            </div>
          </div>

          <nav
            aria-label="Header quick links"
            className="hidden md:flex items-center gap-6 text-xs sm:text-sm font-medium"
          >
            {HEADER_QUICK_LINKS.map((link) => {
              const isActive = activePage === link.id;
              return (
                <button
                  key={link.id}
                  type="button"
                  onClick={() => handleSelect(link.id)}
                  className={`py-1 border-b-2 transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                    isActive
                      ? "border-sky-400 text-sky-300"
                      : "border-transparent text-slate-400 hover:text-slate-100 hover:border-white/25"
                  }`}
                >
                  {link.label}
                </button>
              );
            })}
          </nav>

          <div className="flex items-center gap-2.5 shrink-0">
            {onOpenSearch && (
              <button
                type="button"
                onClick={onOpenSearch}
                aria-label="Open quick search (Ctrl+K)"
                className="px-3 py-2 min-h-[40px] rounded-lg border border-white/15 bg-white/5 hover:bg-white/10 text-slate-300 hover:text-slate-100 text-xs font-medium inline-flex items-center gap-2 whitespace-nowrap shrink-0 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                <Search className="w-3.5 h-3.5 text-sky-400" aria-hidden="true" />
                <span className="hidden sm:inline">Search</span>
                <kbd className="hidden xl:inline px-1.5 py-0.5 text-[10px] font-mono rounded bg-slate-900 border border-white/10 text-slate-400">
                  ⌘K
                </kbd>
              </button>
            )}

            <button
              type="button"
              onClick={() => handleSelect("image-analysis")}
              className="px-4 py-2 min-h-[40px] rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold text-xs sm:text-sm transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-300"
            >
              Analyze Face Image
            </button>
          </div>
        </header>

        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-[1360px] w-full mx-auto space-y-6">
          {children}
        </main>

        {/* Application Footer */}
        <footer className="mt-auto border-t border-white/10 bg-[#0C1220]/70">
          <div className="max-w-[1360px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col gap-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="text-sm font-semibold text-slate-100">
                  CelebA Face Intelligence Suite
                </div>
                <p className="text-xs text-slate-400">
                  Turning Visual Data into Intelligent Insights · Machine Learning · Deep Learning ·
                  Generative AI
                </p>
              </div>

              <nav
                aria-label="Footer navigation"
                className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-300"
              >
                {FOOTER_LINKS.map((link) => (
                  <button
                    key={link.id}
                    type="button"
                    onClick={() => handleSelect(link.id)}
                    className="hover:text-sky-300 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 rounded"
                  >
                    {link.label}
                  </button>
                ))}
              </nav>
            </div>

            <div className="pt-4 border-t border-white/10 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs text-slate-400">
              <span>Developed by Alok Agarwal · Python, PyTorch, CNN, BiLSTM, ViT, XGBoost, Gemini AI & MySQL</span>
              <span>In-Memory Processing · Zero Image Retention</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
