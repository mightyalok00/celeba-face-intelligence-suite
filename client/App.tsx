import { useCallback, useEffect, useState } from "react";
import { CommandPalette } from "./components/CommandPalette";
import { Layout } from "./components/Layout";
import { AiChatPage } from "./pages/AiChatPage";
import { CameraAnalysisPage } from "./pages/CameraAnalysisPage";
import { DashboardPage } from "./pages/DashboardPage";
import { DatasetAnalyticsPage } from "./pages/DatasetAnalyticsPage";
import { ImageAnalysisPage } from "./pages/ImageAnalysisPage";
import { ModelComparisonPage } from "./pages/ModelComparisonPage";
import { PerformanceReportsPage } from "./pages/PerformanceReportsPage";
import { SystemHealthPage } from "./pages/SystemHealthPage";
import type {
  BenchmarkSuitesResponse,
  DatasetAnalyticsResponse,
  ModelsResponse,
  PageId,
  SystemHealthResponse,
} from "./types";

const PAGE_TO_PATH: Record<PageId, string> = {
  dashboard: "/",
  "image-analysis": "/analyze",
  "camera-analysis": "/camera",
  "model-comparison": "/models",
  "dataset-analytics": "/dataset",
  "performance-reports": "/reports",
  "ai-chat": "/assistant",
  "system-health": "/system",
};

const PATH_TO_PAGE: Record<string, PageId> = {
  "/": "dashboard",
  "/analyze": "image-analysis",
  "/camera": "camera-analysis",
  "/models": "model-comparison",
  "/dataset": "dataset-analytics",
  "/reports": "performance-reports",
  "/assistant": "ai-chat",
  "/system": "system-health",
};

function resolveInitialPage(): PageId {
  const normalized = window.location.pathname.replace(/\/+$/, "") || "/";
  return PATH_TO_PAGE[normalized] ?? "dashboard";
}

export default function App() {
  const [activePage, setActivePage] = useState<PageId>(() => resolveInitialPage());
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [benchmarks, setBenchmarks] = useState<BenchmarkSuitesResponse | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [analytics, setAnalytics] = useState<DatasetAnalyticsResponse | null>(null);
  const [health, setHealth] = useState<SystemHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pendingAnalysisImage, setPendingAnalysisImage] = useState<string | null>(null);

  const navigateTo = useCallback((page: PageId) => {
    setActivePage(page);
    const targetPath = PAGE_TO_PATH[page] ?? "/";
    if (window.location.pathname !== targetPath) {
      window.history.pushState({ page }, "", targetPath);
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const handleAnalyzeGalleryImage = useCallback(
    (imageUrl: string) => {
      setPendingAnalysisImage(imageUrl);
      navigateTo("image-analysis");
    },
    [navigateTo]
  );

  useEffect(() => {
    const onPopState = () => {
      setActivePage(resolveInitialPage());
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener("popstate", onPopState);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("popstate", onPopState);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [bResp, mResp, aResp, hResp] = await Promise.all([
        fetch("/api/benchmarks"),
        fetch("/api/models"),
        fetch("/api/dataset/analytics"),
        fetch("/api/system/health"),
      ]);

      if (!bResp.ok || !mResp.ok || !aResp.ok || !hResp.ok) {
        throw new Error("One or more studio backend endpoints returned an error.");
      }

      const [bData, mData, aData, hData] = await Promise.all([
        bResp.json() as Promise<BenchmarkSuitesResponse>,
        mResp.json() as Promise<ModelsResponse>,
        aResp.json() as Promise<DatasetAnalyticsResponse>,
        hResp.json() as Promise<SystemHealthResponse>,
      ]);

      setBenchmarks(bData);
      setModels(mData);
      setAnalytics(aData);
      setHealth(hData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load CelebA Studio data.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  return (
    <>
      <Layout
        activePage={activePage}
        onNavigate={navigateTo}
        onOpenSearch={() => setPaletteOpen(true)}
      >
        {activePage === "dashboard" && (
          <DashboardPage
            benchmarks={benchmarks}
            analytics={analytics}
            health={health}
            loading={loading}
            error={error}
            onNavigate={navigateTo}
          />
        )}
        {activePage === "image-analysis" && (
          <ImageAnalysisPage
            health={health}
            models={models}
            initialImageUrl={pendingAnalysisImage}
            onConsumeInitialImage={() => setPendingAnalysisImage(null)}
            onNavigate={navigateTo}
            onRefreshHealth={() => void loadAll()}
          />
        )}
        {activePage === "camera-analysis" && (
          <CameraAnalysisPage health={health} models={models} onNavigate={navigateTo} />
        )}
        {activePage === "model-comparison" && (
          <ModelComparisonPage benchmarks={benchmarks} models={models} />
        )}
        {activePage === "dataset-analytics" && (
          <DatasetAnalyticsPage
            analytics={analytics}
            onAnalyzeGalleryImage={handleAnalyzeGalleryImage}
          />
        )}
        {activePage === "performance-reports" && (
          <PerformanceReportsPage benchmarks={benchmarks} analytics={analytics} />
        )}
        {activePage === "ai-chat" && <AiChatPage />}
        {activePage === "system-health" && (
          <SystemHealthPage health={health} onRefresh={() => void loadAll()} />
        )}
      </Layout>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onNavigate={navigateTo}
        benchmarks={benchmarks}
        analytics={analytics}
      />
    </>
  );
}
