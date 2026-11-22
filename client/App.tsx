import { useCallback, useEffect, useState } from "react";
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

export default function App() {
  const [activePage, setActivePage] = useState<PageId>("dashboard");
  const [benchmarks, setBenchmarks] = useState<BenchmarkSuitesResponse | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [analytics, setAnalytics] = useState<DatasetAnalyticsResponse | null>(null);
  const [health, setHealth] = useState<SystemHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
    <Layout activePage={activePage} onNavigate={setActivePage}>
      {activePage === "dashboard" && (
        <DashboardPage
          benchmarks={benchmarks}
          analytics={analytics}
          health={health}
          loading={loading}
          error={error}
          onNavigate={setActivePage}
        />
      )}
      {activePage === "image-analysis" && <ImageAnalysisPage />}
      {activePage === "camera-analysis" && <CameraAnalysisPage />}
      {activePage === "model-comparison" && (
        <ModelComparisonPage benchmarks={benchmarks} models={models} />
      )}
      {activePage === "dataset-analytics" && <DatasetAnalyticsPage analytics={analytics} />}
      {activePage === "performance-reports" && (
        <PerformanceReportsPage benchmarks={benchmarks} analytics={analytics} />
      )}
      {activePage === "ai-chat" && <AiChatPage />}
      {activePage === "system-health" && (
        <SystemHealthPage health={health} onRefresh={() => void loadAll()} />
      )}
    </Layout>
  );
}
