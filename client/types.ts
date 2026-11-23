export type PageId =
  | "dashboard"
  | "image-analysis"
  | "camera-analysis"
  | "model-comparison"
  | "dataset-analytics"
  | "performance-reports"
  | "ai-chat"
  | "system-health";

export interface BenchmarkRow {
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

export interface ModelRegistryEntry {
  id: string;
  name: string;
  strategy: string;
  test_loss: number;
  accuracy: string;
  precision: number;
  recall: number;
  f1_score: number;
  status: string;
}

export interface CheckpointInfo {
  architecture: string;
  filename: string;
  exists: boolean;
  sizeBytes: number | null;
  sizeMb: string | null;
}

export interface BenchmarkSuitesResponse {
  realImageProduction: {
    sourceFile: string;
    datasetType: "real_image_holdout";
    sampleCount: number;
    targetAttribute: string;
    rows: BenchmarkRow[];
  };
  latestTrainingRun: {
    sourceFile: string;
    datasetType: "real_image_training_run";
    timestamp: string;
    elapsedSeconds: number;
    trainSamples: number;
    testSamples: number;
    optimalEnsembleWeights: Record<string, number>;
    rows: BenchmarkRow[];
  };
  ciSmokeTestSynthetic: {
    sourceFile: string;
    datasetType: "synthetic_smoke_test";
    sampleCount: number;
    note: string;
    rows: BenchmarkRow[];
  };
  crossValidation: {
    sourceFile: string;
    folds: { fold: number; valAccuracy: number }[];
    meanAccuracy: number;
  };
}

export interface ModelCardDoc {
  id: string;
  title: string;
  filename: string;
  content: string;
}

export interface ModelsResponse {
  registry: {
    project: string;
    version: string;
    target_attribute: string;
    identity_isolation_strategy: string;
    evaluated_models: ModelRegistryEntry[];
  };
  checkpoints: CheckpointInfo[];
  modelCards: ModelCardDoc[];
  reproducibilityConfigYaml: string;
  mlBackend: {
    url: string;
    online: boolean;
    verifiedCheckpoints: string[];
    error: string | null;
  };
}

export interface DatasetAnalyticsResponse {
  manifest: {
    dataset_name: string;
    version: string;
    total_images: number;
    total_unique_identities: number;
    total_binary_attributes: number;
    image_format: string;
    resolution: number[];
    canonical_target_size: number[];
    target_attribute: string;
    evaluation_protocol: Record<string, unknown>;
  };
  experimentMetadata: {
    experiment_id: string;
    random_seed: number;
    target_attribute: string;
    identity_disjoint_method: string;
    split_statistics: {
      train_identities: number;
      test_identities: number;
      identity_leakage_count: number;
    };
    optimal_weights: number[];
    training_duration_seconds: number;
    verified_accuracy_leader: string;
  };
  splitManifestSummary: {
    totalRows: number;
    splits: { split: string; images: number; uniqueIdentities: number }[];
    identityOverlapCount: number;
  };
  identityFileStats: {
    totalMappings: number;
    uniqueIdentities: number;
    celebritiesCatalogCount: number;
  };
  galleryStats: {
    totalSamples: number;
    uniqueClusters: number;
    attributeCounts: { attribute: string; count: number; percentage: number }[];
    summaryRates: {
      smiling: number;
      young: number;
      male: number;
      eyeglasses: number;
    };
  };
  figures: {
    filename: string;
    title: string;
    category: "dataset" | "performance" | "confusion_matrix";
    url: string;
  }[];
}

export interface GallerySample {
  id: number;
  filename: string;
  thumbnail: string;
  cluster_id: number;
  smiling: boolean;
  eyeglasses: boolean;
  male: boolean;
  young: boolean;
  attributes: string[];
}

export interface GeminiAnalysisResult {
  status: string;
  source: string;
  model: string;
  retained_on_disk: boolean;
  privacy_notice: string;
  latency_ms: number;
  analysis: {
    expression: {
      is_smiling: boolean;
      confidence: number;
      primary_emotion: string;
    };
    visual_attributes: {
      hair_color: string;
      wearing_eyeglasses: boolean;
      facial_hair: boolean;
      headwear_or_accessories: string[];
      notable_visual_cues: string[];
    };
    image_quality: {
      lighting: string;
      framing: string;
      occlusion_level: string;
    };
    reasoning_summary: string;
  };
}

export interface MlPredictionResult {
  source: string;
  architecture: string;
  use_tta: boolean;
  tta_views_evaluated: number;
  probability: number;
  prediction_uncertainty_std: number;
  prediction: number;
  confidence: number;
  checkpoint_loaded: string;
  checkpoint_verified: boolean;
  is_ground_truth_lookup: boolean;
  latency_ms?: number;
  retained_on_disk?: boolean;
}

export interface MlConsensusModelEntry {
  architecture: string;
  probability: number;
  prediction: number;
  confidence: number;
  uncertainty_std: number;
  slsqp_raw_weight: number;
  normalized_weight: number;
  checkpoint_loaded: string;
}

export interface MlConsensusResult {
  source: string;
  use_tta: boolean;
  weighting_strategy: string;
  models_evaluated: number;
  weighted_probability: number;
  consensus_prediction: number;
  agreement_ratio: number;
  per_model: MlConsensusModelEntry[];
  latency_ms?: number;
  retained_on_disk?: boolean;
}

export interface SystemHealthResponse {
  status: "healthy" | "degraded";
  service: string;
  nodeVersion: string;
  environment: string;
  port: number;
  uptimeSeconds: number;
  timestamp: string;
  services: {
    geminiApi: {
      configured: boolean;
      defaultModel: string;
      keyStorage: "server_env_only";
    };
    pythonMlBackend: {
      url: string;
      reachable: boolean;
      engineReady: boolean;
      verifiedCheckpoints: string[];
      latencyMs: number | null;
      error: string | null;
    };
    mysqlDatabase: {
      configured: boolean;
      host: string | null;
      port: number;
      database: string | null;
      reachable: boolean;
      error: string | null;
    };
  };
  checkpoints: CheckpointInfo[];
  privacyAndSecurity: {
    imageRetention: string;
    identityInferenceBlocked: boolean;
    sensitiveAttributeInferenceBlocked: boolean;
    magicByteValidation: boolean;
    maxUploadMb: number;
    rateLimiting: {
      readWindowLimit: string;
      inferenceWindowLimit: string;
    };
  };
}

export interface ApiEndpointDoc {
  method: "GET" | "POST";
  path: string;
  summary: string;
  authOrPolicy: string;
}
