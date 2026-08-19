import {
  CustomerSummary,
  CustomerDetail,
  SegmentSummary,
  StateSummary,
  StateTransitionMatrix,
  BehaviorChange,
  UpliftOverview,
  Recommendation,
  OfflineBacktestReport,
  ModelRun,
  AgentResponse,
  TrafficForecastReport,
  CustomerNextAction,
  DatasetMeta,
} from "../types";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    return await res.json();
  } catch (err) {
    console.error(`API fetch error on ${url}:`, err);
    throw err;
  }
}

export const api = {
  getHealth: () => fetchJson<{ status: string; total_customers: number; trained_models: number }>(`${API_BASE}/health`),

  getDatasetMeta: () => fetchJson<DatasetMeta>(`${API_BASE}/analytics/dataset-meta`),

  getAnalyticsOverview: () =>
    fetchJson<{
      total_customers: number;
      total_events: number;
      total_revenue: number;
      at_risk_customers_count: number;
      at_risk_percentage: number;
      revenue_at_risk: number;
      addressable_portfolio_uplift: number;
    }>(`${API_BASE}/analytics/overview`),

  getCustomers: (params?: { limit?: number; offset?: number; state?: string; segment_id?: number; search?: string }) => {
    const query = new URLSearchParams();
    if (params?.limit) query.set("limit", params.limit.toString());
    if (params?.offset) query.set("offset", params.offset.toString());
    if (params?.state) query.set("state", params.state);
    if (params?.segment_id !== undefined) query.set("segment_id", params.segment_id.toString());
    if (params?.search) query.set("search", params.search);
    return fetchJson<{ total: number; limit: number; offset: number; items: CustomerSummary[] }>(
      `${API_BASE}/customers?${query.toString()}`
    );
  },

  getCustomerDetail: (customerId: string) => fetchJson<CustomerDetail>(`${API_BASE}/customers/${customerId}`),

  getSegments: () => fetchJson<SegmentSummary[]>(`${API_BASE}/segments`),

  getClusterScatter: () =>
    fetchJson<{
      selected_k: number;
      silhouette_score: number;
      davies_bouldin_index: number;
      calinski_harabasz_score: number;
      k_candidates: Array<{ k: number; silhouette: number; davies_bouldin: number; calinski: number; selected: boolean }>;
      pca_variance_explained: number[];
      outlier_policy: { criteria: string; p99_spend_threshold: number };
      centroids: Array<{ cluster_id: number; x: number; y: number; segment_label: string }>;
      scatter_points: Array<{
        customer_id: string;
        cluster_id: number;
        segment_label: string;
        x: number;
        y: number;
        spend: number;
        recency_days: number;
        frequency_30d: number;
        total_orders?: number;
        top_category?: string;
      }>;
    }>(`${API_BASE}/segments/scatter`),

  getDataHealth: () =>
    fetchJson<{
      status: string;
      data_quality_score: number;
      total_rows_ingested: number;
      total_customers: number;
      feature_records: number;
      prediction_coverage_pct: number;
      segment_coverage_pct: number;
      state_coverage_pct: number;
      missing_ids_count: number;
      missing_timestamps_count: number;
      duplicate_rows_count: number;
      models: Record<string, string>;
      last_synced_at: string;
    }>(`${API_BASE}/analytics/data-health`),

  getStates: () => fetchJson<StateSummary[]>(`${API_BASE}/behavior/states`),

  getTransitionMatrix: () => fetchJson<StateTransitionMatrix & { counts_matrix?: number[][]; total_transitions_observed?: number }>(`${API_BASE}/behavior/transitions`),

  getBehaviorChanges: (limit: number = 50, severity?: string) => {
    const query = new URLSearchParams({ limit: limit.toString() });
    if (severity) query.set("severity", severity);
    return fetchJson<BehaviorChange[]>(`${API_BASE}/behavior/changes?${query.toString()}`);
  },

  getChurnOverview: () =>
    fetchJson<{
      total_scored_customers: number;
      high_risk_customers_count: number;
      high_risk_percentage: number;
      average_churn_probability: number;
      optimal_decision_threshold: number;
      primary_metric_pr_auc: number;
      secondary_metric_roc_auc: number;
      f1_score: number;
      cost_ratio_assumption: string;
      confusion_matrix: number[][];
    }>(`${API_BASE}/predictions/churn/overview`),

  getTopChurnRisk: (limit: number = 50, search?: string) => {
    const query = new URLSearchParams({ limit: limit.toString() });
    if (search) query.set("search", search);
    return fetchJson<any[]>(`${API_BASE}/predictions/churn/top-risk?${query.toString()}`);
  },

  getUpliftOverview: () => fetchJson<UpliftOverview>(`${API_BASE}/uplift/overview`),

  getTopPersuadables: (limit: number = 50) => fetchJson<any[]>(`${API_BASE}/uplift/top-persuadables?limit=${limit}`),

  getRecommendations: (params?: { limit?: number; action_type?: string; confidence?: string }) => {
    const query = new URLSearchParams();
    if (params?.limit) query.set("limit", params.limit.toString());
    if (params?.action_type) query.set("action_type", params.action_type);
    if (params?.confidence) query.set("confidence", params.confidence);
    return fetchJson<Recommendation[]>(`${API_BASE}/recommendations?${query.toString()}`);
  },

  getOfflineBacktest: () => fetchJson<OfflineBacktestReport>(`${API_BASE}/recommendations/offline-backtest`),

  getModelRuns: () => fetchJson<ModelRun[]>(`${API_BASE}/models`),

  queryAgent: (prompt: string, sessionId: string = "web_session") =>
    fetchJson<AgentResponse>(`${API_BASE}/agent/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, session_id: sessionId }),
    }),

  // --- Universal Data Mode API Methods ---
  uploadCsv: async (file: File): Promise<any> => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE}/universal/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `Upload failed with status ${res.status}`);
    }
    return await res.json();
  },

  confirmAndAnalyze: (datasetId: string, columnMapping: Record<string, string>) =>
    fetchJson<{ dataset_id: string; filename: string; report: any; status: string }>(
      `${API_BASE}/universal/confirm-and-analyze`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: datasetId, column_mapping: columnMapping }),
      }
    ),

  getDatasets: () => fetchJson<any[]>(`${API_BASE}/universal/datasets`),

  getDatasetDetail: (datasetId: string) => fetchJson<any>(`${API_BASE}/universal/dataset/${datasetId}`),

  switchMode: (mode: string, datasetId?: string) =>
    fetchJson<{ active_mode: string; message?: string; dataset_id?: string; filename?: string; mode_label?: string }>(
      `${API_BASE}/universal/switch-mode`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, dataset_id: datasetId }),
      }
    ),

  getPresets: () => fetchJson<any[]>(`${API_BASE}/universal/presets`),

  getPresetDownloadUrl: (presetId: string) => `${API_BASE}/universal/presets/${presetId}/download`,

  getPdfReportUrl: () => `${API_BASE}/reports/pdf`,

  getActiveContext: () => fetchJson<any>(`${API_BASE}/universal/active-context`),

  // --- E-Commerce & Traffic Forecast API Methods ---
  getTrafficForecast: () => fetchJson<TrafficForecastReport>(`${API_BASE}/analytics/traffic-forecast`),

  getCustomerNextAction: (customerId: string) =>
    fetchJson<CustomerNextAction>(`${API_BASE}/analytics/customer-next-action/${customerId}`),
};
