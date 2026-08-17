export interface CustomerSummary {
  customer_id: string;
  visitor_id?: string;
  email?: string;
  first_seen?: string;
  last_seen?: string;
  total_events: number;
  total_revenue: number;
  total_orders: number;
  is_cold_start: boolean;
  current_state: string;
  segment_label: string;
  churn_probability?: number;
  opportunity_score?: number;
}

export interface ExecutiveOverview {
  total_customers: number;
  total_events: number;
  total_orders?: number;
  total_products?: number;
  total_recommendations?: number;
  total_revenue: number;
  total_revenue_inr?: number;
  at_risk_customers_count: number;
  at_risk_percentage: number;
  revenue_at_risk: number;
  revenue_at_risk_inr?: number;
  addressable_portfolio_uplift: number;
  portfolio_actionable_uplift_inr?: number;
}

export interface CustomerFeature {
  recency_days: number;
  frequency_7d: number;
  frequency_30d: number;
  frequency_90d: number;
  monetary_total: number;
  monetary_30d: number;
  aov: number;
  views_7d: number;
  views_30d: number;
  views_90d: number;
  carts_7d: number;
  carts_30d: number;
  carts_90d: number;
  transactions_7d: number;
  transactions_30d: number;
  transactions_90d: number;
  cart_to_view_ratio: number;
  conversion_rate: number;
  velocity_7d_30d: number;
  engagement_velocity: number;
  trend_slope: number;
  top_category?: string;
  category_entropy: number;
  unique_items_viewed: number;
  is_cold_start: boolean;
}

export interface CustomerDetail {
  customer_id: string;
  visitor_id?: string;
  email?: string;
  first_seen?: string;
  last_seen?: string;
  total_events: number;
  total_revenue: number;
  total_orders: number;
  is_cold_start: boolean;
  features?: CustomerFeature;
  current_state: string;
  previous_state?: string;
  segment_label: string;
  segment_id?: number;
  churn_prediction?: {
    predicted_class: string;
    predicted_probability?: number | null;
    decision_threshold: number;
    shap_values: Record<string, number>;
    is_cold_start?: boolean;
    status_text?: string;
    ci_low?: number | null;
    ci_high?: number | null;
  };
  next_event_prediction?: {
    predicted_event: string;
    predicted_probability: number;
  };
  uplift_estimate?: {
    estimated_uplift?: number | null;
    decile?: number | null;
    ci_low?: number | null;
    ci_high?: number | null;
    model_used?: string;
    label?: string;
    is_available?: boolean;
    status_text?: string;
  };
  top_recommendation?: {
    action_type: string;
    what: string;
    why: string;
    evidence: Record<string, any>;
    expected_impact: number;
    confidence: string;
  };
  recent_events: Array<{
    event_id: string;
    event_type: string;
    item_id?: string;
    category_id?: string;
    timestamp: string;
    revenue: number;
  }>;
}

export interface SegmentSummary {
  segment_id: number;
  segment_label: string;
  customer_count: number;
  percentage: number;
  avg_revenue: number;
  avg_recency_days: number;
  avg_frequency_30d: number;
  avg_cart_ratio: number;
  top_category: string;
  silhouette_score: number;
}

export interface StateSummary {
  state: string;
  customer_count: number;
  percentage: number;
  avg_recency_days: number;
  avg_churn_risk: number;
  avg_revenue: number;
}

export interface StateTransitionMatrix {
  states: string[];
  matrix: number[][];
}

export interface BehaviorChange {
  change_id: string;
  customer_id: string;
  metric: string;
  baseline_value: number;
  current_value: number;
  pct_change: number;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  detection_method: string;
  detected_at: string;
}

export interface UpliftOverview {
  model_used: string;
  randomization_assumption: string;
  qini_score: number;
  auuc: number;
  baseline_two_model_qini: number;
  x_learner_qini: number;
  deciles: Array<{
    decile: number;
    population_fraction: number;
    treated_conversion_rate: number;
    control_conversion_rate: number;
    estimated_uplift: number;
    cumulative_qini: number;
  }>;
}

export interface Recommendation {
  recommendation_id: string;
  customer_id: string;
  email?: string;
  action_type: string;
  rank: number;
  score: number;
  what_text: string;
  why_text: string;
  evidence: Record<string, any>;
  expected_impact: number;
  confidence_level: string;
  status: string;
  created_at: string;
}

export interface OfflineBacktestReport {
  status: string;
  total_evaluated_customers: number;
  total_evaluated_accounts?: number;
  action_diversity_index: number;
  is_degenerate_policy: boolean;
  top_action_share: number;
  average_expected_impact_inr: number;
  total_portfolio_uplift_inr: number;
  offline_backtest_conclusion: string;
}

export interface ModelRun {
  run_id: string;
  model_name: string;
  model_version: string;
  model_type: string;
  dataset_hash: string;
  row_count: number;
  train_timestamp: string;
  status: string;
  pr_auc?: number;
  roc_auc?: number;
  f1_score?: number;
  qini_score?: number;
  silhouette_score?: number;
}

export interface AgentToolCall {
  tool_name: string;
  tool_args: Record<string, any>;
  tool_result: any;
  cached: boolean;
}

export interface AgentResponse {
  run_id: string;
  observed_data: string;
  model_prediction: string;
  model_estimate: string;
  recommendation: string;
  tool_calls: AgentToolCall[];
  duration_seconds: number;
  raw_response: string;
}

// --- Universal Data Mode Types ---
export interface ColumnProfile {
  column_name: string;
  detected_type: string;
  raw_dtype: string;
  unique_values: number;
  cardinality_ratio: number;
  null_count: number;
  null_percentage: number;
  sample_values: string[];
  min?: number;
  max?: number;
  mean?: number;
}

export interface DatasetProfile {
  filename: string;
  file_size_bytes: number;
  encoding: string;
  delimiter: string;
  total_rows: number;
  total_columns: number;
  missing_rate_pct: number;
  duplicate_rows_count: number;
  duplicate_rate_pct: number;
  numerical_columns_count: number;
  categorical_columns_count: number;
  datetime_columns_count: number;
  id_columns_count: number;
  columns: ColumnProfile[];
  status: string;
}

export interface DetectedColumn {
  column_name: string;
  detected_semantic_type: string;
  confidence: number;
  options: string[];
  raw_dtype: string;
  unique_values: number;
  sample_values: string[];
}

export interface CapabilityItem {
  name: string;
  description?: string;
  reason?: string;
}

export interface DatasetCapabilities {
  dataset_mode: string;
  domain: string;
  mode_label: string;
  is_customer_intelligence_supported: boolean;
  capabilities?: Record<string, { available: boolean; reason: string }>;
  enabled_capabilities: CapabilityItem[];
  disabled_capabilities: CapabilityItem[];
  limitations: string[];
  schema_summary: Record<string, boolean>;
}

export interface UploadProfileResponse {
  dataset_id: string;
  filename: string;
  profile: DatasetProfile;
  detected_columns: DetectedColumn[];
  initial_mapping: Record<string, string>;
  capabilities: DatasetCapabilities;
  status: string;
}

export interface UniversalReport {
  dataset_name: string;
  dataset_mode: string;
  domain: string;
  mode_label: string;
  total_rows: number;
  total_columns: number;
  execution_duration_seconds: number;
  capabilities: DatasetCapabilities;
  model_results: Array<{
    model_name: string;
    model_type: string;
    metric_name: string;
    metric_value: number;
    roc_auc?: number;
    status: string;
    details: string;
  }>;
  key_insights: string[];
  customer_summaries: CustomerSummary[];
  segment_summaries: SegmentSummary[];
  generic_analytics?: {
    numerical_attributes: string[];
    correlation_matrix: Record<string, Record<string, number>>;
    detected_outliers_count: number;
    outlier_percentage: number;
    summary_statistics: Record<string, any>;
  };
  limitations: string[];
  status: string;
}

export interface ActiveDatasetContext {
  active_mode: "DEMO" | "UPLOAD";
  dataset_id?: string;
  dataset_name: string;
  dataset_mode?: string;
  domain?: string;
  mode_label: string;
  row_count?: number;
  col_count?: number;
  is_customer_intelligence: boolean;
  report?: UniversalReport;
}

// --- E-Commerce & Traffic Forecast Types ---
export interface HourlyTrafficPoint {
  hour: number;
  label: string;
  event_count: number;
  traffic_index: number;
  is_peak_window: boolean;
  activity_type: string;
}

export interface WeeklyTrafficPoint {
  day_index: number;
  day_name: string;
  event_count: number;
  demand_multiplier: number;
  is_weekend: boolean;
}

export interface CategorySurge {
  category_id: string;
  category_name: string;
  event_count: number;
  demand_share_pct: number;
  surge_velocity: string;
}

export interface TrafficForecastReport {
  total_analyzed_events: number;
  hourly_traffic_curve: HourlyTrafficPoint[];
  weekly_traffic_curve: WeeklyTrafficPoint[];
  next_peak_window: {
    window_description: string;
    urgency: string;
    hours_until_peak: number;
    projected_traffic_lift_pct: number;
    recommended_action: string;
  };
  ecommerce_funnel: {
    views: number;
    carts: number;
    purchases: number;
    has_funnel_events?: boolean;
    funnel_note?: string;
    cart_abandonment_rate_pct?: number | null;
    cart_to_view_ratio_pct?: number | null;
    checkout_conversion_rate_pct?: number | null;
    recovered_cart_revenue_potential_inr?: number;
  };
  category_surges: CategorySurge[];
  festive_sale_multiplier: {
    event_name: string;
    traffic_multiplier: number;
    projected_conversion_uplift_pct: string;
    peak_categories: string[];
  };
}

export interface CustomerNextAction {
  customer_id: string;
  predicted_next_action: string;
  expected_visit_timing: string;
  probability: number;
  cart_recovery_potential: string;
  discount_voucher_recommendation: string;
  category_affinity: string;
}


