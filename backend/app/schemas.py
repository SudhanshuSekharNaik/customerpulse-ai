"""Pydantic request and response schemas for CustomerPulse AI FastAPI backend."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


# --- Auth Schemas ---
class Token(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    access_token: str
    token_type: str
    role: str
    username: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True


# --- Customer Schemas ---
class EventOut(BaseModel):
    event_id: str
    event_type: str
    item_id: Optional[str] = None
    category_id: Optional[str] = None
    timestamp: datetime
    revenue: float

    class Config:
        from_attributes = True


class CustomerFeatureOut(BaseModel):
    recency_days: float
    frequency_7d: int
    frequency_30d: int
    frequency_90d: int
    monetary_total: float
    monetary_30d: float
    aov: float
    views_7d: int
    views_30d: int
    views_90d: int
    carts_7d: int
    carts_30d: int
    carts_90d: int
    transactions_7d: int
    transactions_30d: int
    transactions_90d: int
    cart_to_view_ratio: float
    conversion_rate: float
    velocity_7d_30d: float
    engagement_velocity: float
    trend_slope: float
    top_category: Optional[str] = None
    category_entropy: float
    unique_items_viewed: int
    is_cold_start: bool

    class Config:
        from_attributes = True


class CustomerSummary(BaseModel):
    customer_id: str
    visitor_id: Optional[str] = None
    email: Optional[str] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_events: int
    total_revenue: float
    total_orders: int
    is_cold_start: bool
    current_state: Optional[str] = None
    segment_label: Optional[str] = None
    churn_probability: Optional[float] = None
    opportunity_score: Optional[float] = None

    class Config:
        from_attributes = True


class CustomerDetail(BaseModel):
    customer_id: str
    visitor_id: Optional[str] = None
    email: Optional[str] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_events: int
    total_revenue: float
    total_orders: int
    is_cold_start: bool
    features: Optional[CustomerFeatureOut] = None
    current_state: Optional[str] = None
    previous_state: Optional[str] = None
    segment_label: Optional[str] = None
    segment_id: Optional[int] = None
    churn_prediction: Optional[Dict[str, Any]] = None
    next_event_prediction: Optional[Dict[str, Any]] = None
    uplift_estimate: Optional[Dict[str, Any]] = None
    top_recommendation: Optional[Dict[str, Any]] = None
    recent_events: List[EventOut] = []

    class Config:
        from_attributes = True


# --- Segment Schemas ---
class SegmentSummary(BaseModel):
    segment_id: int
    segment_label: str
    customer_count: int
    percentage: float
    avg_revenue: float
    avg_recency_days: float
    avg_frequency_30d: float
    avg_cart_ratio: float
    top_category: str
    silhouette_score: float


# --- State Schemas ---
class StateSummary(BaseModel):
    state: str
    customer_count: int
    percentage: float
    avg_recency_days: float
    avg_churn_risk: float
    avg_revenue: float


class StateTransitionMatrix(BaseModel):
    states: List[str]
    matrix: List[List[float]]


# --- Prediction Schemas ---
class PredictionOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction_id: str
    customer_id: str
    model_type: str
    predicted_class: str
    predicted_probability: float
    decision_threshold: float
    shap_values: Dict[str, float]
    confidence_interval_low: Optional[float] = None
    confidence_interval_high: Optional[float] = None
    created_at: datetime


# --- Uplift Schemas ---
class UpliftDecileOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    decile: int
    population_fraction: float
    treated_conversion_rate: float
    control_conversion_rate: float
    estimated_uplift: float
    cumulative_qini: float


class UpliftOverviewOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_used: str
    randomization_assumption: str
    qini_score: float
    auuc: float
    baseline_two_model_qini: float
    x_learner_qini: float
    deciles: List[UpliftDecileOut]


# --- Recommendation Schemas ---
class RecommendationOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    recommendation_id: str
    customer_id: str
    action_type: str
    rank: int
    score: float
    what_text: str
    why_text: str
    evidence: Dict[str, Any]
    expected_impact: float
    confidence_level: str
    status: str
    created_at: datetime


# --- Model Registry Schemas ---
class ModelRunOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    run_id: str
    model_name: str
    model_version: str
    model_type: str
    dataset_hash: str
    row_count: int
    train_timestamp: datetime
    status: str
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    f1_score: Optional[float] = None
    qini_score: Optional[float] = None
    silhouette_score: Optional[float] = None


# --- AI Agent Schemas ---
class AgentQueryRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prompt: str
    session_id: Optional[str] = "default_session"


class AgentToolCallOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Any
    cached: bool


class AgentResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    run_id: str
    observed_data: str
    model_prediction: str
    model_estimate: str
    recommendation: str
    tool_calls: List[AgentToolCallOut] = []
    duration_seconds: float
    raw_response: str
