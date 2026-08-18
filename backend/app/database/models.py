"""SQLAlchemy database models for CustomerPulse AI.
Includes all 15 core entities + User authentication & RBAC.
Compatible with SQLite (local development) and PostgreSQL (production).
"""

from datetime import datetime
import json
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), default="ANALYST", nullable=False)  # ADMIN, ANALYST, VIEWER
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(64), unique=True, index=True, nullable=False)
    visitor_id = Column(String(64), index=True, nullable=True)
    email = Column(String(128), nullable=True)
    first_seen = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, nullable=True)
    total_events = Column(Integer, default=0)
    total_revenue = Column(Float, default=0.0)
    total_orders = Column(Integer, default=0)
    is_cold_start = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    events = relationship("Event", back_populates="customer", cascade="all, delete-orphan")
    features = relationship("CustomerFeature", back_populates="customer", uselist=False, cascade="all, delete-orphan")
    segment = relationship("CustomerSegment", back_populates="customer", uselist=False, cascade="all, delete-orphan")
    state = relationship("CustomerState", back_populates="customer", uselist=False, cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="customer", cascade="all, delete-orphan")
    behavior_changes = relationship("BehaviorChange", back_populates="customer", cascade="all, delete-orphan")
    uplift_predictions = relationship("UpliftPrediction", back_populates="customer", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="customer", cascade="all, delete-orphan")


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(64), unique=True, index=True, nullable=False)
    customer_id = Column(String(64), ForeignKey("customers.customer_id"), index=True, nullable=False)
    visitor_id = Column(String(64), index=True, nullable=True)
    event_type = Column(String(32), index=True, nullable=False)  # view, addtocart, transaction
    item_id = Column(String(64), index=True, nullable=True)
    category_id = Column(String(64), index=True, nullable=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    transaction_id = Column(String(64), nullable=True)
    revenue = Column(Float, default=0.0)

    customer = relationship("Customer", back_populates="events")

    __table_args__ = (
        Index("idx_events_cust_time", "customer_id", "timestamp"),
        Index("idx_events_type_time", "event_type", "timestamp"),
    )


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(64), unique=True, index=True, nullable=False)
    category_id = Column(String(64), index=True, nullable=True)
    name = Column(String(256), nullable=False)
    price = Column(Float, default=0.0)
    total_views = Column(Integer, default=0)
    total_carts = Column(Integer, default=0)
    total_purchases = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class CustomerFeature(Base):
    __tablename__ = "customer_features"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(64), ForeignKey("customers.customer_id"), unique=True, index=True, nullable=False)
    as_of_date = Column(DateTime, default=datetime.utcnow)
    
    # RFM metrics
    recency_days = Column(Float, default=0.0)
    frequency_7d = Column(Integer, default=0)
    frequency_30d = Column(Integer, default=0)
    frequency_90d = Column(Integer, default=0)
    monetary_total = Column(Float, default=0.0)
    monetary_30d = Column(Float, default=0.0)
    aov = Column(Float, default=0.0)

    # Event count breakdowns
    views_7d = Column(Integer, default=0)
    views_30d = Column(Integer, default=0)
    views_90d = Column(Integer, default=0)
    carts_7d = Column(Integer, default=0)
    carts_30d = Column(Integer, default=0)
    carts_90d = Column(Integer, default=0)
    transactions_7d = Column(Integer, default=0)
    transactions_30d = Column(Integer, default=0)
    transactions_90d = Column(Integer, default=0)

    # Funnel and velocity
    cart_to_view_ratio = Column(Float, default=0.0)
    purchase_to_cart_ratio = Column(Float, default=0.0)
    conversion_rate = Column(Float, default=0.0)
    purchase_interval_days = Column(Float, default=0.0)
    velocity_7d_30d = Column(Float, default=1.0)
    engagement_velocity = Column(Float, default=0.0)
    trend_slope = Column(Float, default=0.0)

    # Category preferences
    top_category = Column(String(64), nullable=True)
    category_entropy = Column(Float, default=0.0)
    unique_items_viewed = Column(Integer, default=0)
    is_cold_start = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="features")


class CustomerSegment(Base):
    __tablename__ = "customer_segments"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(64), ForeignKey("customers.customer_id"), unique=True, index=True, nullable=False)
    segment_id = Column(Integer, index=True, nullable=False)
    segment_label = Column(String(128), nullable=False)
    silhouette_score = Column(Float, default=0.0)
    distance_to_centroid = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="segment")


class CustomerState(Base):
    __tablename__ = "customer_states"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(64), ForeignKey("customers.customer_id"), unique=True, index=True, nullable=False)
    current_state = Column(String(32), index=True, nullable=False)  # NEW, EXPLORING, ENGAGED, CONVERTING, LOYAL, DECLINING, AT_RISK, DORMANT, RECOVERING
    previous_state = Column(String(32), nullable=True)
    transition_date = Column(DateTime, default=datetime.utcnow)
    state_duration_days = Column(Integer, default=0)
    state_confidence = Column(Float, default=1.0)
    transition_probability = Column(Float, default=1.0)
    updated_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="state")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String(64), unique=True, index=True, nullable=False)
    customer_id = Column(String(64), ForeignKey("customers.customer_id"), index=True, nullable=False)
    model_type = Column(String(32), index=True, nullable=False)  # churn, next_event
    model_version = Column(String(32), default="v1.0")
    predicted_class = Column(String(64), nullable=False)
    predicted_probability = Column(Float, nullable=True)
    pr_auc_at_eval = Column(Float, nullable=True)
    decision_threshold = Column(Float, default=0.5)
    shap_values_json = Column(Text, nullable=True)  # JSON string of feature importance
    confidence_interval_low = Column(Float, nullable=True)
    confidence_interval_high = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="predictions")

    @property
    def shap_payload(self):
        if self.shap_values_json:
            try:
                return json.loads(self.shap_values_json)
            except Exception:
                return {}
        return {}

    @property
    def shap_values(self):
        payload = self.shap_payload
        if isinstance(payload, dict):
            if "shap_values" in payload and isinstance(payload["shap_values"], dict):
                return payload["shap_values"]
            return payload
        return {}

    @property
    def drivers(self):
        payload = self.shap_payload
        if isinstance(payload, dict) and "drivers" in payload and isinstance(payload["drivers"], list):
            return payload["drivers"]
        
        # Derive dynamically from flat shap_values if legacy record
        shaps = self.shap_values
        if not shaps or not isinstance(shaps, dict):
            return []
        
        derived = []
        for feat, val in sorted(shaps.items(), key=lambda x: abs(float(x[1])), reverse=True):
            f_val = float(val)
            derived.append({
                "feature": str(feat),
                "shap_value": round(f_val, 4),
                "direction": "increases_risk" if f_val > 0 else "decreases_risk",
            })
        return derived

    @property
    def positive_drivers(self):
        payload = self.shap_payload
        if isinstance(payload, dict) and "positive_drivers" in payload and isinstance(payload["positive_drivers"], list):
            return payload["positive_drivers"]
        return [d for d in self.drivers if d["direction"] == "increases_risk"][:3]

    @property
    def protective_drivers(self):
        payload = self.shap_payload
        if isinstance(payload, dict) and "protective_drivers" in payload and isinstance(payload["protective_drivers"], list):
            return payload["protective_drivers"]
        return [d for d in self.drivers if d["direction"] == "decreases_risk"][:3]

    @property
    def top_risk_factor(self):
        payload = self.shap_payload
        if isinstance(payload, dict) and "top_risk_factor" in payload and payload["top_risk_factor"]:
            return payload["top_risk_factor"]
        pos = self.positive_drivers
        if pos:
            return {"feature": pos[0]["feature"], "shap_value": pos[0]["shap_value"]}
        if self.drivers:
            return {"feature": self.drivers[0]["feature"], "shap_value": self.drivers[0]["shap_value"]}
        return None

    @property
    def feature_values(self):
        payload = self.shap_payload
        if isinstance(payload, dict) and "feature_values" in payload:
            return payload["feature_values"]
        return {}


class BehaviorChange(Base):
    __tablename__ = "behavior_changes"

    id = Column(Integer, primary_key=True, index=True)
    change_id = Column(String(64), unique=True, index=True, nullable=False)
    customer_id = Column(String(64), ForeignKey("customers.customer_id"), index=True, nullable=False)
    metric = Column(String(64), nullable=False)  # frequency, monetary, cart_activity, view_velocity
    baseline_value = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    pct_change = Column(Float, nullable=False)
    severity = Column(String(16), index=True, nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    detection_method = Column(String(32), nullable=False)  # Z_SCORE, ROLLING_AVG, EWMA, ISOLATION_FOREST
    detected_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="behavior_changes")


class UpliftPrediction(Base):
    __tablename__ = "uplift_predictions"

    id = Column(Integer, primary_key=True, index=True)
    uplift_id = Column(String(64), unique=True, index=True, nullable=False)
    customer_id = Column(String(64), ForeignKey("customers.customer_id"), index=True, nullable=False)
    treatment_type = Column(String(64), default="DIRECT_INTERVENTION")
    estimated_uplift = Column(Float, nullable=False)  # e.g. +0.084 (8.4% incremental probability)
    uplift_decile = Column(Integer, index=True, nullable=False)  # 1 to 10
    confidence_interval_low = Column(Float, nullable=False)
    confidence_interval_high = Column(Float, nullable=False)
    model_used = Column(String(32), default="X_LEARNER")  # TWO_MODEL, X_LEARNER
    randomization_assumption_valid = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="uplift_predictions")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    recommendation_id = Column(String(64), unique=True, index=True, nullable=False)
    customer_id = Column(String(64), ForeignKey("customers.customer_id"), index=True, nullable=False)
    action_type = Column(String(64), index=True, nullable=False)  # NO_ACTION, PRODUCT_RECOMMENDATION, REMINDER, EMAIL, DISCOUNT, LOYALTY_REWARD, WIN_BACK, CROSS_SELL, UPSELL
    rank = Column(Integer, default=1)
    score = Column(Float, default=0.0)
    what_text = Column(Text, nullable=False)
    why_text = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=True)  # JSON summary of state, churn, uplift, behavior
    expected_impact = Column(Float, default=0.0)  # Estimated incremental revenue or retention impact
    confidence_level = Column(String(16), default="HIGH")  # LOW, MEDIUM, HIGH
    status = Column(String(32), default="PENDING")  # PENDING, ACCEPTED, REJECTED, EXECUTED
    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="recommendations")

    @property
    def evidence(self):
        if self.evidence_json:
            try:
                return json.loads(self.evidence_json)
            except Exception:
                return {}
        return {}


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), unique=True, index=True, nullable=False)
    model_name = Column(String(64), index=True, nullable=False)
    model_version = Column(String(32), default="v1.0")
    model_type = Column(String(32), nullable=False)  # segmentation, churn, next_event, uplift
    dataset_hash = Column(String(64), nullable=False)
    row_count = Column(Integer, nullable=False)
    hyperparameters_json = Column(Text, nullable=True)
    train_timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(32), default="COMPLETED")
    pr_auc = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    qini_score = Column(Float, nullable=True)
    silhouette_score = Column(Float, nullable=True)

    metrics = relationship("ModelMetric", back_populates="model_run", cascade="all, delete-orphan")


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("model_runs.run_id"), index=True, nullable=False)
    metric_name = Column(String(64), nullable=False)
    metric_value = Column(Float, nullable=False)
    dataset_split = Column(String(16), default="TEST")  # TRAIN, VAL, TEST
    created_at = Column(DateTime, default=datetime.utcnow)

    model_run = relationship("ModelRun", back_populates="metrics")


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), unique=True, index=True, nullable=False)
    session_id = Column(String(64), index=True, nullable=False)
    user_prompt = Column(Text, nullable=False)
    final_response = Column(Text, nullable=False)
    tool_calls_count = Column(Integer, default=0)
    token_count = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    tool_calls = relationship("AgentToolCall", back_populates="agent_run", cascade="all, delete-orphan")


class AgentToolCall(Base):
    __tablename__ = "agent_tool_calls"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("agent_runs.run_id"), index=True, nullable=False)
    tool_name = Column(String(64), nullable=False)
    tool_args_json = Column(Text, nullable=True)
    tool_result_json = Column(Text, nullable=True)
    cached = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    agent_run = relationship("AgentRun", back_populates="tool_calls")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), nullable=True)
    action = Column(String(64), index=True, nullable=False)
    resource = Column(String(128), nullable=False)
    details_json = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class UploadedDataset(Base):
    __tablename__ = "uploaded_datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(String(64), unique=True, index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    dataset_mode = Column(String(64), nullable=False)  # CUSTOMER_EVENT, CUSTOMER_TRANSACTION, CAMPAIGN_UPLIFT, BUSINESS_SALES, GENERIC_TABULAR
    domain = Column(String(64), default="CUSTOMER")
    mode_label = Column(String(128), nullable=True)
    row_count = Column(Integer, default=0)
    col_count = Column(Integer, default=0)
    profile_json = Column(Text, nullable=True)
    column_mapping_json = Column(Text, nullable=True)
    capabilities_json = Column(Text, nullable=True)
    report_json = Column(Text, nullable=True)
    raw_csv_path = Column(String(512), nullable=True)
    status = Column(String(32), default="PROFILED", index=True)  # PROFILED, ANALYZED, ACTIVE, FAILED
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    analyzed_at = Column(DateTime, nullable=True)

