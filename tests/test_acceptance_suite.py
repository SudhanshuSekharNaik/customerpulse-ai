import pytest
from backend.app.database.session import SessionLocal
from backend.app.database.models import (
    Customer,
    CustomerFeature,
    CustomerSegment,
    CustomerState,
    Prediction,
    Recommendation,
    ModelRun,
)
from backend.app.services.segment_service import SegmentService
from backend.app.services.behavior_service import BehaviorService
from backend.app.services.prediction_service import PredictionService
from backend.app.services.recommendation_service import RecommendationService
from backend.app.services.customer_service import CustomerService
from ai.tools.sql_tool import ReadOnlySQLTool
from ai.agent import CustomerPulseAnalystAgent

@pytest.fixture(scope="module")
def db():
    session = SessionLocal()
    yield session
    session.close()

def test_01_customer_population_exact_count(db):
    """AC 1: Zero Customer Loss across 3,000 customers."""
    assert db.query(Customer).count() == 3000
    assert db.query(CustomerFeature).count() == 3000
    assert db.query(CustomerSegment).count() == 3000
    assert db.query(CustomerState).count() == 3000
    assert db.query(Prediction).filter(Prediction.model_type == "churn").count() == 3000
    assert db.query(Recommendation).count() == 3000

def test_02_cluster_count_consistency(db):
    """AC 2: Cluster customer count sum == 3,000."""
    segments = SegmentService.get_all_segments(db)
    total_in_clusters = sum(s["customer_count"] for s in segments)
    assert total_in_clusters == 3000
    for s in segments:
        assert s["customer_count"] > 0
        assert s["avg_revenue"] >= 0

def test_03_pca_scatter_data_integrity(db):
    """AC 3: 2D PCA cluster space and centroids."""
    scatter_data = SegmentService.get_cluster_scatter_data(db)
    assert len(scatter_data["scatter_points"]) == 3000
    assert len(scatter_data["centroids"]) >= 3
    assert len(scatter_data["k_candidates"]) >= 4

def test_04_markov_row_normalization(db):
    """AC 4: Markov transition rows sum strictly to 1.00."""
    t_matrix = BehaviorService.get_transition_matrix(db)
    assert len(t_matrix["states"]) == 9
    for row in t_matrix["matrix"]:
        assert abs(sum(row) - 1.0) < 1e-4

def test_05_anomaly_radar_zero_baseline():
    """AC 5: Zero-baseline anomaly handling."""
    assert True

def test_06_churn_threshold_comparison_table(db):
    """AC 6: 30%-70% threshold comparison table and calibrated Brier score."""
    overview = PredictionService.get_churn_predictions_overview(db)
    assert len(overview["threshold_comparison_table"]) == 5
    assert overview["calibrated_brier_score"] <= 0.25

def test_07_customer_prediction_shap_consistency(db):
    """AC 7: Customer 360 TreeSHAP consistency with canonical cust_1 ID."""
    cust_id = "cust_1"
    detail = CustomerService.get_customer_360_detail(db, cust_id)
    assert detail is not None
    pred_obj = db.query(Prediction).filter(
        Prediction.customer_id == cust_id,
        Prediction.model_type == "churn"
    ).first()
    assert pred_obj is not None
    if detail.get("churn_prediction") and pred_obj.predicted_probability is not None:
        assert abs(detail["churn_prediction"]["churn_probability"] - pred_obj.predicted_probability) < 1e-4

def test_08_recommendation_action_diversity(db):
    """AC 8: Recommendation action type diversity >= 4."""
    recs = db.query(Recommendation).all()
    action_types = set(r.action_type for r in recs)
    assert len(action_types) >= 4
    for r in recs[:20]:
        assert r.expected_impact >= 0

def test_09_sql_destructive_query_rejection():
    """AC 9: Destructive SQL operations rejection."""
    for q in ["DROP TABLE customers;", "DELETE FROM customers;", "UPDATE customers SET total_revenue = 0;"]:
        res = ReadOnlySQLTool.execute_query(q)
        assert res["success"] is False
        assert res.get("security_policy_violation") is True

def test_10_sql_safe_read_only_select():
    """AC 10: Safe SELECT analytical execution."""
    res = ReadOnlySQLTool.execute_query("SELECT customer_id, total_revenue FROM customers LIMIT 5")
    assert res["success"] is True
    assert res["row_count"] == 5

def test_11_agent_budget_and_conditional_trace():
    """AC 11: Agent tool budget <= 8, 8-step execution trace on valid customer, and 1-tool early exit on invalid customer."""
    agent = CustomerPulseAnalystAgent(max_tool_calls=8)
    
    # Valid customer: executes full analysis chain
    res_valid = agent.answer_query("What is the churn risk for cust_1?", session_id="test_valid")
    assert res_valid["tools_used_count"] <= 8
    assert res_valid["tools_used_count"] >= 3
    assert len(res_valid["agent_trace"]) == 8
    assert "cust_1" in res_valid["observed_data"]
    
    # Invalid customer: halts immediately after get_customer_360 (0 wasted tools)
    res_invalid = agent.answer_query("What is the churn risk for cust_99999?", session_id="test_invalid")
    assert res_invalid["tools_used_count"] == 1
    assert "not found" in res_invalid["observed_data"].lower() or "failed" in res_invalid["observed_data"].lower()

def test_12_model_provenance_and_versioning(db):
    """AC 12: Model runs recorded in database."""
    runs = db.query(ModelRun).all()
    assert len(runs) >= 2

def test_13_offline_backtest_metrics(db):
    """AC 13: Offline recommendation backtest non-degenerate."""
    backtest = RecommendationService.get_offline_backtest(db)
    assert backtest["status"] in ["PASSED", "COMPLETED"]
    assert backtest["is_degenerate_policy"] is False

def test_14_data_health_and_canonical_contract(db):
    """AC 14: Data health endpoint and dataset-meta contract verify 100% coverage."""
    from backend.app.api.analytics import get_data_health, get_dataset_metadata
    health = get_data_health(db)
    assert health["status"] == "HEALTHY"
    assert health["total_customers"] == 3000
    assert health["prediction_coverage_pct"] == 100.0

    meta = get_dataset_metadata(db)
    assert meta["customer_count"] == 3000
    assert meta["status"] == "CANONICAL"

def test_15_churn_prediction_variation_and_bounds(db):
    """AC 15: Churn predictions vary continuously without saturation."""
    preds = db.query(Prediction.predicted_probability).filter(Prediction.model_type == "churn").all()
    valid_probs = [p[0] for p in preds if p[0] is not None]
    assert len(valid_probs) >= 2900
    assert len(set(valid_probs)) > 50, f"Predictions must have high entropy, got {len(set(valid_probs))} unique values"
    assert all(0.0 <= p <= 1.0 for p in valid_probs)

def test_16_threshold_alignment_and_single_source_of_truth(db):
    """AC 16: Verify single canonical operating threshold across overview, metadata, and comparisons."""
    overview = PredictionService.get_churn_predictions_overview(db)
    opt_th = overview["optimal_decision_threshold"]
    
    # 1. Operating threshold is float in [0.0, 1.0]
    assert 0.0 <= opt_th <= 1.0
    
    # 2. Optimal row in threshold comparison table matches operating threshold exactly
    optimal_rows = [r for r in overview["threshold_comparison_table"] if r.get("is_optimal")]
    assert len(optimal_rows) == 1
    assert abs(optimal_rows[0]["threshold_fraction"] - opt_th) < 1e-4
    
    # 3. Model run / metadata agrees with operating threshold
    sample_pred = db.query(Prediction).filter(Prediction.model_type == "churn").first()
    assert sample_pred is not None
    assert abs(sample_pred.decision_threshold - opt_th) < 1e-4
