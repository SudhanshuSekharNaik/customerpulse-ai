"""Comprehensive Data Integrity & Statistical Correctness Guardrail Tests.
Enforces zero metric fabrication, mathematical invariants, and cross-panel consistency.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

from backend.app.database.session import SessionLocal
from backend.app.database.models import (
    Customer,
    Event,
    Product,
    CustomerState,
    CustomerFeature,
    Prediction,
    BehaviorChange,
    Recommendation,
    ModelRun,
    UploadedDataset,
)
from backend.app.services.behavior_service import BehaviorService
from backend.app.services.prediction_service import PredictionService
from backend.app.api.analytics import get_executive_overview
from ml.ecommerce.traffic_forecast import EcommerceTrafficEngine


@pytest.fixture(scope="module")
def db():
    session = SessionLocal()
    yield session
    session.close()


def test_cross_page_count_consistency(db):
    """Verify single source of truth: total_customers and total_events match database counts."""
    db_customers = db.query(Customer).count()
    db_events = db.query(Event).count()
    db_orders = db.query(Customer.total_orders).all()

    overview = get_executive_overview(db=db)

    assert overview["total_customers"] == db_customers, f"Overview customers {overview['total_customers']} != DB {db_customers}"
    assert overview["total_events"] == db_events, f"Overview events {overview['total_events']} != DB {db_events}"
    expected_orders = sum(o[0] or 0 for o in db_orders)
    assert overview["total_orders"] == expected_orders, f"Overview orders {overview['total_orders']} != DB orders sum {expected_orders}"


def test_orders_and_spend_invariant(db):
    """Zero customers should have total_revenue > 0 and total_orders == 0."""
    invalid_customers = db.query(Customer).filter(
        Customer.total_revenue > 0,
        Customer.total_orders == 0,
    ).all()

    assert len(invalid_customers) == 0, f"Found {len(invalid_customers)} customers with spend > 0 but 0 orders!"


def test_percentage_range_sanity_clamps(db):
    """Assert all percentages and rates are strictly bounded within [0.0, 100.0] or None when no funnel events."""
    traffic_res = EcommerceTrafficEngine.analyze_traffic_and_peaks(db=db)
    funnel = traffic_res["ecommerce_funnel"]

    # Checkout conversion, cart abandonment, cart-to-view
    if funnel.get("has_funnel_events", True):
        assert 0.0 <= funnel["checkout_conversion_rate_pct"] <= 100.0, f"Invalid conversion rate {funnel['checkout_conversion_rate_pct']}"
        assert 0.0 <= funnel["cart_abandonment_rate_pct"] <= 100.0, f"Invalid abandonment rate {funnel['cart_abandonment_rate_pct']}"
        assert 0.0 <= funnel["cart_to_view_ratio_pct"] <= 100.0, f"Invalid cart/view ratio {funnel['cart_to_view_ratio_pct']}"
    else:
        assert funnel["cart_abandonment_rate_pct"] is None
        assert funnel["cart_to_view_ratio_pct"] is None
        assert funnel["checkout_conversion_rate_pct"] is None
        assert "only has completed orders" in funnel.get("funnel_note", "")

    # Executive overview percentages
    overview = get_executive_overview(db=db)
    assert 0.0 <= overview["at_risk_percentage"] <= 100.0, f"Invalid at risk percentage {overview['at_risk_percentage']}"


def test_recommendations_not_flat_spend_formula(db):
    """Assert expected impact is margin-calibrated and not a flat 10% spend multiplier."""
    recs = db.query(Recommendation, Customer)\
        .join(Customer, Customer.customer_id == Recommendation.customer_id)\
        .limit(100).all()

    assert len(recs) > 0, "No recommendations found in database"

    impacts = np.array([r.expected_impact for r, c in recs])
    spends = np.array([float(c.total_revenue) for r, c in recs])

    # If all values were flat 10% spend, correlation would be exactly 1.0 and std(impact / spend) would be 0
    ratios = impacts / np.maximum(1.0, spends)
    assert np.std(ratios) > 0.001, "Expected impact is flat-multiplied against spend without variance!"

    # Check that multiple confidence levels are assigned
    confidences = set(r.confidence_level for r, c in recs)
    assert len(confidences) >= 2, f"Expected varied confidence levels, got: {confidences}"


def test_anomaly_radar_per_customer_baselines(db):
    """Assert anomaly baseline values vary across customers (not a static global constant)."""
    changes = BehaviorService.get_behavior_changes(db=db, limit=50)

    assert len(changes) > 0, "No anomaly behavior changes found"

    baselines = [c["baseline_value"] for c in changes]
    unique_baselines = set(baselines)
    assert len(unique_baselines) > 1, f"All anomaly baselines are identical constant: {unique_baselines}"

    # Assert multiple anomaly metrics exist
    metrics = set(c["metric"] for c in changes)
    assert len(metrics) >= 2, f"Expected multiple anomaly metrics, got: {metrics}"

    # Assert multiple severity levels exist
    severities = set(c["severity"] for c in changes)
    assert len(severities) >= 2, f"Expected varied severities, got: {severities}"


def test_predictions_shap_drivers_variance(db):
    """Assert top churn customers return distinct SHAP drivers and probabilities."""
    top_churn = PredictionService.get_top_churn_customers(db=db, limit=20)

    assert len(top_churn) > 0, "No churn predictions found"

    # Verify probability sorting
    probs = [c["churn_probability"] for c in top_churn]
    assert probs == sorted(probs, reverse=True), "Top churn customers must be sorted descending by probability"

    # Verify orders and states are populated
    for c in top_churn:
        assert c["total_orders"] >= 1, f"Customer {c['customer_id']} has 0 orders"
        assert c["current_state"] in ("LOYAL", "ENGAGED", "EXPLORING", "CONVERTING", "DECLINING", "AT_RISK", "DORMANT", "RECOVERING", "NEW")
        assert c["top_shap_driver"] is not None
        assert "feature" in c["top_shap_driver"]
        assert "shap_value" in c["top_shap_driver"]

    # Verify SHAP values vary across rows
    shap_vals = [c["top_shap_driver"]["shap_value"] for c in top_churn]
    assert len(set(shap_vals)) > 1, f"SHAP driver values must vary across customers, got {set(shap_vals)}"


def test_markov_transition_matrix_row_sum_invariant(db):
    """Every state with customer population > 0 must have outgoing transition probabilities summing to 1.0 ± 0.01."""
    trans = BehaviorService.get_transition_matrix(db=db)
    states = trans["states"]
    matrix = trans["matrix"]

    states_dist = BehaviorService.get_state_distribution(db=db)
    dist_map = {s["state"]: s["customer_count"] for s in states_dist}

    for idx, st in enumerate(states):
        row = matrix[idx]
        row_sum = round(sum(row), 2)
        cust_count = dist_map.get(st, 0)
        if cust_count > 0:
            assert abs(row_sum - 1.0) <= 0.01, f"State '{st}' with {cust_count} customers has row sum {row_sum} != 1.0 (row: {row})"


from sqlalchemy import desc


def test_pr_auc_alignment_with_model_run(db):
    """Predictions overview PR-AUC must match the latest ModelRun PR-AUC in database."""
    latest_run = db.query(ModelRun).filter(ModelRun.model_type == "churn", ModelRun.status == "COMPLETED").order_by(desc(ModelRun.train_timestamp)).first()
    if latest_run and latest_run.pr_auc is not None:
        ov = PredictionService.get_churn_predictions_overview(db=db)
        assert round(ov["pr_auc"], 4) == round(float(latest_run.pr_auc), 4), f"Overview PR-AUC {ov['pr_auc']} != ModelRun {latest_run.pr_auc}"
