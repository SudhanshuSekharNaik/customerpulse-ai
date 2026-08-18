"""Comprehensive test suite verifying the clean e-commerce benchmark dataset:
- amazon_ecommerce_customerpulse_clean.csv
- 12,000 event rows
- 1,299 unique customers
- 7,813 views, 2,903 add-to-carts, 1,284 purchases
- ₹36,638,759.90 total purchase revenue
- Exact cross-service consistency for target test accounts (AMZ_CUST_00001, AMZ_CUST_00002, AMZ_CUST_00006, AMZ_CUST_00532, AMZ_CUST_00848)
"""

import os
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database.session import SessionLocal
from backend.app.database.models import Customer, CustomerFeature, CustomerSegment, CustomerState, Prediction, Event
from backend.app.services.customer_service import CustomerService
from backend.app.services.prediction_service import PredictionService
from ai.tools.prediction_tools import PredictionTools
from ml.universal.semantic_detector import SemanticColumnDetector
from ml.universal.pipeline import UniversalPipelineRunner

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_clean_dataset():
    """Load and process amazon_ecommerce_customerpulse_clean.csv."""
    csv_path = "data/amazon_ecommerce_customerpulse_clean.csv"
    if not os.path.exists(csv_path):
        # Fallback to amazon_ecommerce_demo.csv
        csv_path = "data/amazon_ecommerce_demo.csv"
    
    assert os.path.exists(csv_path), f"Clean dataset file not found at {csv_path}"
    df = pd.read_csv(csv_path)
    
    # Assert raw CSV metrics
    assert len(df) == 12000, f"Expected 12000 rows, got {len(df)}"
    cust_col = "canonical_customer_id" if "canonical_customer_id" in df.columns else "customer_id"
    assert df[cust_col].nunique() == 1299, f"Expected 1299 unique customers, got {df[cust_col].nunique()}"
    
    ev_counts = df["event_type"].value_counts().to_dict()
    assert ev_counts.get("view") == 7813, f"Expected 7813 views, got {ev_counts.get('view')}"
    assert ev_counts.get("addtocart") == 2903, f"Expected 2903 carts, got {ev_counts.get('addtocart')}"
    assert ev_counts.get("purchase") == 1284, f"Expected 1284 purchases, got {ev_counts.get('purchase')}"
    
    purchase_rev = df[df["event_type"] == "purchase"]["sales_amount_inr"].sum()
    assert abs(purchase_rev - 36638759.90) < 1.0, f"Expected ~36,638,759.90 purchase revenue, got {purchase_rev}"

    # Run pipeline to load into database
    detected = SemanticColumnDetector.detect_all_columns(df)
    mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}
    UniversalPipelineRunner.run_pipeline(
        raw_df=df,
        column_mapping=mapping,
        dataset_id="ds_clean_amazon_benchmark",
        dataset_name="amazon_ecommerce_customerpulse_clean.csv",
    )


def test_executive_overview_kpis():
    """Verify Executive Overview API calculates dynamic dataset KPIs matching source of truth."""
    resp = client.get("/api/analytics/overview")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_customers"] == 1299, f"Expected 1299 customers, got {data['total_customers']}"
    assert data["total_events"] == 12000, f"Expected 12000 events, got {data['total_events']}"
    assert data["total_orders"] == 1284, f"Expected 1284 orders, got {data['total_orders']}"
    assert abs(data["total_revenue"] - 36638759.90) < 1.0, f"Expected ₹36,638,759.90 total revenue, got {data['total_revenue']}"
    assert data["total_orders"] != data["total_events"], "Orders must not equal events"


def test_target_test_accounts_full_consistency():
    """Verify the 5 required target test accounts across all services and tables:
    AMZ_CUST_00001, AMZ_CUST_00002, AMZ_CUST_00006, AMZ_CUST_00532, AMZ_CUST_00848
    """
    db = SessionLocal()
    try:
        target_cids = [
            "AMZ_CUST_00001",
            "AMZ_CUST_00002",
            "AMZ_CUST_00006",
            "AMZ_CUST_00532",
            "AMZ_CUST_00848",
        ]

        for cid in target_cids:
            cust = db.query(Customer).filter(Customer.customer_id == cid).first()
            feat = db.query(CustomerFeature).filter(CustomerFeature.customer_id == cid).first()
            st = db.query(CustomerState).filter(CustomerState.customer_id == cid).first()
            seg = db.query(CustomerSegment).filter(CustomerSegment.customer_id == cid).first()
            pred = db.query(Prediction).filter(Prediction.customer_id == cid, Prediction.model_type == "churn").first()

            assert cust is not None, f"Customer {cid} not found in database"
            assert feat is not None, f"CustomerFeature for {cid} not found"
            assert st is not None, f"CustomerState for {cid} not found"
            assert seg is not None, f"CustomerSegment for {cid} not found"
            assert pred is not None, f"Prediction for {cid} not found"

            # 1. Customer 360 API & Service
            c360_detail = CustomerService.get_customer_360_detail(db, cid)
            assert c360_detail is not None

            # 2. AI Analyst Tool
            ai_churn = PredictionTools.get_churn_analysis(cid)
            assert ai_churn is not None

            # Verify: Customer 360 Churn == Predictions Churn == AI Analyst Churn
            c360_prob = c360_detail["churn_prediction"]["predicted_probability"] if c360_detail.get("churn_prediction") else None
            assert c360_prob == pred.predicted_probability, f"C360 churn {c360_prob} != DB pred {pred.predicted_probability} for {cid}"
            assert ai_churn["churn_probability"] == pred.predicted_probability, f"AI churn {ai_churn['churn_probability']} != DB pred {pred.predicted_probability} for {cid}"

            # Verify: Customer 360 Total Spend == Feature Table Total Spend == Customer Table Total Revenue
            assert cust.total_revenue == feat.monetary_total, f"Cust rev {cust.total_revenue} != Feat rev {feat.monetary_total} for {cid}"
            assert c360_detail["total_revenue"] == cust.total_revenue, f"C360 rev {c360_detail['total_revenue']} != Cust rev {cust.total_revenue} for {cid}"

            # Verify: Customer 360 Orders == Purchase order count
            assert cust.total_orders == feat.transactions_90d, f"Cust orders {cust.total_orders} != Feat tx {feat.transactions_90d} for {cid}"
            assert c360_detail["total_orders"] == cust.total_orders, f"C360 orders {c360_detail['total_orders']} != Cust orders {cust.total_orders} for {cid}"

            # Verify: Segmentation Customer == Customer 360 Segment
            assert c360_detail["segment_label"] == seg.segment_label, f"C360 segment {c360_detail['segment_label']} != DB seg {seg.segment_label} for {cid}"

            # Verify: State Machine Lifecycle == Customer 360 Lifecycle
            assert c360_detail["current_state"] == st.current_state, f"C360 state {c360_detail['current_state']} != DB state {st.current_state} for {cid}"

            # Verify: SHAP main risk factor is populated and valid
            top_risk = pred.top_risk_factor
            if not cust.is_cold_start:
                assert top_risk is not None, f"Expected top_risk_factor for active customer {cid}"
                assert "feature" in top_risk
                assert "shap_value" in top_risk
    finally:
        db.close()


def test_canonical_customer_ids_preserved_no_synthetic_mutation():
    """Verify that canonical customer IDs (e.g. AMZ_CUST_00001) are preserved across all tables without conversion to cust_1."""
    db = SessionLocal()
    try:
        sample_custs = db.query(Customer.customer_id).limit(50).all()
        for (cid,) in sample_custs:
            assert cid.startswith("AMZ_CUST_"), f"Customer ID '{cid}' was mutated from canonical format"
    finally:
        db.close()
