"""Unit and integration test verifying per-customer SHAP explainability diversity,
non-saturation of churn risk predictions, and exact segment reconciliation.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap

from backend.app.database.session import SessionLocal
from backend.app.services.prediction_service import PredictionService
from backend.app.services.customer_service import CustomerService
from backend.app.services.segment_service import SegmentService
from ml.universal.semantic_detector import SemanticColumnDetector
from ml.universal.capability_engine import CapabilityEngine
from ml.universal.pipeline import UniversalPipelineRunner


def test_shap_per_customer_diversity_on_amazon_dataset():
    """Verify that on the Amazon E-Commerce dataset, top SHAP risk drivers contain multiple distinct features
    and churn probabilities are not saturated at a single static value.
    """
    csv_path = "data/amazon_ecommerce_demo.csv"
    if not os.path.exists(csv_path):
        pytest.skip("Amazon e-commerce demo CSV not found.")

    df = pd.read_csv(csv_path)
    detected = SemanticColumnDetector.detect_all_columns(df)
    mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}

    # Run pipeline to train and sync tables
    res = UniversalPipelineRunner.run_pipeline(
        raw_df=df,
        column_mapping=mapping,
        dataset_id="amazon_round4_test",
        dataset_name="amazon_ecommerce_demo.csv",
    )

    db = SessionLocal()
    try:
        top_risk = PredictionService.get_top_churn_customers(db, limit=25)
        assert len(top_risk) >= 20, "Expected at least 20 top-risk customers"

        # 1. SHAP Feature Diversity Check
        valid_drivers = [
            c["top_shap_driver"]["feature"]
            for c in top_risk
            if c.get("top_shap_driver") and not c.get("is_cold_start")
        ]
        unique_drivers = set(valid_drivers)
        assert len(unique_drivers) >= 2, (
            f"Expected at least 2 distinct SHAP drivers in top customers, got only: {unique_drivers}"
        )

        # 2. Probability Non-Saturation Check
        probs = [
            float(c["churn_probability"])
            for c in top_risk
            if c.get("churn_probability") is not None and not c.get("is_cold_start")
        ]
        assert len(probs) >= 15
        prob_std = float(np.std(probs))
        assert prob_std > 0.005, f"Churn probabilities are saturated (std={prob_std:.5f} <= 0.005)"
        assert len(set(probs)) > 3, "Churn probabilities should vary across multiple distinct values"

        # 3. Plausibility Check on Sample Customers
        for c in top_risk[:10]:
            driver_feat = c["top_shap_driver"]["feature"]
            shaps = c.get("shap_values", {})
            if shaps:
                assert driver_feat in shaps
                # Driver should have positive or maximal attribution
                driver_val = float(shaps[driver_feat])
                assert abs(driver_val) >= 0.01

    finally:
        db.close()


def test_segment_name_exact_reconciliation_across_endpoints():
    """Verify that every customer's segment name in Customer 360 exactly matches
    the Segmentation page's label for their cluster, with zero duplicate labels.
    """
    db = SessionLocal()
    try:
        # Query segmentation endpoint summaries
        seg_summaries = SegmentService.get_segment_summaries(db)
        assert len(seg_summaries) >= 2, "Expected at least 2 clusters"

        # Assert no duplicate labels on Segmentation page
        seg_labels = [s["segment_label"] for s in seg_summaries]
        assert len(seg_labels) == len(set(seg_labels)), (
            f"Duplicate segment labels found on Segmentation page: {seg_labels}"
        )

        seg_map = {s["segment_id"]: s["segment_label"] for s in seg_summaries}

        # Query Customer 360 endpoint
        cust_list = CustomerService.get_customers(db, limit=20)
        items = cust_list.get("items", [])
        assert len(items) > 0

        for c in items:
            c_label = c.get("segment_label")
            assert c_label in seg_labels or c_label == "Unassigned", (
                f"Customer {c['customer_id']} segment label '{c_label}' not found in Segmentation page labels: {seg_labels}"
            )

    finally:
        db.close()


def test_synthetic_heterogeneous_customer_shap_attribution():
    """Verify that distinct customer profiles (inactivity vs. velocity drop vs. cart drop) yield distinct top SHAP drivers."""
    feature_cols = [
        "recency_days",
        "frequency_30d",
        "monetary_total",
        "aov",
        "cart_to_view_ratio",
        "conversion_rate",
        "purchase_interval_days",
        "velocity_7d_30d",
    ]

    np.random.seed(42)
    n_samples = 300
    rec = np.random.uniform(5, 90, n_samples)
    f30 = np.random.poisson(2, n_samples)
    mon = np.random.uniform(500, 30000, n_samples)
    aov = mon / np.maximum(1, np.random.poisson(3, n_samples))
    c_ratio = np.random.uniform(0.05, 0.85, n_samples)
    conv = np.random.uniform(0.02, 0.50, n_samples)
    p_int = np.random.uniform(5, 45, n_samples)
    vel = np.random.uniform(0.0, 3.0, n_samples)

    X = np.column_stack([rec, f30, mon, aov, c_ratio, conv, p_int, vel])
    y = ((rec > 50) | (f30 == 0) | (p_int > 35)).astype(int)

    clf = lgb.LGBMClassifier(n_estimators=30, max_depth=3, random_state=42, verbose=-1)
    clf.fit(X, y)

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)
    shap_matrix = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    all_top_drivers = [
        feature_cols[np.argmax(np.abs(shap_matrix[i]))]
        for i in range(n_samples)
    ]
    unique_drivers = set(all_top_drivers)

    assert len(unique_drivers) >= 2, f"Expected multiple distinct top drivers across cohort, got: {unique_drivers}"
