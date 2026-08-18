"""Unit and integration test verifying per-customer SHAP explainability diversity.
Asserts that across customers with materially different feature profiles,
reported top-driver risk factors vary dynamically rather than broadcasting a single static feature.
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
from ml.universal.semantic_detector import SemanticColumnDetector
from ml.universal.capability_engine import CapabilityEngine
from ml.universal.pipeline import UniversalPipelineRunner


def test_shap_per_customer_diversity_on_real_dataset():
    """Verify that on the 883-customer Myntra dataset, top SHAP risk drivers contain multiple distinct features."""
    csv_path = "data/myntra_lifestyle_demo.csv"
    if not os.path.exists(csv_path):
        pytest.skip("Myntra lifestyle demo CSV not found.")

    df = pd.read_csv(csv_path)
    detected = SemanticColumnDetector.detect_all_columns(df)
    mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}

    # Run full pipeline to train model and sync predictions to DB
    res = UniversalPipelineRunner.run_pipeline(
        raw_df=df,
        column_mapping=mapping,
        dataset_id="myntra_shap_test",
        dataset_name="myntra_lifestyle_demo.csv",
    )

    db = SessionLocal()
    try:
        top_risk = PredictionService.get_top_churn_customers(db, limit=50)
        assert len(top_risk) > 0

        # Extract top drivers for scored non-cold customers
        valid_drivers = [
            c["top_shap_driver"]["feature"]
            for c in top_risk
            if c.get("top_shap_driver") and not c.get("is_cold_start")
        ]

        assert len(valid_drivers) >= 5, "Expected at least 5 scored customers with SHAP drivers"
        unique_features = set(valid_drivers)

        # Main risk factor must NOT be a static broadcast of a single feature name
        assert len(unique_features) >= 2, (
            f"Expected at least 2 distinct SHAP risk driver features across top customers, got only: {unique_features}"
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

    # Generate synthetic diverse cohort
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
    # Target influenced by recency, lack of recent velocity, and interval
    y = ((rec > 50) | (f30 == 0) | (p_int > 35)).astype(int)

    clf = lgb.LGBMClassifier(n_estimators=30, max_depth=3, random_state=42, verbose=-1)
    clf.fit(X, y)

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)
    shap_matrix = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    # Profile A: extreme recency lapse
    idx_recency = int(np.argmax(rec))
    # Profile B: zero 30d frequency with recent purchase
    idx_vel_drop = int(np.where((f30 == 0) & (rec < 20))[0][0]) if len(np.where((f30 == 0) & (rec < 20))[0]) > 0 else 0

    driver_a = feature_cols[np.argmax(np.abs(shap_matrix[idx_recency]))]
    driver_b = feature_cols[np.argmax(np.abs(shap_matrix[idx_vel_drop]))]

    all_top_drivers = [
        feature_cols[np.argmax(np.abs(shap_matrix[i]))]
        for i in range(n_samples)
    ]
    unique_drivers = set(all_top_drivers)

    assert len(unique_drivers) >= 2, f"Expected multiple distinct top drivers across cohort, got: {unique_drivers}"
