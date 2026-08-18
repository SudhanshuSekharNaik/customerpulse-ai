"""Regression test suite for single canonical prediction registry,
cross-endpoint mathematical consistency (Customer 360 vs Predictions API),
per-customer TreeSHAP explainability without fabrication, and hard validation guardrails.
"""

import os
import random
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database.session import SessionLocal
from backend.app.database.models import Customer, Prediction, ModelRun, CustomerFeature
from backend.app.services.customer_service import CustomerService
from backend.app.services.prediction_service import PredictionService
from ml.prediction.churn import ChurnPredictorTrainer
from ml.universal.semantic_detector import SemanticColumnDetector
from ml.universal.pipeline import UniversalPipelineRunner

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_amazon_dataset_and_predictions():
    """Ensure the amazon_ecommerce_demo.csv dataset is loaded and scored."""
    csv_path = "data/amazon_ecommerce_demo.csv"
    if not os.path.exists(csv_path):
        pytest.skip("data/amazon_ecommerce_demo.csv dataset not found.")

    df = pd.read_csv(csv_path)
    detected = SemanticColumnDetector.detect_all_columns(df)
    mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}

    # Run universal pipeline to establish canonical state and predictions
    UniversalPipelineRunner.run_pipeline(
        raw_df=df,
        column_mapping=mapping,
        dataset_id="ds_amazon_demo_regression",
        dataset_name="amazon_ecommerce_demo.csv",
    )


def test_customer_360_and_predictions_api_probability_consistency():
    """Customer 360 and Predictions API must return the EXACT same churn probability
    for the same customer_id across at least 20 randomly selected customers.
    """
    db = SessionLocal()
    try:
        # Fetch top predictions from Predictions API
        resp = client.get("/api/predictions/churn/top-risk?limit=100")
        assert resp.status_code == 200
        api_preds = resp.json()
        assert len(api_preds) >= 25, f"Expected at least 25 predictions from API, got {len(api_preds)}"

        # Sample at least 20 random customers from the predictions
        sampled_preds = random.sample(api_preds, 25)

        for item in sampled_preds:
            cid = item["customer_id"]
            pred_prob = item["churn_probability"]

            # 1. Query Customer 360 Detail endpoint
            resp_detail = client.get(f"/api/customers/{cid}")
            assert resp_detail.status_code == 200
            detail_data = resp_detail.json()
            c360_prob = detail_data["churn_prediction"]["predicted_probability"] if detail_data.get("churn_prediction") else None

            # Assert exact equality between Customer 360 Detail and Predictions API
            assert c360_prob == pred_prob, (
                f"Contradiction detected for customer '{cid}': "
                f"Customer 360 Detail probability = {c360_prob}, Predictions API probability = {pred_prob}"
            )

            # 2. Query canonical database Prediction record directly
            db_pred = db.query(Prediction).filter(
                Prediction.customer_id == cid,
                Prediction.model_type == "churn"
            ).first()
            assert db_pred is not None, f"Database prediction not found for {cid}"
            assert db_pred.predicted_probability == pred_prob, (
                f"DB prediction {db_pred.predicted_probability} != API prediction {pred_prob} for {cid}"
            )

            # 3. Query Customer 360 list search
            resp_list = client.get(f"/api/customers?search={cid}")
            assert resp_list.status_code == 200
            items = resp_list.json().get("items", [])
            matched = next((c for c in items if c["customer_id"] == cid), None)
            if matched:
                assert matched["churn_probability"] == pred_prob, (
                    f"Customer list probability {matched['churn_probability']} != Predictions API {pred_prob} for {cid}"
                )

    finally:
        db.close()


def test_churn_probabilities_non_saturated_multiple_unique_values():
    """Verify that churn probabilities are not collapsed/saturated at a single static value (e.g. 84.9%).
    At least 2 distinct probabilities must exist across the customer cohort.
    """
    db = SessionLocal()
    try:
        active_preds = db.query(Prediction.predicted_probability).filter(
            Prediction.model_type == "churn",
            Prediction.predicted_probability.isnot(None),
        ).all()
        probs = [float(p[0]) for p in active_preds]

        assert len(probs) >= 20, "Expected at least 20 active churn predictions"
        unique_probs = set(probs)
        assert len(unique_probs) > 5, (
            f"Prediction collapse detected: Only {len(unique_probs)} unique churn probabilities found across {len(probs)} customers: {unique_probs}"
        )
        assert np.std(probs) > 0.01, f"Prediction variance too low (std={np.std(probs):.5f})"

    finally:
        db.close()


def test_shap_per_customer_exact_drivers_and_no_fabrication():
    """Verify that each customer has individual TreeSHAP explanations containing:
    - top_risk_factor
    - positive_drivers
    - protective_drivers
    - feature_values
    and that SHAP drivers vary across different customers.
    """
    db = SessionLocal()
    try:
        top_risk = PredictionService.get_top_churn_customers(db, limit=30)
        assert len(top_risk) >= 20

        top_drivers_set = set()
        for p in top_risk:
            if p.get("is_cold_start"):
                continue

            # Must have non-null probability
            assert p["churn_probability"] is not None
            assert 0.0 <= p["churn_probability"] <= 1.0

            # Must have top_risk_factor
            assert p["top_risk_factor"] is not None, f"Customer {p['customer_id']} missing top_risk_factor"
            assert "feature" in p["top_risk_factor"]
            assert "shap_value" in p["top_risk_factor"]
            assert not np.isnan(p["top_risk_factor"]["shap_value"])

            top_drivers_set.add(p["top_risk_factor"]["feature"])

            # Must have drivers list
            assert "drivers" in p
            assert isinstance(p["drivers"], list)

            # Check positive and protective driver directions
            for d in p.get("positive_drivers", []):
                assert d["direction"] == "increases_risk"
                assert d["shap_value"] > 0

            for d in p.get("protective_drivers", []):
                assert d["direction"] == "decreases_risk"
                assert d["shap_value"] < 0

        # Cohort SHAP diversity: at least 2 distinct top features across top 30
        assert len(top_drivers_set) >= 2, f"Expected diverse SHAP drivers across cohort, got only: {top_drivers_set}"

    finally:
        db.close()


def test_hard_validation_guardrails_fail_on_prediction_collapse():
    """Verify that the model validation guardrail fails with ValueError if all customers receive identical probabilities."""
    trainer = ChurnPredictorTrainer()

    # Create dummy customer features with identical collapsed outputs
    dummy_feats = pd.DataFrame([
        {"customer_id": f"cust_{i}", "recency_days": 10.0, "is_cold_start": False, **{col: 1.0 for col in ChurnPredictorTrainer.FEATURE_COLS}}
        for i in range(10)
    ])

    # Mock a dummy model that outputs identical probability 0.849 for all rows
    class MockCollapsedModel:
        def predict_proba(self, X):
            return np.array([[0.151, 0.849]] * len(X))

    class MockExplainer:
        def shap_values(self, X):
            return np.ones_like(X) * 0.5

    trainer.best_model = MockCollapsedModel()
    trainer.explainer = MockExplainer()

    with pytest.raises(ValueError, match="Prediction collapse detected"):
        trainer.score_all_customers_and_save(dummy_feats)


def test_no_nan_or_inf_in_predictions_or_shap():
    """Verify zero NaN or Inf values across all customer predictions and SHAP explanations in database."""
    db = SessionLocal()
    try:
        preds = db.query(Prediction).filter(Prediction.model_type == "churn").all()
        assert len(preds) > 0

        for p in preds:
            if p.predicted_probability is not None:
                assert not np.isnan(p.predicted_probability), f"Customer {p.customer_id} has NaN probability"
                assert not np.isinf(p.predicted_probability), f"Customer {p.customer_id} has Inf probability"

            shaps = p.shap_values
            if shaps and isinstance(shaps, dict):
                for feat, val in shaps.items():
                    assert not np.isnan(float(val)), f"Customer {p.customer_id} has NaN SHAP for {feat}"
                    assert not np.isinf(float(val)), f"Customer {p.customer_id} has Inf SHAP for {feat}"
    finally:
        db.close()
