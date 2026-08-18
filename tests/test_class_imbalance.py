"""Tests for class imbalance handling, cost-optimal decision threshold tuning,
and prediction score variance across heterogeneous customer cohorts.
"""

import pytest
import numpy as np
from ml.prediction.churn import ChurnPredictorTrainer
from backend.app.database.session import SessionLocal
from backend.app.services.prediction_service import PredictionService


def test_cost_optimal_threshold_calculation():
    trainer = ChurnPredictorTrainer(cost_ratio_fn_fp=5.0)
    
    # Synthetic imbalanced outcomes (10% positive churn)
    y_true = np.array([1]*10 + [0]*90)
    y_probs = np.linspace(0.05, 0.95, 100)
    
    best_th, min_cost = trainer.find_cost_optimal_threshold(y_true, y_probs)
    assert 0.05 <= best_th <= 0.95
    assert min_cost >= 0.0


def test_churn_risk_variance_across_predictions_endpoint():
    """Assert that across top churn risk customers, probabilities have non-zero variance (not all clustered at one value)."""
    db = SessionLocal()
    try:
        top_risk = PredictionService.get_top_churn_customers(db, limit=25)
        probs = [
            float(c["churn_probability"])
            for c in top_risk
            if c.get("churn_probability") is not None and not c.get("is_cold_start")
        ]
        if len(probs) >= 5:
            prob_std = float(np.std(probs))
            assert prob_std > 0.001, f"Expected standard deviation above epsilon, got: {prob_std}"
    finally:
        db.close()
