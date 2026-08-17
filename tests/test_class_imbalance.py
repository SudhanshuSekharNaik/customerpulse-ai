"""Tests for class imbalance handling and cost-optimal decision threshold tuning."""

import pytest
import numpy as np
from ml.prediction.churn import ChurnPredictorTrainer


def test_cost_optimal_threshold_calculation():
    trainer = ChurnPredictorTrainer(cost_ratio_fn_fp=5.0)
    
    # Synthetic imbalanced outcomes (10% positive churn)
    y_true = np.array([1]*10 + [0]*90)
    y_probs = np.linspace(0.05, 0.95, 100)
    
    best_th, min_cost = trainer.find_cost_optimal_threshold(y_true, y_probs)
    assert 0.05 <= best_th <= 0.95
    assert min_cost >= 0.0
