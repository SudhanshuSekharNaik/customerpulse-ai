"""Tests for offline policy backtesting and action diversity."""

import pytest
from ml.recommendation.offline_policy_eval import OfflinePolicyEvaluator


def test_offline_policy_diversity():
    evaluator = OfflinePolicyEvaluator()
    
    # Diverse policy sample
    diverse_recs = [
        {"action_type": "DISCOUNT", "expected_impact": 350.0, "score": 85.0},
        {"action_type": "WIN_BACK", "expected_impact": 500.0, "score": 90.0},
        {"action_type": "LOYALTY_REWARD", "expected_impact": 200.0, "score": 88.0},
        {"action_type": "PRODUCT_RECOMMENDATION", "expected_impact": 150.0, "score": 70.0},
        {"action_type": "REMINDER", "expected_impact": 100.0, "score": 75.0},
    ]
    report = evaluator.evaluate_policy(diverse_recs)
    assert report["status"] == "PASSED"
    assert report["action_diversity_index"] > 0.5
    assert report["is_degenerate_policy"] is False


def test_offline_policy_degenerate_detection():
    evaluator = OfflinePolicyEvaluator()
    # Degenerate policy where 90% is DISCOUNT
    degen_recs = [{"action_type": "DISCOUNT", "expected_impact": 300.0, "score": 80.0} for _ in range(9)] + [
        {"action_type": "WIN_BACK", "expected_impact": 500.0, "score": 90.0}
    ]
    report = evaluator.evaluate_policy(degen_recs)
    assert report["is_degenerate_policy"] is True
