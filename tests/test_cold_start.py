"""Tests for cold-start identification and heuristic rule routing."""

import pytest
import pandas as pd
from ml.prediction.cold_start import ColdStartHandler


def test_cold_start_detection():
    row_cold = pd.Series({"is_cold_start": True, "frequency_90d": 1, "views_7d": 1, "carts_7d": 0})
    row_established = pd.Series({"is_cold_start": False, "frequency_90d": 15, "views_7d": 5, "carts_7d": 2})

    assert ColdStartHandler.is_cold_start(row_cold) is True
    assert ColdStartHandler.is_cold_start(row_established) is False


def test_cold_start_prediction_payload():
    row_cold = pd.Series({"is_cold_start": True, "frequency_90d": 1, "views_7d": 3, "carts_7d": 1})
    pred = ColdStartHandler.predict_cold_start(row_cold)
    
    assert pred["is_cold_start"] is True
    assert pred["status"] == "INSUFFICIENT_HISTORY"
    assert pred["confidence_band"] == "LOW"
