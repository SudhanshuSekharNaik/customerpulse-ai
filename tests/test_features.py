"""Tests for leakage-safe Customer 360 Feature Engineering."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from ml.data.features import FeatureEngineer


def test_leakage_safe_features():
    fe = FeatureEngineer()
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    
    events_df = pd.DataFrame({
        "event_id": ["e1", "e2", "e3"],
        "customer_id": ["c1", "c1", "c1"],
        "visitor_id": ["c1", "c1", "c1"],
        "event_type": ["view", "addtocart", "transaction"],
        "item_id": ["i1", "i1", "i1"],
        "category_id": ["cat1", "cat1", "cat1"],
        "timestamp": [
            base_time,
            base_time + timedelta(days=10),
            base_time + timedelta(days=20),
        ],
        "transaction_id": [None, None, "tx1"],
        "revenue": [0.0, 0.0, 150.0],
    })

    # Cutoff at Day 15 (before transaction on Day 20)
    cutoff = base_time + timedelta(days=15)
    feats = fe.compute_customer_features(events_df, cutoff_date=cutoff)
    
    assert len(feats) == 1
    row = feats.iloc[0]
    # Transactions at cutoff should be 0 because event happened on Day 20
    assert row["transactions_90d"] == 0
    assert row["carts_90d"] == 1
    assert row["views_90d"] == 1
