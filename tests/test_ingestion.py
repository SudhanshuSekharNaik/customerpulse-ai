"""Tests for data ingestion and quality validation."""

import pytest
import pandas as pd
from ml.data.ingestion import DataIngestion
from ml.data.validation import DataQualityValidator


def test_data_ingestion_generation():
    ingestion = DataIngestion()
    events_df, content_hash = ingestion.load_or_generate_retailrocket(sample_fraction=0.05)
    assert len(events_df) > 0
    assert len(content_hash) == 16
    assert "customer_id" in events_df.columns
    assert "event_type" in events_df.columns


def test_data_quality_validator():
    validator = DataQualityValidator()
    df = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "customer_id": ["c1", "c2"],
        "event_type": ["view", "transaction"],
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="D"),
        "item_id": ["i1", "i2"],
    })
    report = validator.validate_events(df)
    assert report["status"] == "PASSED"
    assert report["quality_score"] == 100.0
