"""Tests for population isolation between RetailRocket and Criteo datasets."""

import pytest
import pandas as pd
from ml.data.ingestion import DataIngestion


def test_population_isolation():
    ingestion = DataIngestion()
    rr_df, _ = ingestion.load_or_generate_retailrocket(sample_fraction=0.05)
    criteo_df, _ = ingestion.load_or_generate_criteo_uplift(sample_fraction=0.05)

    rr_cust_ids = set(rr_df["customer_id"].astype(str))
    
    # Criteo dataset must NEVER contain RetailRocket customer IDs
    for col in criteo_df.columns:
        criteo_vals = set(criteo_df[col].astype(str))
        overlap = rr_cust_ids.intersection(criteo_vals)
        assert len(overlap) == 0, f"Violation: Overlap detected between RetailRocket and Criteo in column {col}"
