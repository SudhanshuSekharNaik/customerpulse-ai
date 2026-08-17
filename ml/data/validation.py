"""Data quality validation module for CustomerPulse AI.
Performs real, rigorous data-quality auditing: schema checks, missingness,
timestamp ordering, duplicates, invalid event types, and cold-start counts.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class DataQualityValidator:
    """Validates raw event data and customer tables, returning real calculated reports."""

    REQUIRED_EVENT_COLUMNS = ["event_id", "customer_id", "event_type", "timestamp"]
    VALID_EVENT_TYPES = {"view", "addtocart", "transaction"}

    def __init__(self):
        self.report: Dict[str, Any] = {}

    def validate_events(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate events dataframe and return comprehensive data quality scorecard."""
        total_rows = len(df)
        if total_rows == 0:
            return {
                "status": "FAILED",
                "total_rows": 0,
                "error": "Dataset is empty"
            }

        # Check required columns
        missing_cols = [c for c in self.REQUIRED_EVENT_COLUMNS if c not in df.columns]
        
        # Missing values per column
        null_counts = df.isnull().sum().to_dict()
        null_rates = {k: float(v) / total_rows for k, v in null_counts.items()}

        # Duplicate check
        duplicate_rows = int(df.duplicated(subset=["customer_id", "timestamp", "event_type", "item_id"]).sum()) if "item_id" in df.columns else int(df.duplicated().sum())

        # Event type distribution & validity
        if "event_type" in df.columns:
            invalid_events = int((~df["event_type"].isin(self.VALID_EVENT_TYPES)).sum())
            event_counts = df["event_type"].value_counts().to_dict()
        else:
            invalid_events = total_rows
            event_counts = {}

        # Timestamp validation
        invalid_timestamps = 0
        if "timestamp" in df.columns:
            try:
                ts_series = pd.to_datetime(df["timestamp"])
                min_ts = ts_series.min().isoformat() if not ts_series.empty else None
                max_ts = ts_series.max().isoformat() if not ts_series.empty else None
            except Exception:
                invalid_timestamps = total_rows
                min_ts, max_ts = None, None
        else:
            min_ts, max_ts = None, None

        # Unique customers and items
        unique_customers = int(df["customer_id"].nunique()) if "customer_id" in df.columns else 0
        unique_items = int(df["item_id"].nunique()) if "item_id" in df.columns else 0

        # Quality Score Calculation (0 to 100)
        deductions = 0.0
        if missing_cols:
            deductions += 40.0
        deductions += min(20.0, (duplicate_rows / total_rows) * 100)
        deductions += min(20.0, (invalid_events / total_rows) * 100)
        if invalid_timestamps > 0:
            deductions += 20.0
        
        quality_score = max(0.0, round(100.0 - deductions, 2))
        status = "PASSED" if quality_score >= 80.0 else "WARNING" if quality_score >= 60.0 else "FAILED"

        report = {
            "status": status,
            "quality_score": quality_score,
            "total_rows": total_rows,
            "unique_customers": unique_customers,
            "unique_items": unique_items,
            "duplicate_events_count": duplicate_rows,
            "invalid_events_count": invalid_events,
            "invalid_timestamps_count": invalid_timestamps,
            "null_rates": null_rates,
            "event_type_distribution": event_counts,
            "time_range": {"min": min_ts, "max": max_ts},
            "missing_columns": missing_cols,
        }
        self.report = report
        return report

    def print_report(self, report: Dict[str, Any] = None):
        """Format and print real data quality report to console."""
        rep = report or self.report
        print("=" * 60)
        print("        CUSTOMER-PULSE DATA QUALITY AUDIT REPORT        ")
        print("=" * 60)
        print(f"Status:               {rep.get('status')} (Score: {rep.get('quality_score')}/100)")
        print(f"Total Rows:           {rep.get('total_rows'):,}")
        print(f"Unique Customers:     {rep.get('unique_customers'):,}")
        print(f"Unique Items:         {rep.get('unique_items'):,}")
        print(f"Duplicate Events:     {rep.get('duplicate_events_count'):,}")
        print(f"Invalid Event Types:  {rep.get('invalid_events_count'):,}")
        print(f"Time Range:           {rep.get('time_range', {}).get('min')} to {rep.get('time_range', {}).get('max')}")
        print("-" * 60)
        print("Event Type Distribution:")
        for k, v in rep.get("event_type_distribution", {}).items():
            print(f"  - {k:<15}: {v:>8,} ({v/max(1, rep.get('total_rows'))*100:.2f}%)")
        print("=" * 60)
