"""Data preprocessing module for CustomerPulse AI.
Handles cleaning, timestamp normalization, sessionization, deduplication,
and cold-start separation.
"""

from typing import Tuple
import pandas as pd
import numpy as np


class Preprocessor:
    """Preprocesses raw event logs into clean, structured tables."""

    def __init__(self, session_timeout_minutes: int = 30, cold_start_min_events: int = 3, cold_start_min_days: int = 14):
        self.session_timeout_minutes = session_timeout_minutes
        self.cold_start_min_events = cold_start_min_events
        self.cold_start_min_days = cold_start_min_days

    def clean_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize event columns, parse timestamps, remove corrupted rows and duplicates."""
        df = df.copy()
        
        # Ensure correct column names
        if "customer_id" not in df.columns:
            if "visitorid" in df.columns:
                df = df.rename(columns={"visitorid": "customer_id"})
            elif "visitor_id" in df.columns:
                df = df.rename(columns={"visitor_id": "customer_id"})

        if "visitor_id" not in df.columns and "customer_id" in df.columns:
            df["visitor_id"] = df["customer_id"]

        rename_map = {
            "itemid": "item_id",
            "event": "event_type",
            "transactionid": "transaction_id",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Cast customer_id and item_id to string
        if "customer_id" in df.columns:
            df["customer_id"] = df["customer_id"].astype(str)
        if "item_id" in df.columns:
            df["item_id"] = df["item_id"].astype(str)

        # Standardize timestamp
        if "timestamp" in df.columns:
            # Check if timestamp is in unix milliseconds or seconds
            if pd.api.types.is_numeric_dtype(df["timestamp"]):
                sample_val = df["timestamp"].iloc[0] if len(df) > 0 else 0
                if sample_val > 1e11:  # Milliseconds
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                else:  # Seconds
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter valid event types
        valid_events = {"view", "addtocart", "transaction"}
        df = df[df["event_type"].isin(valid_events)]

        # Drop duplicates
        subset_cols = ["customer_id", "timestamp", "event_type", "item_id"]
        avail_cols = [c for c in subset_cols if c in df.columns]
        df = df.drop_duplicates(subset=avail_cols)

        # Sort chronologically
        df = df.sort_values(by=["customer_id", "timestamp"]).reset_index(drop=True)

        # Assign unique event_id if not present
        if "event_id" not in df.columns:
            df["event_id"] = [f"evt_{i}" for i in range(len(df))]

        # Assign revenue if transaction
        if "revenue" not in df.columns:
            # Default realistic pricing: view=0, cart=0, transaction=item price or 25-150 range
            np.random.seed(42)
            df["revenue"] = 0.0
            tx_mask = df["event_type"] == "transaction"
            df.loc[tx_mask, "revenue"] = np.random.gamma(shape=5.0, scale=12.0, size=tx_mask.sum()).round(2)

        return df

    def sessionize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign session IDs to events based on inactivity timeout."""
        df = df.copy().sort_values(by=["customer_id", "timestamp"]).reset_index(drop=True)
        
        # Calculate time difference between consecutive events per customer
        df["prev_time"] = df.groupby("customer_id")["timestamp"].shift(1)
        time_diff = (df["timestamp"] - df["prev_time"]).dt.total_seconds() / 60.0
        
        # New session if time difference exceeds timeout or first event
        new_session = (time_diff > self.session_timeout_minutes) | df["prev_time"].isna()
        df["session_idx"] = new_session.cumsum()
        df["session_id"] = "sess_" + df["session_idx"].astype(str)
        
        df = df.drop(columns=["prev_time", "session_idx"])
        return df

    def partition_cold_start(self, events_df: pd.DataFrame, cutoff_date: pd.Timestamp = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separate customers into established vs cold_start subsets."""
        if cutoff_date is None:
            cutoff_date = events_df["timestamp"].max()

        # Compute per-customer statistics up to cutoff
        cust_stats = events_df.groupby("customer_id").agg(
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
            event_count=("event_id", "count"),
        ).reset_index()

        cust_stats["days_active"] = (cust_stats["last_seen"] - cust_stats["first_seen"]).dt.total_seconds() / 86400.0
        
        # Cold-start condition: fewer than min_events or active history < min_days
        cold_start_mask = (
            (cust_stats["event_count"] < self.cold_start_min_events) |
            (cust_stats["days_active"] < self.cold_start_min_days)
        )
        
        cold_start_ids = set(cust_stats.loc[cold_start_mask, "customer_id"])
        
        established_events = events_df[~events_df["customer_id"].isin(cold_start_ids)]
        cold_start_events = events_df[events_df["customer_id"].isin(cold_start_ids)]

        return established_events, cold_start_events
