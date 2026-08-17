"""Dataset Adapter for Canonical Entity Representation."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class DatasetAdapter:
    """Transforms arbitrary user dataframes into internal canonical schema
    while preserving all extra attributes in metadata.
    """

    CANONICAL_COLUMNS = [
        "entity_id",
        "customer_id",
        "transaction_id",
        "entity_type",
        "timestamp",
        "event_type",
        "product_id",
        "category",
        "revenue",
        "quantity",
        "rating",
        "returned",
        "city",
        "session_id",
        "treatment",
        "conversion",
        "churn_label",
        "metadata",
    ]

    @classmethod
    def transform_to_canonical(
        cls,
        df: pd.DataFrame,
        column_mapping: Dict[str, str],  # col_name -> semantic_type
        dataset_mode: str = "CUSTOMER_EVENT",
        domain: str = "CUSTOMER",
    ) -> pd.DataFrame:
        """Create normalized canonical dataframe from raw tabular input."""
        reverse_mapping: Dict[str, str] = {v: k for k, v in column_mapping.items()}

        # Identify key source columns
        col_cust = (
            reverse_mapping.get("CUSTOMER_ID")
            or reverse_mapping.get("USER_ID")
            or reverse_mapping.get("ACCOUNT_ID")
            or reverse_mapping.get("ENTITY_ID")
        )
        col_tx = reverse_mapping.get("TRANSACTION_ID") or reverse_mapping.get("ORDER_ID")
        col_time = reverse_mapping.get("TIMESTAMP") or reverse_mapping.get("DATE") or reverse_mapping.get("ORDER_DATE")
        col_event = reverse_mapping.get("EVENT_TYPE")
        col_rev = (
            reverse_mapping.get("REVENUE")
            or reverse_mapping.get("SALES_AMOUNT")
            or reverse_mapping.get("ORDER_VALUE")
            or reverse_mapping.get("PRICE")
            or reverse_mapping.get("AMOUNT")
        )
        col_prod = reverse_mapping.get("PRODUCT_ID") or reverse_mapping.get("ITEM_ID")
        col_cat = reverse_mapping.get("PRODUCT_CATEGORY") or reverse_mapping.get("CATEGORY") or reverse_mapping.get("DEPARTMENT")
        col_qty = reverse_mapping.get("QUANTITY") or reverse_mapping.get("QTY")
        col_rate = reverse_mapping.get("RATING")
        col_ret = reverse_mapping.get("RETURNED")
        col_loc = reverse_mapping.get("COUNTRY") or reverse_mapping.get("LOCATION") or reverse_mapping.get("CITY")
        col_sess = reverse_mapping.get("SESSION_ID")
        col_treat = reverse_mapping.get("TREATMENT")
        col_conv = reverse_mapping.get("CONVERSION") or reverse_mapping.get("OUTCOME")
        col_churn = reverse_mapping.get("CHURN_LABEL") or reverse_mapping.get("TARGET")

        # Entity type detection
        if domain == "HEALTHCARE":
            entity_type = "patient"
        elif domain == "MARKETING_CAMPAIGN":
            entity_type = "campaign_target"
        elif domain == "SALES":
            entity_type = "sales_record"
        else:
            entity_type = "customer"

        canonical_df = pd.DataFrame()

        # 1. Entity ID
        if col_cust and col_cust in df.columns:
            canonical_df["entity_id"] = df[col_cust].astype(str)
        else:
            canonical_df["entity_id"] = [f"entity_{i+1}" for i in range(len(df))]

        canonical_df["customer_id"] = canonical_df["entity_id"]
        canonical_df["entity_type"] = entity_type

        # 2. Transaction ID
        if col_tx and col_tx in df.columns:
            canonical_df["transaction_id"] = df[col_tx].astype(str)
        else:
            canonical_df["transaction_id"] = [f"tx_{i+1:08d}" for i in range(len(df))]

        # 3. Timestamp
        if col_time and col_time in df.columns:
            canonical_df["timestamp"] = pd.to_datetime(df[col_time], errors="coerce")
            if canonical_df["timestamp"].isnull().any():
                canonical_df["timestamp"] = canonical_df["timestamp"].fillna(pd.Timestamp.utcnow())
        else:
            canonical_df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="15min")

        # 4. Event Type
        if col_event and col_event in df.columns:
            canonical_df["event_type"] = df[col_event].astype(str).str.lower()
        else:
            canonical_df["event_type"] = "transaction" if col_rev else "interaction"

        # 5. Revenue / Monetary Value
        if col_rev and col_rev in df.columns:
            canonical_df["revenue"] = pd.to_numeric(df[col_rev], errors="coerce").fillna(0.0)
        else:
            canonical_df["revenue"] = 0.0

        # 6. Quantity, Product & Category
        canonical_df["product_id"] = df[col_prod].astype(str) if col_prod and col_prod in df.columns else "item_general"
        canonical_df["category"] = df[col_cat].astype(str) if col_cat and col_cat in df.columns else "General"
        canonical_df["quantity"] = pd.to_numeric(df[col_qty], errors="coerce").fillna(1).astype(int) if col_qty and col_qty in df.columns else 1
        canonical_df["rating"] = pd.to_numeric(df[col_rate], errors="coerce").fillna(5.0) if col_rate and col_rate in df.columns else 5.0
        canonical_df["returned"] = pd.to_numeric(df[col_ret], errors="coerce").fillna(0).astype(int) if col_ret and col_ret in df.columns else 0
        canonical_df["city"] = df[col_loc].astype(str) if col_loc and col_loc in df.columns else "Urban"

        # 7. Treatment & Conversion
        if col_treat and col_treat in df.columns:
            canonical_df["treatment"] = pd.to_numeric(df[col_treat], errors="coerce").fillna(0).astype(int)
        else:
            canonical_df["treatment"] = 0

        if col_conv and col_conv in df.columns:
            canonical_df["conversion"] = pd.to_numeric(df[col_conv], errors="coerce").fillna(0).astype(int)
        else:
            canonical_df["conversion"] = 0

        if col_churn and col_churn in df.columns:
            canonical_df["churn_label"] = pd.to_numeric(df[col_churn], errors="coerce").fillna(0).astype(int)
        else:
            canonical_df["churn_label"] = None

        # 8. Extra unmapped columns in metadata
        mapped_cols = set(column_mapping.keys())
        extra_cols = [c for c in df.columns if c not in mapped_cols]
        if extra_cols:
            canonical_df["metadata"] = df[extra_cols].to_dict(orient="records")
        else:
            canonical_df["metadata"] = [{} for _ in range(len(df))]

        return canonical_df
