"""Leakage-safe Customer 360 Feature Engineering for CustomerPulse AI.
Calculates windowed (7d, 30d, 90d), RFM, funnel conversion, engagement velocity,
and category preference features strictly using historical events up to cutoff timestamp T.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import entropy

from backend.app.database.session import SessionLocal
from backend.app.database.models import Customer, Event, CustomerFeature


class FeatureEngineer:
    """Computes point-in-time leakage-safe Customer 360 features."""

    def __init__(self, cold_start_min_events: int = 3, cold_start_min_days: int = 14):
        self.cold_start_min_events = cold_start_min_events
        self.cold_start_min_days = cold_start_min_days

    def compute_customer_features(
        self,
        events_df: pd.DataFrame,
        cutoff_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Extract leakage-safe feature vector for every customer as of cutoff_date.
        Only events with timestamp <= cutoff_date are considered.
        """
        df = events_df.copy()
        if cutoff_date is None:
            cutoff_date = df["timestamp"].max()

        # Strict point-in-time filter
        df = df[df["timestamp"] <= cutoff_date].copy()
        if df.empty:
            return pd.DataFrame()

        # 7d, 30d, 90d window thresholds
        t_7d = cutoff_date - timedelta(days=7)
        t_30d = cutoff_date - timedelta(days=30)
        t_90d = cutoff_date - timedelta(days=90)

        # Slice window events
        df_7d = df[df["timestamp"] >= t_7d]
        df_30d = df[df["timestamp"] >= t_30d]
        df_90d = df[df["timestamp"] >= t_90d]

        # Aggregate per customer
        all_customers = df["customer_id"].unique()
        records = []

        # Pre-group data for speed
        grp_all = df.groupby("customer_id")
        grp_7d = df_7d.groupby("customer_id")
        grp_30d = df_30d.groupby("customer_id")
        grp_90d = df_90d.groupby("customer_id")

        for cid in all_customers:
            cust_events = grp_all.get_group(cid)
            first_seen = cust_events["timestamp"].min()
            last_seen = cust_events["timestamp"].max()
            
            # Recency
            recency_days = max(0.0, (cutoff_date - last_seen).total_seconds() / 86400.0)
            tenure_days = max(0.1, (cutoff_date - first_seen).total_seconds() / 86400.0)
            active_days = max(0.1, (last_seen - first_seen).total_seconds() / 86400.0)
            
            total_events = len(cust_events)
            is_cold_start = (total_events < self.cold_start_min_events) or (active_days < self.cold_start_min_days)

            # Frequency & Counts
            c7 = grp_7d.get_group(cid) if cid in grp_7d.groups else pd.DataFrame()
            c30 = grp_30d.get_group(cid) if cid in grp_30d.groups else pd.DataFrame()
            c90 = grp_90d.get_group(cid) if cid in grp_90d.groups else pd.DataFrame()

            freq_7d = len(c7)
            freq_30d = len(c30)
            freq_90d = len(c90)

            views_7d = int((c7["event_type"] == "view").sum()) if not c7.empty else 0
            views_30d = int((c30["event_type"] == "view").sum()) if not c30.empty else 0
            views_90d = int((c90["event_type"] == "view").sum()) if not c90.empty else 0

            carts_7d = int((c7["event_type"] == "addtocart").sum()) if not c7.empty else 0
            carts_30d = int((c30["event_type"] == "addtocart").sum()) if not c30.empty else 0
            carts_90d = int((c90["event_type"] == "addtocart").sum()) if not c90.empty else 0

            txs_7d = int((c7["event_type"] == "transaction").sum()) if not c7.empty else 0
            txs_30d = int((c30["event_type"] == "transaction").sum()) if not c30.empty else 0
            txs_90d = int((c90["event_type"] == "transaction").sum()) if not c90.empty else 0

            # Monetary
            monetary_total = float(cust_events["revenue"].sum())
            monetary_30d = float(c30["revenue"].sum()) if not c30.empty else 0.0
            total_purchases = int((cust_events["event_type"] == "transaction").sum())
            aov = round(monetary_total / max(1, total_purchases), 2) if total_purchases > 0 else 0.0

            # Funnel & Velocity
            cart_to_view = round(carts_90d / max(1, views_90d), 4)
            tx_to_cart = round(txs_90d / max(1, carts_90d), 4)
            conv_rate = round(txs_90d / max(1, views_90d), 4)

            # Purchase interval
            tx_events = cust_events[cust_events["event_type"] == "transaction"].sort_values("timestamp")
            if len(tx_events) >= 2:
                intervals = tx_events["timestamp"].diff().dt.total_seconds().dropna() / 86400.0
                purchase_interval = float(intervals.mean())
            else:
                purchase_interval = float(recency_days)

            # Velocity ratio (recent 7d daily rate vs 30d baseline daily rate)
            rate_7d = freq_7d / 7.0
            rate_30d = freq_30d / 30.0
            velocity_7d_30d = round(rate_7d / max(0.01, rate_30d), 3)

            # Engagement velocity (views 7d vs views 30d)
            view_rate_7d = views_7d / 7.0
            view_rate_30d = views_30d / 30.0
            engagement_velocity = round(view_rate_7d - view_rate_30d, 3)

            # Activity Trend slope over 4 weekly buckets
            w1 = len(cust_events[(cust_events["timestamp"] >= cutoff_date - timedelta(days=28)) & (cust_events["timestamp"] < cutoff_date - timedelta(days=21))])
            w2 = len(cust_events[(cust_events["timestamp"] >= cutoff_date - timedelta(days=21)) & (cust_events["timestamp"] < cutoff_date - timedelta(days=14))])
            w3 = len(cust_events[(cust_events["timestamp"] >= cutoff_date - timedelta(days=14)) & (cust_events["timestamp"] < cutoff_date - timedelta(days=7))])
            w4 = freq_7d
            weekly_counts = np.array([w1, w2, w3, w4], dtype=float)
            x_axis = np.array([1, 2, 3, 4], dtype=float)
            # Linear regression slope: cov(x, y) / var(x)
            trend_slope = float(np.polyfit(x_axis, weekly_counts, 1)[0]) if len(weekly_counts) == 4 else 0.0

            # Category Preferences & Diversity Entropy
            cat_counts = cust_events["category_id"].dropna().value_counts()
            top_cat = str(cat_counts.index[0]) if not cat_counts.empty else "general"
            if len(cat_counts) > 1:
                probs = cat_counts.values / cat_counts.values.sum()
                cat_entropy = float(entropy(probs, base=2))
            else:
                cat_entropy = 0.0

            unique_items = int(cust_events["item_id"].nunique())

            record = {
                "customer_id": cid,
                "as_of_date": cutoff_date,
                "recency_days": round(recency_days, 1),
                "frequency_7d": freq_7d,
                "frequency_30d": freq_30d,
                "frequency_90d": freq_90d,
                "monetary_total": round(monetary_total, 2),
                "monetary_30d": round(monetary_30d, 2),
                "aov": aov,
                "views_7d": views_7d,
                "views_30d": views_30d,
                "views_90d": views_90d,
                "carts_7d": carts_7d,
                "carts_30d": carts_30d,
                "carts_90d": carts_90d,
                "transactions_7d": txs_7d,
                "transactions_30d": txs_30d,
                "transactions_90d": txs_90d,
                "cart_to_view_ratio": cart_to_view,
                "purchase_to_cart_ratio": tx_to_cart,
                "conversion_rate": conv_rate,
                "purchase_interval_days": round(purchase_interval, 1),
                "velocity_7d_30d": velocity_7d_30d,
                "engagement_velocity": engagement_velocity,
                "trend_slope": round(trend_slope, 3),
                "top_category": top_cat,
                "category_entropy": round(cat_entropy, 3),
                "unique_items_viewed": unique_items,
                "is_cold_start": is_cold_start,
            }
            records.append(record)

        feature_df = pd.DataFrame(records)
        return feature_df

    def save_features_to_db(self, feature_df: pd.DataFrame):
        """Persist computed features to customer_features table."""
        db = SessionLocal()
        try:
            db.query(CustomerFeature).delete()
            db.commit()

            objs = [
                CustomerFeature(
                    customer_id=row["customer_id"],
                    as_of_date=row["as_of_date"],
                    recency_days=float(row["recency_days"]),
                    frequency_7d=int(row["frequency_7d"]),
                    frequency_30d=int(row["frequency_30d"]),
                    frequency_90d=int(row["frequency_90d"]),
                    monetary_total=float(row["monetary_total"]),
                    monetary_30d=float(row["monetary_30d"]),
                    aov=float(row["aov"]),
                    views_7d=int(row["views_7d"]),
                    views_30d=int(row["views_30d"]),
                    views_90d=int(row["views_90d"]),
                    carts_7d=int(row["carts_7d"]),
                    carts_30d=int(row["carts_30d"]),
                    carts_90d=int(row["carts_90d"]),
                    transactions_7d=int(row["transactions_7d"]),
                    transactions_30d=int(row["transactions_30d"]),
                    transactions_90d=int(row["transactions_90d"]),
                    cart_to_view_ratio=float(row["cart_to_view_ratio"]),
                    purchase_to_cart_ratio=float(row["purchase_to_cart_ratio"]),
                    conversion_rate=float(row["conversion_rate"]),
                    purchase_interval_days=float(row["purchase_interval_days"]),
                    velocity_7d_30d=float(row["velocity_7d_30d"]),
                    engagement_velocity=float(row["engagement_velocity"]),
                    trend_slope=float(row["trend_slope"]),
                    top_category=str(row["top_category"]),
                    category_entropy=float(row["category_entropy"]),
                    unique_items_viewed=int(row["unique_items_viewed"]),
                    is_cold_start=bool(row["is_cold_start"]),
                )
                for _, row in feature_df.iterrows()
            ]
            db.bulk_save_objects(objs)
            db.commit()
            print(f"Saved {len(objs):,} customer feature records to database.")
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
