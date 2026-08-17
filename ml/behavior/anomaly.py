"""Anomaly detection and Opportunity Scoring for CustomerPulse AI.
Implements Isolation Forest anomaly detection on behavioral trajectories,
and computes composite Opportunity Scores broken into auditable sub-components.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Detects multi-variate behavioral anomalies using Isolation Forest."""

    FEATURE_COLS = [
        "recency_days",
        "frequency_30d",
        "velocity_7d_30d",
        "cart_to_view_ratio",
        "trend_slope",
    ]

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def detect_anomalies(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Score customers for behavioral anomaly severity."""
        df = features_df.copy()
        X = df[self.FEATURE_COLS].fillna(0).values
        
        preds = self.model.fit_predict(X)
        scores = self.model.decision_function(X)

        # Map scores to 0-1 anomaly score (lower decision function = more anomalous)
        min_s, max_s = scores.min(), scores.max()
        norm_scores = 1.0 - ((scores - min_s) / max(1e-5, max_s - min_s))

        df["is_anomaly"] = preds == -1
        df["anomaly_score"] = norm_scores.round(3)
        return df


class OpportunityScoreEngine:
    """Computes transparent, composite Opportunity Scores (0-100) from 5 sub-components:
    - Customer Value (30%)
    - Behavioral Deterioration (25%)
    - Predicted Intent (20%)
    - Intervention Responsiveness (15%)
    - Time Sensitivity (10%)
    """

    @staticmethod
    def compute_opportunity_scores(
        features_df: pd.DataFrame,
        uplift_df: pd.DataFrame = None,
        churn_probs: Dict[str, float] = None,
    ) -> pd.DataFrame:
        """Compute composite Opportunity Score and all 5 auditable sub-components."""
        df = features_df.copy()

        # 1. Customer Value Sub-Score (0 to 30 pts)
        max_monetary = max(1.0, df["monetary_total"].quantile(0.95))
        monetary_norm = np.clip(df["monetary_total"] / max_monetary, 0, 1)
        freq_norm = np.clip(df["frequency_90d"] / max(1.0, df["frequency_90d"].quantile(0.95)), 0, 1)
        value_score = (0.7 * monetary_norm + 0.3 * freq_norm) * 30.0

        # 2. Behavioral Deterioration Sub-Score (0 to 25 pts)
        rec_norm = np.clip(df["recency_days"] / 60.0, 0, 1)
        vel_drop = np.clip(1.0 - (df["velocity_7d_30d"] / 1.5), 0, 1)
        trend_drop = np.clip(-df["trend_slope"] / 3.0, 0, 1)
        deterioration_score = (0.4 * rec_norm + 0.3 * vel_drop + 0.3 * trend_drop) * 25.0

        # 3. Predicted Intent Sub-Score (0 to 20 pts)
        cart_norm = np.clip(df["cart_to_view_ratio"] * 3.0, 0, 1)
        views_norm = np.clip(df["views_7d"] / max(1.0, df["views_7d"].quantile(0.95)), 0, 1)
        intent_score = (0.6 * cart_norm + 0.4 * views_norm) * 20.0

        # 4. Intervention Responsiveness (0 to 15 pts)
        if uplift_df is not None and not uplift_df.empty:
            uplift_map = uplift_df.set_index("customer_id")["estimated_uplift"].to_dict()
            uplift_vals = df["customer_id"].map(uplift_map).fillna(0.05)
            uplift_norm = np.clip(uplift_vals / 0.15, 0, 1)
        else:
            uplift_norm = np.clip(df["cart_to_view_ratio"] * 2.0 + 0.2, 0, 1)
        responsiveness_score = uplift_norm * 15.0

        # 5. Time Sensitivity (0 to 10 pts)
        # Urgency peaks when customer is declining/at-risk (recency 14-35 days) before going dormant
        urgency = np.exp(-((df["recency_days"] - 21.0) ** 2) / (2 * (10.0 ** 2)))
        sensitivity_score = urgency * 10.0

        # Total Opportunity Score
        total_opp_score = value_score + deterioration_score + intent_score + responsiveness_score + sensitivity_score

        df["subscore_value"] = value_score.round(1)
        df["subscore_deterioration"] = deterioration_score.round(1)
        df["subscore_intent"] = intent_score.round(1)
        df["subscore_responsiveness"] = responsiveness_score.round(1)
        df["subscore_time_sensitivity"] = sensitivity_score.round(1)
        df["opportunity_score"] = np.clip(total_opp_score.round(1), 0.0, 100.0)

        # Priority tier
        df["opportunity_tier"] = pd.cut(
            df["opportunity_score"],
            bins=[-1, 35, 60, 80, 100],
            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        )

        return df
