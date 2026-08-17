"""Cold-start customer prediction and rule-based handling for CustomerPulse AI.
Ensures new customers (<14-30 days or <3 events) are not scored with zero-imputed garbage features.
"""

from typing import Dict, Any, List
import pandas as pd


class ColdStartHandler:
    """Handles inference and baseline heuristics for cold-start customers."""

    @staticmethod
    def is_cold_start(row: pd.Series) -> bool:
        """Check if customer meets cold-start criteria."""
        return bool(row.get("is_cold_start", False)) or int(row.get("frequency_90d", 0)) < 3

    @staticmethod
    def predict_cold_start(row: pd.Series) -> Dict[str, Any]:
        """Generate rule-based prediction for cold-start customer without pseudo-feature imputation."""
        views_7d = int(row.get("views_7d", 0))
        carts_7d = int(row.get("carts_7d", 0))
        
        # Rule-based next event estimation
        if carts_7d > 0:
            next_event = "ADD_TO_CART"
            next_prob = 0.55
        elif views_7d > 2:
            next_event = "VIEW"
            next_prob = 0.65
        else:
            next_event = "INACTIVE"
            next_prob = 0.50

        return {
            "is_cold_start": True,
            "status": "INSUFFICIENT_HISTORY",
            "churn_probability": 0.40,
            "churn_prediction": "UNCERTAIN_COLD_START",
            "next_event_prediction": next_event,
            "next_event_probability": next_prob,
            "confidence_band": "LOW",
            "decision_rule": "New customer heuristic: insufficient historical events for ML inference",
        }
