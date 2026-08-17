"""Prediction and explainability tools for AI Analyst Agent."""

from typing import Dict, Any, List
from backend.app.database.session import SessionLocal
from backend.app.database.models import Prediction, Customer


class PredictionTools:
    @staticmethod
    def get_churn_analysis(customer_id: str) -> Dict[str, Any]:
        """Fetch churn prediction, risk level, decision threshold, and SHAP drivers for a customer."""
        db = SessionLocal()
        try:
            pred = db.query(Prediction).filter(
                Prediction.customer_id == customer_id,
                Prediction.model_type == "churn"
            ).first()
            if not pred:
                return {"error": f"No churn prediction found for customer '{customer_id}'."}

            return {
                "customer_id": customer_id,
                "predicted_class": pred.predicted_class,
                "predicted_probability": pred.predicted_probability,
                "decision_threshold": pred.decision_threshold,
                "confidence_interval_95": [pred.confidence_interval_low, pred.confidence_interval_high],
                "top_shap_factors": pred.shap_values,
            }
        finally:
            db.close()

    @staticmethod
    def get_next_event(customer_id: str) -> Dict[str, Any]:
        """Fetch predicted next action (VIEW / CART / BUY / INACTIVE) and probability distribution."""
        db = SessionLocal()
        try:
            pred = db.query(Prediction).filter(
                Prediction.customer_id == customer_id,
                Prediction.model_type == "next_event"
            ).first()
            if not pred:
                return {"error": f"No next-event prediction found for customer '{customer_id}'."}

            return {
                "customer_id": customer_id,
                "predicted_event": pred.predicted_class,
                "predicted_probability": pred.predicted_probability,
                "class_probabilities": pred.shap_values,
            }
        finally:
            db.close()
