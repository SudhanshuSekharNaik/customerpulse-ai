"""Recommendation and Next-Best-Action domain tools for AI Analyst Agent."""

from typing import Dict, Any, List
from backend.app.database.session import SessionLocal
from backend.app.database.models import Recommendation, UpliftPrediction


class RecommendationTools:
    @staticmethod
    def get_customer_recommendation(customer_id: str) -> Dict[str, Any]:
        """Fetch prioritized Next-Best-Action recommendation card and evidence for customer."""
        db = SessionLocal()
        try:
            rec = db.query(Recommendation).filter(Recommendation.customer_id == customer_id).first()
            if not rec:
                return {"error": f"No recommendation found for customer '{customer_id}'."}

            return {
                "customer_id": customer_id,
                "action_type": rec.action_type,
                "what": rec.what_text,
                "why": rec.why_text,
                "evidence": rec.evidence,
                "expected_impact": rec.expected_impact,
                "confidence_level": rec.confidence_level,
            }
        finally:
            db.close()

    @staticmethod
    def get_customer_uplift(customer_id: str) -> Dict[str, Any]:
        """Fetch causal treatment uplift estimate and 95% bootstrap confidence interval."""
        db = SessionLocal()
        try:
            uplift = db.query(UpliftPrediction).filter(UpliftPrediction.customer_id == customer_id).first()
            if not uplift:
                return {"error": f"No uplift prediction found for customer '{customer_id}'."}

            return {
                "customer_id": customer_id,
                "estimated_uplift": uplift.estimated_uplift,
                "uplift_decile": uplift.uplift_decile,
                "confidence_interval_95": [uplift.confidence_interval_low, uplift.confidence_interval_high],
                "model_used": uplift.model_used,
                "label": "Estimated treatment uplift",
            }
        finally:
            db.close()
