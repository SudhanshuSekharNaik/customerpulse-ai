"""Next-Best-Action recommendation service and offline backtest."""

from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.database.models import Recommendation, Customer
from ml.recommendation.offline_policy_eval import OfflinePolicyEvaluator


class RecommendationService:
    @staticmethod
    def get_recommendations(
        db: Session,
        limit: int = 50,
        action_type: str = None,
        confidence: str = None,
    ) -> List[Dict[str, Any]]:
        query = db.query(Recommendation, Customer).join(Customer, Customer.customer_id == Recommendation.customer_id)
        if action_type:
            query = query.filter(Recommendation.action_type == action_type)
        if confidence:
            query = query.filter(Recommendation.confidence_level == confidence)

        recs = query.order_by(desc(Recommendation.score)).limit(limit).all()

        return [
            {
                "recommendation_id": r.recommendation_id,
                "customer_id": r.customer_id,
                "email": c.email,
                "action_type": r.action_type,
                "rank": r.rank,
                "score": r.score,
                "what_text": r.what_text,
                "why_text": r.why_text,
                "evidence": r.evidence,
                "expected_impact": r.expected_impact,
                "confidence_level": r.confidence_level,
                "status": r.status,
                "created_at": r.created_at,
            }
            for r, c in recs
        ]

    @staticmethod
    def get_offline_policy_backtest(db: Session) -> Dict[str, Any]:
        recs = db.query(Recommendation).all()
        rec_dicts = [
            {
                "action_type": r.action_type,
                "expected_impact": r.expected_impact,
                "score": r.score,
            }
            for r in recs
        ]
        evaluator = OfflinePolicyEvaluator()
        return evaluator.evaluate_policy(rec_dicts)

    @classmethod
    def get_offline_backtest(cls, db: Session) -> Dict[str, Any]:
        return cls.get_offline_policy_backtest(db)
