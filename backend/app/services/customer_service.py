"""Customer 360 query and retrieval service."""

from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.database.models import Customer, Event, CustomerFeature, CustomerSegment, CustomerState, Prediction, UpliftPrediction, Recommendation


class CustomerService:
    @staticmethod
    def get_customers(
        db: Session,
        limit: int = 50,
        offset: int = 0,
        state: Optional[str] = None,
        segment_id: Optional[int] = None,
        min_revenue: Optional[float] = None,
        churn_risk_only: bool = False,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        query = db.query(Customer)

        if search:
            query = query.filter(Customer.customer_id.contains(search) | Customer.email.contains(search))
        if min_revenue is not None:
            query = query.filter(Customer.total_revenue >= min_revenue)
        if state:
            query = query.join(CustomerState).filter(CustomerState.current_state == state)
        if segment_id is not None:
            query = query.join(CustomerSegment).filter(CustomerSegment.segment_id == segment_id)

        total_count = query.count()
        customers = query.order_by(desc(Customer.total_revenue)).offset(offset).limit(limit).all()

        results = []
        for c in customers:
            churn_pred = next((p for p in c.predictions if p.model_type == "churn"), None)
            churn_prob = churn_pred.predicted_probability if churn_pred else None
            
            # Opportunity score estimation
            opp_score = 65.0
            if c.features:
                opp_score = min(99.0, (c.features.monetary_total / 100.0) + (c.features.frequency_30d * 3.0))

            results.append({
                "customer_id": c.customer_id,
                "visitor_id": c.visitor_id,
                "email": c.email,
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
                "total_events": c.total_events,
                "total_revenue": round(c.total_revenue, 2),
                "total_orders": c.total_orders,
                "is_cold_start": c.is_cold_start,
                "current_state": c.state.current_state if c.state else "UNKNOWN",
                "segment_label": c.segment.segment_label if c.segment else "Unassigned",
                "churn_probability": churn_prob,
                "opportunity_score": round(opp_score, 1),
            })

        return {
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "items": results,
        }

    @staticmethod
    def get_customer_360_detail(db: Session, customer_id: str) -> Optional[Dict[str, Any]]:
        customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
        if not customer:
            return None

        recent_events = db.query(Event).filter(Event.customer_id == customer_id).order_by(desc(Event.timestamp)).limit(25).all()
        churn_pred = next((p for p in customer.predictions if p.model_type == "churn"), None)
        nxt_pred = next((p for p in customer.predictions if p.model_type == "next_event"), None)
        uplift_rec = customer.uplift_predictions[0] if customer.uplift_predictions else None
        top_rec = customer.recommendations[0] if customer.recommendations else None

        feat_dict = None
        if customer.features:
            f = customer.features
            feat_dict = {
                "recency_days": f.recency_days,
                "frequency_7d": f.frequency_7d,
                "frequency_30d": f.frequency_30d,
                "frequency_90d": f.frequency_90d,
                "monetary_total": f.monetary_total,
                "monetary_30d": f.monetary_30d,
                "aov": f.aov,
                "views_7d": f.views_7d,
                "views_30d": f.views_30d,
                "views_90d": f.views_90d,
                "carts_7d": f.carts_7d,
                "carts_30d": f.carts_30d,
                "carts_90d": f.carts_90d,
                "transactions_7d": f.transactions_7d,
                "transactions_30d": f.transactions_30d,
                "transactions_90d": f.transactions_90d,
                "cart_to_view_ratio": f.cart_to_view_ratio,
                "conversion_rate": f.conversion_rate,
                "velocity_7d_30d": f.velocity_7d_30d,
                "engagement_velocity": f.engagement_velocity,
                "trend_slope": f.trend_slope,
                "top_category": f.top_category,
                "category_entropy": f.category_entropy,
                "unique_items_viewed": f.unique_items_viewed,
                "is_cold_start": f.is_cold_start,
            }

        return {
            "customer_id": customer.customer_id,
            "visitor_id": customer.visitor_id,
            "email": customer.email,
            "first_seen": customer.first_seen,
            "last_seen": customer.last_seen,
            "total_events": customer.total_events,
            "total_revenue": round(customer.total_revenue, 2),
            "total_orders": customer.total_orders,
            "is_cold_start": customer.is_cold_start,
            "features": feat_dict,
            "current_state": customer.state.current_state if customer.state else "UNKNOWN",
            "previous_state": customer.state.previous_state if customer.state else None,
            "segment_label": customer.segment.segment_label if customer.segment else "Unassigned",
            "segment_id": customer.segment.segment_id if customer.segment else None,
            "churn_prediction": {
                "predicted_class": churn_pred.predicted_class if churn_pred else "UNKNOWN",
                "predicted_probability": churn_pred.predicted_probability if churn_pred else 0.0,
                "decision_threshold": churn_pred.decision_threshold if churn_pred else 0.5,
                "shap_values": churn_pred.shap_values if churn_pred else {},
                "ci_low": churn_pred.confidence_interval_low if churn_pred else None,
                "ci_high": churn_pred.confidence_interval_high if churn_pred else None,
            } if churn_pred else None,
            "next_event_prediction": {
                "predicted_event": nxt_pred.predicted_class if nxt_pred else "UNKNOWN",
                "predicted_probability": nxt_pred.predicted_probability if nxt_pred else 0.0,
            } if nxt_pred else None,
            "uplift_estimate": {
                "estimated_uplift": uplift_rec.estimated_uplift if uplift_rec else 0.0,
                "decile": uplift_rec.uplift_decile if uplift_rec else 5,
                "ci_low": uplift_rec.confidence_interval_low if uplift_rec else 0.0,
                "ci_high": uplift_rec.confidence_interval_high if uplift_rec else 0.0,
                "model_used": uplift_rec.model_used if uplift_rec else "X_LEARNER",
                "label": "Estimated treatment uplift",
            } if uplift_rec else None,
            "top_recommendation": {
                "action_type": top_rec.action_type if top_rec else "NO_ACTION",
                "what": top_rec.what_text if top_rec else "Maintain standard organic flow",
                "why": top_rec.why_text if top_rec else "No critical risk trigger",
                "evidence": top_rec.evidence if top_rec else {},
                "expected_impact": top_rec.expected_impact if top_rec else 0.0,
                "confidence": top_rec.confidence_level if top_rec else "HIGH",
            } if top_rec else None,
            "recent_events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "item_id": e.item_id,
                    "category_id": e.category_id,
                    "timestamp": e.timestamp,
                    "revenue": e.revenue,
                }
                for e in recent_events
            ],
        }
