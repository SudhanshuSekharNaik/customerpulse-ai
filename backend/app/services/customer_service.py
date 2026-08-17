import os
import json
import math
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
            is_cold = bool(c.is_cold_start or (churn_pred and churn_pred.predicted_class in ["COLD_START_UNCERTAIN", "UNCERTAIN_COLD_START"]))
            churn_prob = churn_pred.predicted_probability if (churn_pred and not is_cold) else None
            
            # Opportunity score computation (0-100 scale, non-capped, well-differentiated)
            # Combines Value (Spend + Orders), Urgency (Churn risk / recency), and Engagement Velocity
            spend = float(c.total_revenue or 0.0)
            orders = int(c.total_orders or 1)
            churn_factor = float(churn_prob) if churn_prob is not None else 0.25

            # 1. Value sub-score (0 to 40 pts) using continuous logarithmic scaling so 45k vs 60k differ clearly
            value_pts = min(40.0, max(5.0, (math.log1p(spend) / math.log1p(100000.0)) * 35.0 + min(5.0, orders * 0.8)))

            # 2. Risk & Urgency sub-score (0 to 35 pts) - high spend + higher churn risk = higher immediate intervention opportunity
            risk_urgency_pts = churn_factor * 35.0

            # 3. Activity / Engagement sub-score (0 to 25 pts)
            recency = float(c.features.recency_days) if (c.features and c.features.recency_days is not None) else 15.0
            recency_urgency = math.exp(-((recency - 20.0) ** 2) / 300.0) * 15.0
            freq_pts = min(10.0, (c.features.frequency_30d * 2.0) if c.features else 2.0)
            engagement_pts = recency_urgency + freq_pts

            opp_score = round(min(98.5, max(12.0, value_pts + risk_urgency_pts + engagement_pts)), 1)

            results.append({
                "customer_id": c.customer_id,
                "visitor_id": c.visitor_id,
                "email": c.email,
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
                "total_events": c.total_events,
                "total_revenue": round(c.total_revenue, 2),
                "total_orders": c.total_orders,
                "is_cold_start": is_cold,
                "current_state": c.state.current_state if c.state else "UNKNOWN",
                "segment_label": c.segment.segment_label if c.segment else "Unassigned",
                "churn_probability": churn_prob,
                "opportunity_score": opp_score,
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

        is_cold = bool(customer.is_cold_start or (churn_pred and churn_pred.predicted_class in ["COLD_START_UNCERTAIN", "UNCERTAIN_COLD_START"]))

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
                "is_cold_start": is_cold,
            }

        # Uplift model verification
        meta_path = "ml/models/uplift_model_metadata.json"
        has_uplift_model = os.path.exists(meta_path)
        if uplift_rec and has_uplift_model:
            uplift_dict = {
                "is_available": True,
                "estimated_uplift": uplift_rec.estimated_uplift,
                "decile": uplift_rec.uplift_decile,
                "ci_low": uplift_rec.confidence_interval_low,
                "ci_high": uplift_rec.confidence_interval_high,
                "model_used": uplift_rec.model_used,
                "label": "Estimated treatment uplift",
            }
        else:
            uplift_dict = {
                "is_available": False,
                "estimated_uplift": None,
                "decile": None,
                "status_text": "Not available for this dataset",
                "reason": "Uplift modeling requires A/B campaign treatment & control data.",
            }

        # Churn prediction payload with honest cold-start handling
        if churn_pred:
            if is_cold:
                churn_dict = {
                    "is_cold_start": True,
                    "status_text": "New customer — not enough history for a prediction yet",
                    "predicted_class": "NEW_CUSTOMER",
                    "predicted_probability": None,
                    "decision_threshold": churn_pred.decision_threshold or 0.50,
                    "shap_values": {},
                    "ci_low": None,
                    "ci_high": None,
                }
            else:
                churn_dict = {
                    "is_cold_start": False,
                    "status_text": "Active ML prediction",
                    "predicted_class": churn_pred.predicted_class,
                    "predicted_probability": churn_pred.predicted_probability,
                    "decision_threshold": churn_pred.decision_threshold or 0.50,
                    "shap_values": churn_pred.shap_values or {},
                    "ci_low": churn_pred.confidence_interval_low,
                    "ci_high": churn_pred.confidence_interval_high,
                }
        else:
            churn_dict = None

        return {
            "customer_id": customer.customer_id,
            "visitor_id": customer.visitor_id,
            "email": customer.email,
            "first_seen": customer.first_seen,
            "last_seen": customer.last_seen,
            "total_events": customer.total_events,
            "total_revenue": round(customer.total_revenue, 2),
            "total_orders": customer.total_orders,
            "is_cold_start": is_cold,
            "features": feat_dict,
            "current_state": customer.state.current_state if customer.state else "UNKNOWN",
            "previous_state": customer.state.previous_state if customer.state else None,
            "segment_label": customer.segment.segment_label if customer.segment else "Unassigned",
            "segment_id": customer.segment.segment_id if customer.segment else None,
            "churn_prediction": churn_dict,
            "next_event_prediction": {
                "predicted_event": nxt_pred.predicted_class if nxt_pred else "UNKNOWN",
                "predicted_probability": nxt_pred.predicted_probability if nxt_pred else 0.0,
            } if nxt_pred else None,
            "uplift_estimate": uplift_dict,
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

