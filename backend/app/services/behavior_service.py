"""Customer State, Markov Transitions, and Behavior Changes service."""

from typing import Dict, Any, List
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.app.database.models import CustomerState, BehaviorChange, CustomerFeature, Customer, Prediction
from ml.behavior.change_detection import STATES


class BehaviorService:
    @staticmethod
    def get_state_distribution(db: Session) -> List[Dict[str, Any]]:
        total_customers = db.query(CustomerState).count()
        states_summary = db.query(
            CustomerState.current_state,
            func.count(CustomerState.customer_id).label("count"),
        ).group_by(CustomerState.current_state).all()

        counts_map = {st: count for st, count in states_summary}
        results = []

        for st in STATES:
            count = counts_map.get(st, 0)
            # Query avg recency & revenue
            feats = db.query(
                func.avg(CustomerFeature.recency_days).label("avg_rec"),
                func.avg(CustomerFeature.monetary_total).label("avg_rev"),
            ).join(CustomerState, CustomerState.customer_id == CustomerFeature.customer_id)\
             .filter(CustomerState.current_state == st).first()

            # Dynamically query mean model churn risk for this state from Prediction table
            churn_stat = db.query(func.avg(Prediction.predicted_probability))\
                .join(CustomerState, CustomerState.customer_id == Prediction.customer_id)\
                .filter(CustomerState.current_state == st, Prediction.model_type == "churn").scalar()

            if churn_stat is not None:
                avg_churn = round(float(churn_stat), 4)
            else:
                avg_rec_val = float(feats.avg_rec or 15.0) if feats else 15.0
                avg_churn = min(0.95, max(0.05, round(avg_rec_val / 120.0, 3)))

            results.append({
                "state": st,
                "customer_count": count,
                "percentage": round((count / max(1, total_customers)) * 100, 1),
                "avg_recency_days": round(float(feats.avg_rec or 0.0), 1) if feats else 0.0,
                "avg_churn_risk": avg_churn,
                "avg_revenue": round(float(feats.avg_rev or 0.0), 2) if feats else 0.0,
            })

        return results

    @staticmethod
    def get_transition_matrix(db: Session) -> Dict[str, Any]:
        """Compute empirical Markov transition probabilities between states."""
        records = db.query(CustomerState.previous_state, CustomerState.current_state).all()
        matrix_counts = {s1: {s2: 0 for s2 in STATES} for s1 in STATES}

        for prev, curr in records:
            p = prev if prev in STATES else None
            c = curr if curr in STATES else None
            if p and c:
                matrix_counts[p][c] += 1

        # Query state customer counts
        states_summary = db.query(
            CustomerState.current_state,
            func.count(CustomerState.customer_id).label("count"),
        ).group_by(CustomerState.current_state).all()
        counts_map = {st: count for st, count in states_summary}

        # Normalize to transition probabilities with strict row-sum invariant
        prob_matrix = []
        for s1 in STATES:
            row_total = sum(matrix_counts[s1].values())
            if row_total > 0:
                row_probs = [
                    round(matrix_counts[s1][s2] / float(row_total), 3)
                    for s2 in STATES
                ]
                # Invariant correction to ensure sum == 1.0
                diff = round(1.0 - sum(row_probs), 3)
                if abs(diff) > 0.0001:
                    max_idx = int(np.argmax(row_probs))
                    row_probs[max_idx] = round(row_probs[max_idx] + diff, 3)
            else:
                # If this state has customers in the dataset, assign self-transition
                state_cust_count = counts_map.get(s1, 0)
                if state_cust_count > 0:
                    row_probs = [1.0 if s2 == s1 else 0.0 for s2 in STATES]
                else:
                    row_probs = [0.0 for _ in STATES]
            prob_matrix.append(row_probs)

        return {
            "states": STATES,
            "matrix": prob_matrix,
        }

    @staticmethod
    def get_behavior_changes(db: Session, limit: int = 50, severity: str = None) -> List[Dict[str, Any]]:
        query = db.query(BehaviorChange)
        if severity:
            query = query.filter(BehaviorChange.severity == severity.upper())
        changes = query.order_by(desc(BehaviorChange.detected_at)).limit(limit).all()

        return [
            {
                "change_id": c.change_id,
                "customer_id": c.customer_id,
                "metric": c.metric,
                "baseline_value": c.baseline_value,
                "current_value": c.current_value,
                "pct_change": c.pct_change,
                "severity": c.severity,
                "detection_method": c.detection_method,
                "detected_at": c.detected_at,
            }
            for c in changes
        ]
