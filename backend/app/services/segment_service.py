"""Customer Segmentation query service."""

import os
import json
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.database.models import CustomerSegment, CustomerFeature, Customer


class SegmentService:
    @staticmethod
    def get_segment_summaries(db: Session) -> List[Dict[str, Any]]:
        segments = db.query(
            CustomerSegment.segment_id,
            CustomerSegment.segment_label,
            CustomerSegment.silhouette_score,
            func.count(CustomerSegment.customer_id).label("count")
        ).group_by(CustomerSegment.segment_id, CustomerSegment.segment_label, CustomerSegment.silhouette_score).all()

        total_customers = db.query(CustomerSegment).count()
        results = []

        for seg_id, label, sil, count in segments:
            # Query average metrics for this segment
            feats = db.query(
                func.avg(CustomerFeature.monetary_total).label("avg_monetary"),
                func.avg(CustomerFeature.recency_days).label("avg_recency"),
                func.avg(CustomerFeature.frequency_30d).label("avg_freq"),
                func.avg(CustomerFeature.cart_to_view_ratio).label("avg_cart"),
            ).join(CustomerSegment, CustomerSegment.customer_id == CustomerFeature.customer_id)\
             .filter(CustomerSegment.segment_id == seg_id).first()

            results.append({
                "segment_id": seg_id,
                "segment_label": label,
                "customer_count": count,
                "percentage": round((count / max(1, total_customers)) * 100, 1),
                "avg_revenue": round(float(feats.avg_monetary or 0.0), 2),
                "avg_recency_days": round(float(feats.avg_recency or 0.0), 1),
                "avg_frequency_30d": round(float(feats.avg_freq or 0.0), 1),
                "avg_cart_ratio": round(float(feats.avg_cart or 0.0), 3),
                "top_category": "Electronics & Apparel",
                "silhouette_score": round(float(sil or 0.0), 3),
            })

        return sorted(results, key=lambda x: x["segment_id"])
