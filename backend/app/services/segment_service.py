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
        if total_customers == 0:
            return []

        # Calculate cohort global averages
        cohort_avg = db.query(
            func.avg(CustomerFeature.monetary_total).label("avg_monetary"),
            func.avg(CustomerFeature.recency_days).label("avg_recency"),
            func.avg(CustomerFeature.frequency_30d).label("avg_freq"),
            func.avg(CustomerFeature.cart_to_view_ratio).label("avg_cart"),
        ).first()

        cohort_mon = float(cohort_avg.avg_monetary or 1.0) if cohort_avg else 1.0
        cohort_rec = float(cohort_avg.avg_recency or 1.0) if cohort_avg else 1.0
        cohort_freq = float(cohort_avg.avg_freq or 1.0) if cohort_avg else 1.0
        cohort_cart = float(cohort_avg.avg_cart or 0.0) if cohort_avg else 0.0

        clusters_raw = []
        for seg_id, raw_label, sil, count in segments:
            feats = db.query(
                func.avg(CustomerFeature.monetary_total).label("avg_monetary"),
                func.avg(CustomerFeature.recency_days).label("avg_recency"),
                func.avg(CustomerFeature.frequency_30d).label("avg_freq"),
                func.avg(CustomerFeature.cart_to_view_ratio).label("avg_cart"),
            ).join(CustomerSegment, CustomerSegment.customer_id == CustomerFeature.customer_id)\
             .filter(CustomerSegment.segment_id == seg_id).first()

            top_cat_row = db.query(
                CustomerFeature.top_category,
                func.count(CustomerFeature.customer_id).label("cat_count")
            ).join(CustomerSegment, CustomerSegment.customer_id == CustomerFeature.customer_id)\
             .filter(CustomerSegment.segment_id == seg_id)\
             .filter(CustomerFeature.top_category != None)\
             .group_by(CustomerFeature.top_category)\
             .order_by(func.count(CustomerFeature.customer_id).desc())\
             .first()

            top_cat = str(top_cat_row.top_category) if top_cat_row and top_cat_row.top_category else "General"

            clusters_raw.append({
                "segment_id": seg_id,
                "avg_monetary": float(feats.avg_monetary or 0.0) if feats else 0.0,
                "avg_recency": float(feats.avg_recency or 0.0) if feats else 0.0,
                "avg_freq": float(feats.avg_freq or 0.0) if feats else 0.0,
                "avg_cart": float(feats.avg_cart or 0.0) if feats else 0.0,
                "count": count,
                "top_category": top_cat,
                "silhouette_score": float(sil or 0.50),
            })

        from ml.universal.segment_labeler import SegmentLabeler
        results = SegmentLabeler.generate_segment_profiles(
            clusters_raw,
            cohort_mon=cohort_mon,
            cohort_rec=cohort_rec,
            cohort_freq=cohort_freq,
            cohort_cart=cohort_cart,
        )

        for r in results:
            r["percentage"] = round((r["customer_count"] / max(1, total_customers)) * 100.0, 1)

        return sorted(results, key=lambda x: x["segment_id"])
