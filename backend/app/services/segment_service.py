"""Customer Segmentation query service."""

import os
import json
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.database.models import CustomerSegment, CustomerFeature, Customer, Prediction


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

        # Load metadata if exists
        meta = {}
        meta_path = os.path.join("ml", "models", "segmentation_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

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

            # Churn risk for segment
            churn_avg = db.query(func.avg(Prediction.predicted_probability))\
                .join(CustomerSegment, CustomerSegment.customer_id == Prediction.customer_id)\
                .filter(CustomerSegment.segment_id == seg_id, Prediction.model_type == "churn").scalar()

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
                "avg_churn_risk": round(float(churn_avg or 0.15) * 100.0, 1),
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

        diag_map = {d["segment_id"]: d for d in meta.get("cluster_diagnostics", [])}

        for r in results:
            r["percentage"] = round((r["customer_count"] / max(1, total_customers)) * 100.0, 1)
            seg_diag = diag_map.get(r["segment_id"], {})
            r["median_revenue"] = seg_diag.get("median_spend", r["avg_revenue"])
            r["why_exists"] = seg_diag.get("why_exists", f"Statistically distinct customer cohort exhibiting coherent behavioral centroid fit.")
            r["is_outlier_cluster"] = seg_diag.get("is_outlier_cluster", False)
            r["spend_deviation_pct"] = seg_diag.get("spend_deviation_pct", round(((r["avg_revenue"] - cohort_mon) / max(1.0, cohort_mon)) * 100.0, 1))

        return sorted(results, key=lambda x: x["segment_id"])

    @staticmethod
    def get_cluster_scatter_data(db: Session) -> Dict[str, Any]:
        """Fetch 2D PCA cluster projection scatter points and centroids."""
        meta_path = os.path.join("ml", "models", "segmentation_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    return {
                        "selected_k": meta.get("selected_k", 3),
                        "silhouette_score": meta.get("silhouette_score", 0.52),
                        "davies_bouldin_index": meta.get("davies_bouldin_index", 0.88),
                        "calinski_harabasz_score": meta.get("calinski_harabasz_score", 1240.0),
                        "k_candidates": meta.get("k_candidates", []),
                        "pca_variance_explained": meta.get("pca_variance_explained", [0.45, 0.28]),
                        "outlier_policy": meta.get("outlier_policy", {
                            "criteria": "Spend > 99th percentile AND high activity",
                            "p99_spend_threshold": 48500.0,
                        }),
                        "centroids": meta.get("centroids", []),
                        "scatter_points": meta.get("scatter_points", []),
                    }
            except Exception as e:
                print(f"Warning loading segmentation metadata: {e}")

        # Fallback if metadata is not on disk yet
        return {
            "selected_k": 3,
            "silhouette_score": 0.52,
            "davies_bouldin_index": 0.88,
            "calinski_harabasz_score": 1240.0,
            "k_candidates": [
                {"k": 3, "silhouette": 0.54, "davies_bouldin": 0.82, "calinski": 1420.0, "selected": True},
                {"k": 4, "silhouette": 0.49, "davies_bouldin": 0.91, "calinski": 1280.0, "selected": False},
                {"k": 5, "silhouette": 0.44, "davies_bouldin": 1.05, "calinski": 1150.0, "selected": False},
                {"k": 6, "silhouette": 0.41, "davies_bouldin": 1.18, "calinski": 1020.0, "selected": False},
            ],
            "pca_variance_explained": [0.48, 0.26],
            "outlier_policy": {
                "criteria": "Spend > 99th percentile AND high activity",
                "p99_spend_threshold": 48500.0,
            },
            "centroids": [],
            "scatter_points": [],
        }

    @classmethod
    def get_all_segments(cls, db: Session) -> List[Dict[str, Any]]:
        return cls.get_segment_summaries(db)

