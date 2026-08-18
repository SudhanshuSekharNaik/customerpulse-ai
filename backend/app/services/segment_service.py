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

        results = []

        for seg_id, raw_label, sil, count in segments:
            # Query average metrics and top category for this segment
            feats = db.query(
                func.avg(CustomerFeature.monetary_total).label("avg_monetary"),
                func.avg(CustomerFeature.recency_days).label("avg_recency"),
                func.avg(CustomerFeature.frequency_30d).label("avg_freq"),
                func.avg(CustomerFeature.cart_to_view_ratio).label("avg_cart"),
            ).join(CustomerSegment, CustomerSegment.customer_id == CustomerFeature.customer_id)\
             .filter(CustomerSegment.segment_id == seg_id).first()

            # Find top category for this segment
            top_cat_row = db.query(
                CustomerFeature.top_category,
                func.count(CustomerFeature.customer_id).label("cat_count")
            ).join(CustomerSegment, CustomerSegment.customer_id == CustomerFeature.customer_id)\
             .filter(CustomerSegment.segment_id == seg_id)\
             .filter(CustomerFeature.top_category != None)\
             .group_by(CustomerFeature.top_category)\
             .order_by(func.count(CustomerFeature.customer_id).desc())\
             .first()

            s_mon = float(feats.avg_monetary or 0.0) if feats else 0.0
            s_rec = float(feats.avg_recency or 0.0) if feats else 0.0
            s_freq = float(feats.avg_freq or 0.0) if feats else 0.0
            s_cart = float(feats.avg_cart or 0.0) if feats else 0.0
            top_cat = str(top_cat_row.top_category) if top_cat_row and top_cat_row.top_category else "General"

            # Clean raw label to human-friendly name
            clean_label = raw_label
            if ":" in clean_label:
                clean_label = clean_label.split(":", 1)[1].strip()
            elif clean_label.startswith("Cluster "):
                clean_label = clean_label.replace("Cluster ", "Segment ")

            # Calculate metric deviations against cohort averages
            mon_diff_pct = ((s_mon - cohort_mon) / max(1.0, cohort_mon)) * 100.0
            rec_diff_pct = ((s_rec - cohort_rec) / max(1.0, cohort_rec)) * 100.0
            freq_diff_pct = ((s_freq - cohort_freq) / max(1.0, cohort_freq)) * 100.0

            mon_diff_str = f"+{mon_diff_pct:.0f}%" if mon_diff_pct >= 0 else f"{mon_diff_pct:.0f}%"
            rec_diff_str = f"+{rec_diff_pct:.0f}%" if rec_diff_pct >= 0 else f"{rec_diff_pct:.0f}%"
            freq_diff_str = f"+{freq_diff_pct:.0f}%" if freq_diff_pct >= 0 else f"{freq_diff_pct:.0f}%"

            # Derive clustering pattern and basis explanation
            if s_mon >= cohort_mon * 1.35 and s_rec <= cohort_rec:
                clean_label = "VIP Champions"
                basis = f"Clustered by High Lifetime Spend (₹{s_mon:,.2f}, {mon_diff_str} vs avg) & Frequent Recent Visits ({s_rec:.1f}d recency). Top revenue power buyers."
                strategy = "Enroll in VIP Prime Loyalty Tier, grant early festival sale access, and offer dedicated support."
                drivers = [f"Spend: ₹{s_mon:,.2f} ({mon_diff_str} vs avg)", f"Recency: {s_rec:.1f} days (Highly Active)", f"Category Affinity: {top_cat}"]
            elif s_mon >= cohort_mon * 1.15 and s_rec > cohort_rec:
                clean_label = "At-Risk High Spenders"
                basis = f"Clustered by High Past Spend (₹{s_mon:,.2f}) combined with Long Inactivity ({s_rec:.1f}d, {rec_diff_str} dormant). Valuable accounts sliding toward churn."
                strategy = "Deploy urgent personalized 15% win-back discount vouchers before the 45-day churn cliff."
                drivers = [f"Inactivity: {s_rec:.1f} days ({rec_diff_str} vs avg)", f"Past Spend: ₹{s_mon:,.2f}", f"Urgency: Churn Prevention"]
            elif s_freq >= cohort_freq * 1.25:
                clean_label = "Frequent Repeat Shoppers"
                basis = f"Clustered by High Repeat Purchase Velocity ({s_freq:.1f} transactions, {freq_diff_str} vs avg) with steady engagement in {top_cat}."
                strategy = "Cross-sell trending accessories and offer subscription multi-pack bundles."
                drivers = [f"Order Velocity: {s_freq:.1f} events ({freq_diff_str} vs avg)", f"Category: {top_cat}", f"Engagement: High Repeat"]
            elif s_rec > cohort_rec * 1.25:
                clean_label = "Dormant / Lapsed Shoppers"
                basis = f"Clustered by Extended Inactivity ({s_rec:.1f} days since last visit) and low engagement frequency."
                strategy = "Re-engage via seasonal flash-sale notifications and price-drop alerts on viewed items."
                drivers = [f"Recency: {s_rec:.1f} days without purchases", f"Historical Spend: ₹{s_mon:,.2f}", f"Strategy: Reactivation"]
            elif s_cart >= cohort_cart * 1.25 and s_cart > 0:
                clean_label = "High-Intent Window Browsers"
                basis = f"Clustered by High Browse & Add-to-Cart Activity with moderate checkout conversion. Strong interest in {top_cat}."
                strategy = "Trigger limited-time cart checkout countdown discounts and free shipping perks."
                drivers = [f"Cart Ratio: {s_cart*100:.1f}%", f"Browsing: High Intent", f"Category: {top_cat}"]
            else:
                clean_label = "Core Steady Customers"
                basis = f"Clustered by Balanced Spend (₹{s_mon:,.2f}) and Standard Inactivity ({s_rec:.1f}d). Forms the steady operational foundation."
                strategy = "Nurture with standard rewards points, product discovery recommendations, and review incentives."
                drivers = [f"Spend: ₹{s_mon:,.2f} (Cohort Average)", f"Recency: {s_rec:.1f} days", f"Category: {top_cat}"]

            results.append({
                "segment_id": seg_id,
                "segment_name": clean_label,
                "segment_label": clean_label,
                "customer_count": count,
                "percentage": round((count / max(1, total_customers)) * 100, 1),
                "avg_revenue": round(s_mon, 2),
                "avg_recency_days": round(s_rec, 1),
                "avg_frequency_30d": round(s_freq, 1),
                "avg_cart_ratio": round(s_cart, 3),
                "top_category": top_cat,
                "silhouette_score": round(float(sil or 0.0), 3),
                "clustering_basis": basis,
                "spend_pattern": f"₹{s_mon:,.2f} ({mon_diff_str} vs avg)",
                "recency_pattern": f"{s_rec:.1f} days ({rec_diff_str} vs avg)",
                "frequency_pattern": f"{s_freq:.1f} events ({freq_diff_str} vs avg)",
                "key_drivers": drivers,
                "recommended_strategy": strategy,
            })

        return sorted(results, key=lambda x: x["segment_id"])
