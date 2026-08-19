"""Canonical customer segment labeling, statistical pattern derivation,
and multi-cluster disambiguation engine for CustomerPulse AI.
Guarantees distinct, non-colliding human-friendly labels across all views.
"""

from typing import Dict, Any, List


class SegmentLabeler:
    """Generates strictly unique, descriptive segment labels and pattern explanations."""

    @staticmethod
    def generate_segment_profiles(
        clusters_stats: List[Dict[str, Any]],
        cohort_mon: float,
        cohort_rec: float,
        cohort_freq: float,
        cohort_cart: float,
    ) -> List[Dict[str, Any]]:
        """Given cluster statistics and global cohort baselines, assigns non-colliding
        human-friendly business labels, key drivers, clustering basis, and strategies.
        """
        if not clusters_stats:
            return []

        # Step 1: Compute initial behavioral archetypes and deviations
        initial_profiles = []
        for c in clusters_stats:
            cid = c.get("cluster_id", c.get("segment_id", 0))
            s_mon = float(c.get("avg_monetary", c.get("avg_revenue", 0.0)))
            s_rec = float(c.get("avg_recency", c.get("avg_recency_days", 0.0)))
            s_freq = float(c.get("avg_freq", c.get("avg_frequency_30d", 0.0)))
            s_cart = float(c.get("avg_cart", c.get("avg_cart_ratio", 0.0)))
            count = int(c.get("count", c.get("customer_count", 0)))
            top_cat = str(c.get("top_category", "General"))
            sil = float(c.get("silhouette_score", 0.50))

            mon_diff_pct = ((s_mon - cohort_mon) / max(1.0, cohort_mon)) * 100.0
            rec_diff_pct = ((s_rec - cohort_rec) / max(1.0, cohort_rec)) * 100.0
            freq_diff_pct = ((s_freq - cohort_freq) / max(1.0, cohort_freq)) * 100.0

            mon_diff_str = f"+{mon_diff_pct:.0f}%" if mon_diff_pct >= 0 else f"{mon_diff_pct:.0f}%"
            rec_diff_str = f"+{rec_diff_pct:.0f}%" if rec_diff_pct >= 0 else f"{rec_diff_pct:.0f}%"
            freq_diff_str = f"+{freq_diff_pct:.0f}%" if freq_diff_pct >= 0 else f"{freq_diff_pct:.0f}%"

            # Baseline archetype assignment
            if s_mon >= cohort_mon * 2.8 or count <= 10:
                base_label = "VIP Elite — Strategic Outlier"
            elif s_mon >= cohort_mon * 1.35 and s_rec <= cohort_rec * 1.05:
                base_label = "VIP Champions"
            elif s_mon >= cohort_mon * 1.15 and s_rec > cohort_rec:
                base_label = "At-Risk High Spenders"
            elif s_freq >= cohort_freq * 1.25:
                base_label = "Frequent Repeat Shoppers"
            elif s_rec > cohort_rec * 1.25:
                base_label = "Dormant / Lapsed Shoppers"
            elif s_cart >= cohort_cart * 1.25 and s_cart > 0:
                base_label = "High-Intent Window Browsers"
            else:
                base_label = "Core Steady Customers"

            initial_profiles.append({
                "segment_id": cid,
                "base_label": base_label,
                "s_mon": s_mon,
                "s_rec": s_rec,
                "s_freq": s_freq,
                "s_cart": s_cart,
                "count": count,
                "top_cat": top_cat,
                "sil": sil,
                "mon_diff_str": mon_diff_str,
                "rec_diff_str": rec_diff_str,
                "freq_diff_str": freq_diff_str,
            })

        # Step 2: Disambiguate duplicate base labels
        label_counts: Dict[str, List[int]] = {}
        for idx, p in enumerate(initial_profiles):
            label_counts.setdefault(p["base_label"], []).append(idx)

        final_profiles = []
        for base_label, indices in label_counts.items():
            if len(indices) == 1:
                idx = indices[0]
                p = initial_profiles[idx]
                p["final_label"] = p["base_label"]
                final_profiles.append(p)
            else:
                # Multiple clusters shared the same label — disambiguate by spend or distinctive metric
                sorted_by_spend = sorted(indices, key=lambda i: initial_profiles[i]["s_mon"], reverse=True)
                for rank, idx in enumerate(sorted_by_spend):
                    p = initial_profiles[idx]
                    if base_label == "VIP Champions":
                        if rank == 0:
                            p["final_label"] = "VIP Champions (Elite Tier)"
                        else:
                            p["final_label"] = "VIP Champions (Core Spenders)"
                    elif base_label == "Ultra-High-Value Outliers":
                        if rank == 0:
                            p["final_label"] = "VIP Elite"
                        elif rank == 1:
                            p["final_label"] = "High-Value Loyalists"
                        else:
                            p["final_label"] = f"High-Value Tier {rank+1}"
                    elif base_label == "Core Steady Customers":
                        if p["s_freq"] > cohort_freq:
                            p["final_label"] = "Steady Active Buyers"
                        elif p["s_rec"] < cohort_rec:
                            p["final_label"] = "Steady Recent Shoppers"
                        else:
                            p["final_label"] = f"Core Steady Customers (Group {rank+1})"
                    elif base_label == "Frequent Repeat Shoppers":
                        p["final_label"] = f"Frequent Repeat Shoppers ({p['top_cat']})"
                    elif base_label == "Dormant / Lapsed Shoppers":
                        if rank == 0:
                            p["final_label"] = "High-Value Dormant Accounts"
                        else:
                            p["final_label"] = "Lapsed Casual Browsers"
                    else:
                        p["final_label"] = f"{base_label} ({p['top_cat']})"
                    final_profiles.append(p)

        # Step 3: Ensure absolute uniqueness fallback
        seen_labels = set()
        for p in final_profiles:
            lbl = p["final_label"]
            if lbl in seen_labels:
                p["final_label"] = f"{lbl} (Cluster {p['segment_id']})"
            seen_labels.add(p["final_label"])

        # Step 4: Build rich descriptive narratives
        results = []
        for p in sorted(final_profiles, key=lambda x: x["segment_id"]):
            s_mon = p["s_mon"]
            s_rec = p["s_rec"]
            s_freq = p["s_freq"]
            s_cart = p["s_cart"]
            top_cat = p["top_cat"]
            final_lbl = p["final_label"]

            if "VIP" in final_lbl or "Ultra-High" in final_lbl:
                basis = f"Clustered by High Lifetime Spend (₹{s_mon:,.2f}, {p['mon_diff_str']} vs avg) & Frequent Recent Visits ({s_rec:.1f}d recency). Top revenue power buyers."
                strategy = "Enroll in VIP Prime Loyalty Tier, grant early festival sale access, and offer dedicated VIP support."
                drivers = [f"Spend: ₹{s_mon:,.2f} ({p['mon_diff_str']} vs avg)", f"Recency: {s_rec:.1f} days (Highly Active)", f"Category Affinity: {top_cat}"]
            elif "At-Risk" in final_lbl:
                basis = f"Clustered by High Past Spend (₹{s_mon:,.2f}) combined with Long Inactivity ({s_rec:.1f}d, {p['rec_diff_str']} dormant). Valuable accounts sliding toward churn."
                strategy = "Deploy urgent personalized 15% win-back discount vouchers before the 45-day churn cliff."
                drivers = [f"Inactivity: {s_rec:.1f} days ({p['rec_diff_str']} vs avg)", f"Past Spend: ₹{s_mon:,.2f}", f"Urgency: Churn Prevention"]
            elif "Frequent" in final_lbl or "Active" in final_lbl:
                basis = f"Clustered by High Repeat Purchase Velocity ({s_freq:.1f} transactions, {p['freq_diff_str']} vs avg) with steady engagement in {top_cat}."
                strategy = "Cross-sell trending accessories and offer subscription multi-pack bundles."
                drivers = [f"Order Velocity: {s_freq:.1f} events ({p['freq_diff_str']} vs avg)", f"Category: {top_cat}", f"Engagement: High Repeat"]
            elif "Dormant" in final_lbl or "Lapsed" in final_lbl:
                basis = f"Clustered by Extended Inactivity ({s_rec:.1f} days since last visit) and low engagement frequency."
                strategy = "Re-engage via seasonal flash-sale notifications and price-drop alerts on viewed items."
                drivers = [f"Recency: {s_rec:.1f} days without purchases", f"Historical Spend: ₹{s_mon:,.2f}", f"Strategy: Reactivation"]
            elif "Window Browsers" in final_lbl or "Intent" in final_lbl:
                basis = f"Clustered by High Browse & Add-to-Cart Activity with moderate checkout conversion. Strong interest in {top_cat}."
                strategy = "Trigger limited-time cart checkout countdown discounts and free shipping perks."
                drivers = [f"Cart Ratio: {s_cart*100:.1f}%", f"Browsing: High Intent", f"Category: {top_cat}"]
            else:
                basis = f"Clustered by Balanced Spend (₹{s_mon:,.2f}) and Standard Inactivity ({s_rec:.1f}d). Forms the steady operational foundation."
                strategy = "Nurture with standard rewards points, product discovery recommendations, and review incentives."
                drivers = [f"Spend: ₹{s_mon:,.2f} (Cohort Average)", f"Recency: {s_rec:.1f} days", f"Category: {top_cat}"]

            results.append({
                "segment_id": p["segment_id"],
                "segment_name": final_lbl,
                "segment_label": final_lbl,
                "customer_count": p["count"],
                "avg_revenue": round(s_mon, 2),
                "avg_recency_days": round(s_rec, 1),
                "avg_frequency_30d": round(s_freq, 1),
                "avg_cart_ratio": round(s_cart, 3),
                "top_category": top_cat,
                "silhouette_score": round(p["sil"], 3),
                "clustering_basis": basis,
                "spend_pattern": f"₹{s_mon:,.2f} ({p['mon_diff_str']} vs avg)",
                "recency_pattern": f"{s_rec:.1f} days ({p['rec_diff_str']} vs avg)",
                "frequency_pattern": f"{s_freq:.1f} events ({p['freq_diff_str']} vs avg)",
                "key_drivers": drivers,
                "recommended_strategy": strategy,
            })

        return results
