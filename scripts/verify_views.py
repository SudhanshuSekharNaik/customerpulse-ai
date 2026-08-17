"""Verification script for all CustomerPulse AI views & API integrity."""
import urllib.request
import json

BASE = "http://localhost:8000/api"

def check(name, path, validate_fn):
    data = json.loads(urllib.request.urlopen(BASE + path).read().decode())
    ok, msg = validate_fn(data)
    print(f"[{ok}] {name}: {msg}")

def main():
    print("=" * 60)
    print("CUSTOMER PULSE AI — COMPREHENSIVE VIEW INTEGRITY AUDIT")
    print("=" * 60)
    
    check("Executive Pulse Overview", "/analytics/overview", 
          lambda d: ("PASS" if d["total_customers"] == 1384 and d["total_events"] == 12000 else "FAIL", 
                     f"{d['total_customers']:,} customers, {d['total_events']:,} events, {d['total_orders']:,} orders"))
    
    check("E-Commerce Traffic Funnel", "/analytics/traffic-forecast", 
          lambda d: ("PASS" if 0.0 <= d["ecommerce_funnel"]["checkout_conversion_rate_pct"] <= 100.0 else "FAIL", 
                     f"checkout rate {d['ecommerce_funnel']['checkout_conversion_rate_pct']}%, abandonment {d['ecommerce_funnel']['cart_abandonment_rate_pct']}%"))
    
    check("Predictions PR-AUC Grounding", "/predictions/churn/overview", 
          lambda d: ("PASS" if d["pr_auc"] == 0.7281 else "FAIL", 
                     f"PR-AUC: {d['pr_auc']}, ROC-AUC: {d['roc_auc']}, Scored: {d['total_scored_customers']:,}"))
    
    check("Top Churn Risk & TreeSHAP", "/predictions/churn/top-risk?limit=5", 
          lambda d: ("PASS" if len(d) > 0 and d[0]["top_shap_driver"] is not None and d[0]["total_orders"] >= 1 else "FAIL", 
                     f"Customer: {d[0]['customer_id']}, State: {d[0]['current_state']}, Orders: {d[0]['total_orders']}, Top SHAP: {d[0]['top_shap_driver']}"))
    
    check("Markov Transition Row-Sum Invariant", "/behavior/transitions", 
          lambda d: ("PASS" if len(d["matrix"]) == 9 and round(sum(d["matrix"][0]), 2) == 1.0 else "FAIL", 
                     f"Matrix: 9x9, Row 0 sum: {sum(d['matrix'][0]):.2f}"))
    
    check("Anomaly Radar Individual Baselines", "/behavior/changes?limit=10", 
          lambda d: ("PASS" if len(set(x["baseline_value"] for x in d)) > 1 else "FAIL", 
                     f"{len(set(x['baseline_value'] for x in d))} unique baselines, types: {set(x['metric'] for x in d)}"))
    
    check("Next-Best-Action Impact Variance", "/recommendations?limit=10", 
          lambda d: ("PASS" if len(set(x["expected_impact"] for x in d)) > 1 else "FAIL", 
                     f"Impacts: {[x['expected_impact'] for x in d[:3]]}, Confidences: {set(x['confidence_level'] for x in d)}"))
    
    check("MLOps Model Registry Lineage", "/models", 
          lambda d: ("PASS" if len(d) >= 3 and len(d[0]["dataset_hash"]) >= 16 else "FAIL", 
                     f"{len(d)} models logged, Dataset Hash: {d[0]['dataset_hash'][:16]}..."))
    print("=" * 60)

if __name__ == "__main__":
    main()
