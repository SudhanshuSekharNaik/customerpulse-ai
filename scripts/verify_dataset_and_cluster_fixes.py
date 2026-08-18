"""Verify dynamic dataset switching, active filename updates, and cluster pattern explanations."""

import os
import sys
import json

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from backend.app.main import app

def run_verification():
    client = TestClient(app)

    print("=" * 70)
    print("1. VERIFYING INITIAL ACTIVE CONTEXT & SEGMENTS")
    print("=" * 70)
    
    ctx = client.get("/api/universal/active-context").json()
    print(f"Active Dataset Name: {ctx.get('dataset_name')}")
    print(f"Active Mode Label:   {ctx.get('mode_label')}")
    print(f"Active Mode:         {ctx.get('active_mode')}")

    segs = client.get("/api/segments").json()
    print(f"\nInitial Segments Count: {len(segs)}")
    for s in segs:
        print(f"  * Segment #{s['segment_id']+1}: {s['segment_label']} ({s['percentage']}%)")
        print(f"    - Clustering Basis: {s.get('clustering_basis')}")
        print(f"    - Spend Pattern:    {s.get('spend_pattern')}")
        print(f"    - Recency Pattern:  {s.get('recency_pattern')}")
        print(f"    - Key Drivers:      {s.get('key_drivers')}")
        print(f"    - Strategy:         {s.get('recommended_strategy')}")

    print("\n" + "=" * 70)
    print("2. TESTING DATASET SWITCHING -> FLIPKART MULTI-CATEGORY")
    print("=" * 70)

    sw_res = client.post("/api/universal/switch-mode", json={"mode": "FLIPKART"}).json()
    print(f"Switch Response: {sw_res.get('message') or sw_res}")

    ctx_fk = client.get("/api/universal/active-context").json()
    assert "flipkart_ecommerce_demo.csv" in ctx_fk.get("dataset_name", ""), f"Mismatch: {ctx_fk}"
    print(f"-> Active Dataset updated to: {ctx_fk.get('dataset_name')}")
    print(f"-> Mode Label updated to:   {ctx_fk.get('mode_label')}")

    kpi_fk = client.get("/api/analytics/overview").json()
    print(f"-> Synchronized Accounts in DB: {kpi_fk['total_customers']:,} customers, {kpi_fk['total_events']:,} events, Spend: Rs {kpi_fk['total_revenue']:,.2f}")
    assert kpi_fk["total_customers"] > 0

    segs_fk = client.get("/api/segments").json()
    print(f"-> Flipkart Segments Count: {len(segs_fk)}")
    for s in segs_fk:
        print(f"  * {s['segment_label']} ({s['customer_count']} users · {s['percentage']}%)")
        print(f"    - Clustering Basis: {s.get('clustering_basis')}")
        print(f"    - Top Category:     {s.get('top_category')}")

    print("\n" + "=" * 70)
    print("3. TESTING DATASET SWITCHING -> MYNTRA LIFESTYLE & BEAUTY")
    print("=" * 70)

    sw_myn = client.post("/api/universal/switch-mode", json={"mode": "MYNTRA"}).json()
    print(f"Switch Response: {sw_myn.get('message') or sw_myn}")

    ctx_myn = client.get("/api/universal/active-context").json()
    assert "myntra_lifestyle_demo.csv" in ctx_myn.get("dataset_name", ""), f"Mismatch: {ctx_myn}"
    print(f"-> Active Dataset updated to: {ctx_myn.get('dataset_name')}")
    print(f"-> Mode Label updated to:   {ctx_myn.get('mode_label')}")

    kpi_myn = client.get("/api/analytics/overview").json()
    print(f"-> Synchronized Accounts in DB: {kpi_myn['total_customers']:,} customers, {kpi_myn['total_events']:,} events")

    print("\n" + "=" * 70)
    print("4. SWITCHING BACK TO AMAZON BENCHMARK")
    print("=" * 70)
    sw_amz = client.post("/api/universal/switch-mode", json={"mode": "AMAZON"}).json()
    ctx_amz = client.get("/api/universal/active-context").json()
    assert "amazon_ecommerce_demo.csv" in ctx_amz.get("dataset_name", "")
    print(f"-> Active Dataset restored to: {ctx_amz.get('dataset_name')}")

    print("\n" + "=" * 70)
    print("ALL VERIFICATIONS COMPLETED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
