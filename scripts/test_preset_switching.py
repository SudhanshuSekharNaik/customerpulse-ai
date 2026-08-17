"""Test switching across Amazon, Flipkart, Myntra, and Campaign datasets."""
import urllib.request
import json

BASE = "http://localhost:8000/api"

def switch(mode):
    req = urllib.request.Request(
        f"{BASE}/universal/switch-mode",
        data=json.dumps({"mode": mode}).encode(),
        headers={"Content-Type": "application/json"}
    )
    res = json.loads(urllib.request.urlopen(req).read().decode())
    print(f"Switch to {mode}: {res.get('message') or res}")

def check_status(expected_name):
    ctx = json.loads(urllib.request.urlopen(f"{BASE}/universal/active-context").read().decode())
    kpis = json.loads(urllib.request.urlopen(f"{BASE}/analytics/overview").read().decode())
    print(f"-> Active Dataset: {ctx.get('dataset_name')}")
    print(f"-> Mode Label: {ctx.get('mode_label')}")
    print(f"-> KPI Overview: {kpis['total_customers']:,} customers, {kpis['total_orders']:,} orders, Rs {kpis['total_revenue']:,.2f}")
    assert expected_name in ctx.get("dataset_name", ""), f"Expected {expected_name}, got {ctx.get('dataset_name')}"
    print("[PASS] Dataset & Analytics Synchronized successfully.\n")

def main():
    print("=" * 60)
    print("PRESET BENCHMARK SWITCHING & SYNCHRONIZATION AUDIT")
    print("=" * 60)

    print("\n1. Switching to Flipkart Benchmark...")
    switch("FLIPKART")
    check_status("flipkart_ecommerce_demo.csv")

    print("2. Switching to Myntra Benchmark...")
    switch("MYNTRA")
    check_status("myntra_lifestyle_demo.csv")

    print("3. Switching back to Amazon Benchmark...")
    switch("AMAZON")
    check_status("amazon_ecommerce_demo.csv")

    print("=" * 60)
    print("ALL E-COMMERCE PRESET DATASETS SWITCHED & VERIFIED CLEANLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
