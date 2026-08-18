import pandas as pd
from ml.universal.semantic_detector import SemanticColumnDetector
from ml.universal.capability_engine import CapabilityEngine
from ml.universal.pipeline import UniversalPipelineRunner
from ml.ecommerce.traffic_forecast import EcommerceTrafficEngine

def test_datasets():
    datasets = [
        ("Amazon", "data/amazon_ecommerce_demo.csv"),
        ("Flipkart", "data/flipkart_ecommerce_demo.csv"),
        ("Myntra", "data/myntra_lifestyle_demo.csv")
    ]
    
    for name, path in datasets:
        df = pd.read_csv(path)
        print(f"\n==================== {name} ({len(df)} rows) ====================")
        
        detected = SemanticColumnDetector.detect_all_columns(df)
        mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}
        capabilities = CapabilityEngine.evaluate_capabilities(mapping, len(df))
        
        res = UniversalPipelineRunner.run_pipeline(
            raw_df=df,
            column_mapping=mapping,
            dataset_id=f"{name.lower()}_demo",
            dataset_name=f"{name} Benchmark",
        )
        tf = res["traffic_forecast"]



        
        print("1. Next Peak Window:", tf["next_peak_window"]["window_description"], f"(Lift: +{tf['next_peak_window']['projected_traffic_lift_pct']}%)")
        print("2. Funnel Metrics:")
        print(f"   - Cart Abandonment Rate: {tf['ecommerce_funnel']['cart_abandonment_rate_pct']}%")
        print(f"   - Cart to View Ratio: {tf['ecommerce_funnel']['cart_to_view_ratio_pct']}%")
        print(f"   - Checkout Conversion: {tf['ecommerce_funnel']['checkout_conversion_rate_pct']}%")
        print("3. Top 3 Surging Categories:")
        for c in tf["category_surges"][:3]:
            print(f"   - {c['category_name']}: {c['demand_share_pct']}% (Surge: {c['surge_velocity']})")
        
        # Sample hourly points (12 AM, 8 AM, 1 PM, 8 PM, 11 PM)
        curve = {item["hour"]: item["traffic_index"] for item in tf["hourly_traffic_curve"]}
        print("4. Hourly Activity Index Sample:")
        for hr in [0, 8, 13, 20, 23]:
            print(f"   - Hour {hr:02d}:00 -> Index {curve.get(hr, 0)}")

if __name__ == "__main__":
    test_datasets()
