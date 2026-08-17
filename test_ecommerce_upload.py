import urllib.request
import json

def test_ecommerce_upload():
    # 1. Upload Amazon/Flipkart/Myntra style CSV
    boundary = "----WebKitBoundary7MA4YW"
    csv_data = """customer_id,event_time,action_type,product_id,product_category,revenue
AMZ_USER_101,2026-08-17 19:15:00,view,PROD_IPHONE,Smartphones,0
AMZ_USER_101,2026-08-17 19:30:00,add_to_cart,PROD_IPHONE,Smartphones,0
AMZ_USER_101,2026-08-17 20:00:00,purchase,PROD_IPHONE,Smartphones,79999
FLIPKART_USER_202,2026-08-17 20:15:00,view,PROD_SNEAKERS,Fashion,0
FLIPKART_USER_202,2026-08-17 20:30:00,add_to_cart,PROD_SNEAKERS,Fashion,0
MYNTRA_USER_303,2026-08-17 21:00:00,view,PROD_KURTA,Ethnic_Wear,0
MYNTRA_USER_303,2026-08-17 21:45:00,purchase,PROD_KURTA,Ethnic_Wear,2499
"""
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="amazon_flipkart_orders.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
        f"{csv_data}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/universal/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    up_res = json.loads(urllib.request.urlopen(req).read().decode())
    ds_id = up_res["dataset_id"]
    mapping = up_res["initial_mapping"]
    print("Uploaded CSV Ingested:", up_res["filename"], "Mode:", up_res["capabilities"]["mode_label"])

    # 2. Confirm and Analyze
    confirm_req = urllib.request.Request(
        "http://127.0.0.1:8000/api/universal/confirm-and-analyze",
        data=json.dumps({"dataset_id": ds_id, "column_mapping": mapping}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    confirm_res = json.loads(urllib.request.urlopen(confirm_req).read().decode())
    print("Live Database Synchronization Status:", confirm_res["status"])

    # 3. Verify /api/customers now returns uploaded e-commerce customers
    cust_res = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/customers?limit=10").read().decode())
    print(f"\nLive Customers in Directory ({cust_res['total']} total):")
    for c in cust_res["items"]:
        print(f"  - Customer ID: {c['customer_id']} | State: {c['current_state']} | Total Revenue: Rs. {c['total_revenue']:,.2f}")

    # 4. Verify Traffic Forecast on uploaded dataset
    tf_res = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/analytics/traffic-forecast").read().decode())
    print(f"\nE-Commerce Traffic Peak Surge Forecast:")
    print(f"  - Next Peak Window: {tf_res['next_peak_window']['window_description']}")
    print(f"  - Cart Abandonment Rate: {tf_res['ecommerce_funnel']['cart_abandonment_rate_pct']}%")
    print(f"  - Top Category Surges: {[c['category_name'] for c in tf_res['category_surges']]}")

    # 5. Verify Customer Next-Action Prediction
    na_res = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/analytics/customer-next-action/FLIPKART_USER_202").read().decode())
    print(f"\nCustomer Next-Action Prediction (FLIPKART_USER_202):")
    print(f"  - Predicted Next Action: {na_res['predicted_next_action']}")
    print(f"  - Expected Visit Timing: {na_res['expected_visit_timing']}")
    print(f"  - Voucher Recommendation: {na_res['discount_voucher_recommendation']}")

if __name__ == "__main__":
    test_ecommerce_upload()
