import io
from fastapi.testclient import TestClient
from backend.app.main import app

def test_ecommerce_upload():
    client = TestClient(app)

    # 1. Upload Amazon/Flipkart/Myntra style CSV
    csv_data = """customer_id,event_time,action_type,product_id,product_category,revenue
AMZ_USER_101,2026-08-17 19:15:00,view,PROD_IPHONE,Smartphones,0
AMZ_USER_101,2026-08-17 19:30:00,add_to_cart,PROD_IPHONE,Smartphones,0
AMZ_USER_101,2026-08-17 20:00:00,purchase,PROD_IPHONE,Smartphones,79999
FLIPKART_USER_202,2026-08-17 20:15:00,view,PROD_SNEAKERS,Fashion,0
FLIPKART_USER_202,2026-08-17 20:30:00,add_to_cart,PROD_SNEAKERS,Fashion,0
MYNTRA_USER_303,2026-08-17 21:00:00,view,PROD_KURTA,Ethnic_Wear,0
MYNTRA_USER_303,2026-08-17 21:45:00,purchase,PROD_KURTA,Ethnic_Wear,2499
"""
    files = {"file": ("amazon_flipkart_orders.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    up_res = client.post("/api/universal/upload", files=files).json()
    ds_id = up_res["dataset_id"]
    mapping = up_res["initial_mapping"]
    assert ds_id is not None
    assert up_res["filename"] == "amazon_flipkart_orders.csv"

    # 2. Confirm and Analyze
    confirm_res = client.post("/api/universal/confirm-and-analyze", json={"dataset_id": ds_id, "column_mapping": mapping}).json()
    assert confirm_res["status"] == "COMPLETED"

    # 3. Verify /api/customers now returns uploaded e-commerce customers
    cust_res = client.get("/api/customers?limit=10").json()
    assert cust_res["total"] >= 3

    # 4. Verify Traffic Forecast on uploaded dataset
    tf_res = client.get("/api/analytics/traffic-forecast").json()
    assert "ecommerce_funnel" in tf_res

    # 5. Verify Customer Next-Action Prediction
    na_res = client.get("/api/analytics/customer-next-action/FLIPKART_USER_202").json()
    assert "predicted_next_action" in na_res

if __name__ == "__main__":
    test_ecommerce_upload()

