import json
import urllib.request
import pytest


def test_live_universal_mode():
    try:
        # 1. Health check
        res = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=1).read().decode())
    except Exception:
        pytest.skip("Live uvicorn server not running on port 8000. Skipping live network test.")
        return

    print("Health Check:", res["status"])

    # 2. Upload CSV
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    csv_data = "cust_id,event_time,action_type,revenue\nC1,2026-01-01,view,0\nC1,2026-01-02,buy,499\nC2,2026-01-03,view,0\nC2,2026-01-04,buy,1299\nC3,2026-01-05,view,0\n"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test_customer_events.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
        f"{csv_data}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/universal/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    upload_res = json.loads(urllib.request.urlopen(req).read().decode())
    print("Upload Result:", upload_res["dataset_id"], upload_res["status"], upload_res["capabilities"]["mode_label"])

    # 3. Confirm and Analyze
    ds_id = upload_res["dataset_id"]
    mapping = upload_res["initial_mapping"]
    confirm_req = urllib.request.Request(
        "http://127.0.0.1:8000/api/universal/confirm-and-analyze",
        data=json.dumps({"dataset_id": ds_id, "column_mapping": mapping}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    confirm_res = json.loads(urllib.request.urlopen(confirm_req).read().decode())
    print("Analysis Result:", confirm_res["status"], confirm_res["report"]["mode_label"], f"{len(confirm_res['report']['customer_summaries'])} customers")

    # 4. Query AI Analyst on uploaded dataset
    agent_req = urllib.request.Request(
        "http://127.0.0.1:8000/api/agent/query",
        data=json.dumps({"prompt": "What does this dataset contain and what are the limitations?"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    agent_res = json.loads(urllib.request.urlopen(agent_req).read().decode())
    print("\n--- AI Analyst Query Response ---")
    print("Observed Data:\n", agent_res["observed_data"])
    print("Model Prediction:\n", agent_res["model_prediction"])
    print("Limitations:\n", agent_res["model_estimate"])

    # 5. Switch back to DEMO mode
    switch_req = urllib.request.Request(
        "http://127.0.0.1:8000/api/universal/switch-mode",
        data=json.dumps({"mode": "DEMO"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    switch_res = json.loads(urllib.request.urlopen(switch_req).read().decode())
    print("\nSwitch Mode Result:", switch_res["active_mode"], switch_res["message"])


if __name__ == "__main__":
    test_live_universal_mode()
