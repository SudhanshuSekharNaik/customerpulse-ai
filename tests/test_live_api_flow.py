"""Integration Test for Universal CSV API endpoints using FastAPI TestClient."""

import json
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_api_universal_mode_flow():
    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"

    # 2. Upload CSV
    csv_content = (
        "cust_id,event_time,action_type,revenue\n"
        "C1,2026-01-01,view,0\n"
        "C1,2026-01-02,buy,499\n"
        "C2,2026-01-03,view,0\n"
        "C2,2026-01-04,buy,1299\n"
        "C3,2026-01-05,view,0\n"
    )
    files = {"file": ("test_customer_events.csv", csv_content.encode("utf-8"), "text/csv")}
    upload_res = client.post("/api/universal/upload", files=files)
    assert upload_res.status_code == 200
    data = upload_res.json()
    assert "dataset_id" in data
    assert data["status"] == "READY_FOR_CONFIRMATION"
    assert data["capabilities"]["is_customer_intelligence_supported"] is True

    ds_id = data["dataset_id"]
    mapping = data["initial_mapping"]

    # 3. Confirm and Analyze
    confirm_res = client.post(
        "/api/universal/confirm-and-analyze",
        json={"dataset_id": ds_id, "column_mapping": mapping},
    )
    assert confirm_res.status_code == 200
    confirm_data = confirm_res.json()
    assert confirm_data["status"] == "COMPLETED"
    assert "report" in confirm_data
    assert len(confirm_data["report"]["customer_summaries"]) > 0

    # 4. Check active context
    ctx_res = client.get("/api/universal/active-context")
    assert ctx_res.status_code == 200
    assert ctx_res.json()["active_mode"] == "UPLOAD"

    # 5. Query AI Analyst on uploaded dataset
    agent_res = client.post(
        "/api/agent/query",
        json={"prompt": "What does this dataset contain and what are the limitations?"},
    )
    assert agent_res.status_code == 200
    agent_data = agent_res.json()
    assert "Observed Data" in agent_data["observed_data"] or "Dataset" in agent_data["observed_data"]
    assert "Limitations" in agent_data["model_estimate"] or "limitations" in agent_data["model_estimate"].lower()

    # 6. Switch back to DEMO mode
    switch_res = client.post("/api/universal/switch-mode", json={"mode": "DEMO"})
    assert switch_res.status_code == 200
    assert switch_res.json()["active_mode"] == "DEMO"
