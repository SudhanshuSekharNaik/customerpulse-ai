"""Tests for prediction API endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"


def test_predictions_churn_overview():
    response = client.get("/api/predictions/churn/overview")
    assert response.status_code == 200
    data = response.json()
    assert "total_scored_customers" in data
    assert "primary_metric_pr_auc" in data


def test_customers_api():
    response = client.get("/api/customers?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) <= 5
