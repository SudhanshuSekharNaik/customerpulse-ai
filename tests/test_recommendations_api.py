"""Tests for recommendations API endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_recommendations_list_api():
    response = client.get("/api/recommendations?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_offline_backtest_api():
    response = client.get("/api/recommendations/offline-backtest")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "action_diversity_index" in data
