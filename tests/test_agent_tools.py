"""Tests for AI Analyst Agent domain tools and structured response formatting."""

import pytest
from ai.agent import CustomerPulseAnalystAgent
from ai.tools.customer_tools import CustomerTools
from ai.tools.segment_tools import SegmentTools
from ai.tools.analytics_tools import AnalyticsTools


def test_customer_tool_execution():
    res = CustomerTools.search_customers(limit=3)
    assert isinstance(res, list)


def test_segment_tool_execution():
    segs = SegmentTools.get_all_segments()
    assert isinstance(segs, list)


def test_analytics_kpis():
    kpis = AnalyticsTools.get_portfolio_kpis()
    assert "total_customers" in kpis
    assert "revenue_at_risk_inr" in kpis
    assert kpis["total_customers"] >= 0


def test_agent_structured_response():
    agent = CustomerPulseAnalystAgent()
    resp = agent.answer_query("What is the state of customer cust_1?")
    assert "run_id" in resp
    assert "observed_data" in resp
    assert "model_prediction" in resp
    assert "model_estimate" in resp
    assert "recommendation" in resp
    assert len(resp["observed_data"]) > 0
