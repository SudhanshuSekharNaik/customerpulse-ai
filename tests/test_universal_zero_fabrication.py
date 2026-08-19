"""Zero-Fabrication & Mathematical Integrity Acceptance Test Suite.
Verifies that all metrics, customer counts, revenues, model evaluations,
and capability gating flags are strictly derived from the uploaded dataset.
"""

import os
import io
import json
import pytest
import pandas as pd
import numpy as np

from ml.universal.profiler import DatasetProfiler
from ml.universal.semantic_detector import SemanticColumnDetector
from ml.universal.capability_engine import CapabilityEngine
from ml.universal.quality_engine import DataQualityEngine
from ml.universal.pipeline import UniversalPipelineRunner
from backend.app.services.pdf_report_generator import PDFReportGenerator
from scripts.generate_amazon_demo import generate_amazon_demo_csv


def test_amazon_benchmark_zero_fabrication():
    """Test 1: Amazon e-commerce CSV evaluation."""
    demo_path = "data/amazon_ecommerce_demo.csv"
    if not os.path.exists(demo_path):
        generate_amazon_demo_csv()

    df = pd.read_csv(demo_path)
    assert len(df) == 12000
    expected_unique_customers = int(df["customer_id"].nunique())
    assert expected_unique_customers == 3000
    sales_col = "sales_amount_inr" if "sales_amount_inr" in df.columns else "sales_amount"
    expected_total_revenue = float(df[sales_col].sum())
    expected_aov = round(expected_total_revenue / float(len(df)), 2)

    # 1. Semantic Column Detection
    detected = SemanticColumnDetector.detect_all_columns(df)
    mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}
    assert mapping["customer_id"] == "CUSTOMER_ID"
    assert mapping["order_id"] == "TRANSACTION_ID"
    assert mapping[sales_col] in ("REVENUE", "PRICE", "ORDER_VALUE")

    # 2. Data Quality Profiling
    dq = DataQualityEngine.evaluate_quality(df, semantic_mapping=mapping)
    assert dq["total_rows"] == 15000
    assert dq["unique_entities"] == 3000
    assert dq["duplicate_rows"] == 0
    assert dq["overall_score"] >= 90.0

    # 3. Strict Capability Evaluation
    caps = CapabilityEngine.evaluate_capabilities(mapping, len(df), df=df)
    # Uplift MUST be locked/unavailable since no treatment/control experiment exists
    assert caps["capabilities"]["uplift_modeling"]["available"] is False
    assert "No treatment/control" in caps["capabilities"]["uplift_modeling"]["reason"]

    # Customer 360 and Segmentation MUST be available
    assert caps["capabilities"]["customer_360"]["available"] is True
    assert caps["capabilities"]["segmentation"]["available"] is True

    # 4. Pipeline Execution
    report = UniversalPipelineRunner.run_pipeline(df, mapping, "amazon_ecommerce_demo.csv", dataset_id="ds_test_amz")
    assert report["total_rows"] == len(df)
    assert report["unique_entities"] == expected_unique_customers

    # Check top customer summaries have valid revenues and honest cold start
    for c in report["customer_summaries"]:
        assert c["total_revenue"] >= 0.0
        if c.get("is_cold_start", False) or c.get("churn_probability") is None:
            assert c["churn_probability"] is None or (isinstance(c["churn_probability"], (int, float)) and not np.isnan(c["churn_probability"]))
        else:
            assert 0.0 <= float(c["churn_probability"]) <= 1.0

    # Check segmentation has optimal K with valid silhouette
    seg_models = [m for m in report["model_results"] if m["model_type"] == "SEGMENTATION"]
    assert len(seg_models) >= 1
    assert seg_models[0]["metric_value"] > 0.0


def test_treatment_campaign_uplift_gating():
    """Test 2: Uplift modeling only enabled when treatment + outcome exist."""
    # A. Dataset WITHOUT treatment
    no_treatment_csv = """cust_id,order_date,amount
U1,2026-01-01,100
U2,2026-01-02,200
U3,2026-01-03,150
"""
    df_no = pd.read_csv(io.StringIO(no_treatment_csv))
    det_no = SemanticColumnDetector.detect_all_columns(df_no)
    map_no = {c["column_name"]: c["detected_semantic_type"] for c in det_no}
    caps_no = CapabilityEngine.evaluate_capabilities(map_no, len(df_no), df=df_no)
    assert caps_no["capabilities"]["uplift_modeling"]["available"] is False

    # B. Dataset WITH treatment and conversion
    uplift_csv = """cust_id,treatment,converted,spend
U1,1,1,120
U2,0,0,0
U3,1,0,0
U4,0,1,80
U5,1,1,200
U6,0,0,0
U7,1,1,150
U8,0,0,0
U9,1,0,0
U10,0,0,0
"""
    df_up = pd.read_csv(io.StringIO(uplift_csv))
    det_up = SemanticColumnDetector.detect_all_columns(df_up)
    map_up = {c["column_name"]: c["detected_semantic_type"] for c in det_up}
    caps_up = CapabilityEngine.evaluate_capabilities(map_up, len(df_up), df=df_up)
    assert caps_up["capabilities"]["uplift_modeling"]["available"] is True


def test_pdf_report_generator():
    """Test 3: Executive PDF report generation produces valid PDF bytes."""
    kpis = {
        "total_customers": 1384,
        "total_events": 12000,
        "total_revenue": 326839045.70,
        "revenue_at_risk": 45200000.0,
        "at_risk_percentage": 14.2,
    }
    caps = {
        "capabilities": {
            "customer_360": {"available": True, "reason": "1,384 accounts detected."},
            "uplift_modeling": {"available": False, "reason": "No treatment variable in dataset."},
            "segmentation": {"available": True, "reason": "K-Means clustering enabled."},
        }
    }
    dq = {
        "overall_score": 96.5,
        "grade": "A+ (Production Ready)",
        "missing_rate_pct": 0.0,
        "duplicate_rows": 0,
        "unique_entities": 1384,
    }
    segs = [
        {"segment_label": "Cluster 0: High-Value VIPs", "customer_count": 420, "percentage": 30.3, "avg_revenue": 520000.0, "avg_recency_days": 12.0, "top_category": "Smartphones"},
        {"segment_label": "Cluster 1: Steady Shoppers", "customer_count": 964, "percentage": 69.7, "avg_revenue": 112000.0, "avg_recency_days": 24.0, "top_category": "Fashion"},
    ]
    states = [{"state": "ENGAGED", "customer_count": 850}, {"state": "DECLINING", "customer_count": 200}]
    recs = [{"customer_id": "CUST_0001", "action_type": "CROSS_SELL", "what_text": "Cross-sell accessories", "why_text": "Recent purchase", "expected_impact": 1500.0}]
    anomalies = [{"customer_id": "CUST_0002", "metric": "Monetary Surge", "severity": "HIGH"}]
    insights = ["Total sales volume of INR 326.8M.", "Cluster 0 represents 68% of portfolio revenue."]

    pdf_bytes = PDFReportGenerator.generate_pdf_bytes(
        dataset_name="amazon_ecommerce_demo.csv",
        kpis=kpis,
        capabilities=caps,
        data_quality=dq,
        segment_summaries=segs,
        state_summaries=states,
        recommendations=recs,
        anomalies=anomalies,
        key_insights=insights,
    )

    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
