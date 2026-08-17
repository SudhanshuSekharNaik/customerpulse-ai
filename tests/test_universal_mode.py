"""Acceptance Tests for Universal CSV Intelligence Mode (Items 86 to 125)."""

import pytest
import io
import pandas as pd
from ml.universal.profiler import DatasetProfiler
from ml.universal.semantic_detector import SemanticColumnDetector
from ml.universal.capability_engine import CapabilityEngine
from ml.universal.adapter import DatasetAdapter
from ml.universal.pipeline import UniversalPipelineRunner


# -------------------------------------------------------------
# TEST A: RetailRocket-style Event CSV
# -------------------------------------------------------------
def test_acceptance_test_a_customer_event_csv():
    csv_data = """customer_id,timestamp,event,product_id
C101,2026-01-01 10:00:00,view,P100
C101,2026-01-01 10:15:00,add_to_cart,P100
C101,2026-01-02 12:00:00,purchase,P100
C102,2026-01-02 14:00:00,view,P200
C103,2026-01-03 16:00:00,view,P300
C103,2026-01-03 16:30:00,purchase,P300
C104,2026-01-04 11:00:00,view,P400
C105,2026-01-05 09:00:00,view,P500
"""
    res = DatasetProfiler.load_and_profile_csv(csv_data.encode("utf-8"), "retail_events.csv")
    df = res["dataframe"]
    
    detected = SemanticColumnDetector.detect_all_columns(df)
    mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}
    
    assert mapping["customer_id"] == "CUSTOMER_ID"
    assert mapping["timestamp"] in ("TIMESTAMP", "DATE")
    assert mapping["event"] == "EVENT_TYPE"
    assert mapping["product_id"] == "PRODUCT_ID"

    caps = CapabilityEngine.evaluate_capabilities(mapping, len(df))
    assert caps["dataset_mode"] == "CUSTOMER_EVENT"
    assert caps["is_customer_intelligence_supported"] is True
    
    # Run pipeline
    report = UniversalPipelineRunner.run_pipeline(df, mapping, "retail_events.csv")
    assert report["dataset_mode"] == "CUSTOMER_EVENT"
    assert len(report["customer_summaries"]) > 0


# -------------------------------------------------------------
# TEST B: Transaction CSV (RFM + Value)
# -------------------------------------------------------------
def test_acceptance_test_b_transaction_csv():
    csv_data = """customer_id,order_date,amount,product_category
C101,2026-01-01,2499.00,Electronics
C101,2026-01-12,1299.00,Accessories
C102,2026-01-05,8999.00,Electronics
C103,2026-01-15,450.00,Apparel
C104,2026-01-20,1200.00,Books
C105,2026-01-22,3400.00,Electronics
"""
    res = DatasetProfiler.load_and_profile_csv(csv_data.encode("utf-8"), "transactions.csv")
    df = res["dataframe"]
    
    detected = SemanticColumnDetector.detect_all_columns(df)
    mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}
    
    assert mapping["customer_id"] == "CUSTOMER_ID"
    assert mapping["amount"] == "REVENUE"
    
    caps = CapabilityEngine.evaluate_capabilities(mapping, len(df))
    assert caps["dataset_mode"] == "CUSTOMER_TRANSACTION"
    assert any("RFM" in c["name"] for c in caps["enabled_capabilities"])


# -------------------------------------------------------------
# TEST C: Campaign Treatment CSV (Uplift Modeling)
# -------------------------------------------------------------
def test_acceptance_test_c_campaign_uplift_csv():
    csv_data = """customer_id,treatment,conversion,revenue
C101,1,1,2499
C102,0,0,0
C103,1,0,0
C104,0,1,1200
C105,1,1,3400
C106,0,0,0
"""
    res = DatasetProfiler.load_and_profile_csv(csv_data.encode("utf-8"), "campaign.csv")
    df = res["dataframe"]
    
    detected = SemanticColumnDetector.detect_all_columns(df)
    mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}
    
    assert mapping["treatment"] == "TREATMENT"
    assert mapping["conversion"] == "CONVERSION"
    
    caps = CapabilityEngine.evaluate_capabilities(mapping, len(df))
    assert caps["dataset_mode"] == "CAMPAIGN_UPLIFT"
    assert any("Uplift" in c["name"] for c in caps["enabled_capabilities"])


# -------------------------------------------------------------
# TEST D: Business Sales CSV
# -------------------------------------------------------------
def test_acceptance_test_d_business_sales_csv():
    csv_data = """date,region,product,sales,profit,quantity
2026-01-01,North,Widget_A,15000,3200,50
2026-01-02,South,Widget_B,22000,5100,75
2026-01-03,East,Widget_A,18000,4000,60
2026-01-04,West,Widget_C,9500,1800,30
2026-01-05,North,Widget_B,28000,6500,90
"""
    res = DatasetProfiler.load_and_profile_csv(csv_data.encode("utf-8"), "sales.csv")
    df = res["dataframe"]
    
    detected = SemanticColumnDetector.detect_all_columns(df)
    mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}
    
    caps = CapabilityEngine.evaluate_capabilities(mapping, len(df))
    assert caps["dataset_mode"] == "BUSINESS_SALES"
    assert caps["is_customer_intelligence_supported"] is False
    assert any("Time-Series Revenue" in c["name"] for c in caps["enabled_capabilities"])


# -------------------------------------------------------------
# TEST E: Generic Employee CSV (Graceful Limitation Enforcement)
# -------------------------------------------------------------
def test_acceptance_test_e_generic_employee_csv():
    csv_data = """employee_id,department,salary,experience_years
E101,Engineering,125000,6
E102,Product,115000,5
E103,Marketing,85000,3
E104,Sales,95000,4
E105,HR,75000,2
E106,Engineering,145000,8
"""
    res = DatasetProfiler.load_and_profile_csv(csv_data.encode("utf-8"), "employees.csv")
    df = res["dataframe"]
    
    detected = SemanticColumnDetector.detect_all_columns(df)
    mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}
    
    caps = CapabilityEngine.evaluate_capabilities(mapping, len(df))
    # Should reject customer churn and uplift legitimately
    assert any("Uplift" in d["name"] for d in caps["disabled_capabilities"])
    assert any("Churn" in d["name"] for d in caps["disabled_capabilities"])
    assert len(caps["limitations"]) > 0

    report = UniversalPipelineRunner.run_pipeline(df, mapping, "employees.csv")
    assert report["dataset_mode"] == "GENERIC_TABULAR"
    assert report["generic_analytics"] is not None


# -------------------------------------------------------------
# TEST F: Malformed / Empty CSV Error Handling
# -------------------------------------------------------------
def test_acceptance_test_f_malformed_csv():
    empty_csv = ""
    with pytest.raises(Exception):
        DatasetProfiler.load_and_profile_csv(empty_csv.encode("utf-8"), "bad.csv")


# -------------------------------------------------------------
# TEST G: Missing Customer ID CSV -> Generic Mode
# -------------------------------------------------------------
def test_acceptance_test_g_missing_customer_id():
    csv_data = """item,category,price,inventory_count
Laptop,Electronics,1200,45
Phone,Electronics,800,120
Chair,Furniture,150,80
Desk,Furniture,300,35
"""
    res = DatasetProfiler.load_and_profile_csv(csv_data.encode("utf-8"), "inventory.csv")
    df = res["dataframe"]
    detected = SemanticColumnDetector.detect_all_columns(df)
    mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}
    caps = CapabilityEngine.evaluate_capabilities(mapping, len(df))
    assert caps["is_customer_intelligence_supported"] is False


# -------------------------------------------------------------
# TEST H: Ambiguous Columns -> Mapping Confirmation with Confidence
# -------------------------------------------------------------
def test_acceptance_test_h_ambiguous_columns():
    csv_data = """val_x,col_time,score_metric
100,2026-01-01 12:00:00,0.85
200,2026-01-02 12:00:00,0.92
300,2026-01-03 12:00:00,0.78
"""
    res = DatasetProfiler.load_and_profile_csv(csv_data.encode("utf-8"), "ambiguous.csv")
    df = res["dataframe"]
    detected = SemanticColumnDetector.detect_all_columns(df)
    
    # Must produce confidence values for each column
    for col in detected:
        assert "confidence" in col
        assert 0.0 <= col["confidence"] <= 1.0
        assert "detected_semantic_type" in col
