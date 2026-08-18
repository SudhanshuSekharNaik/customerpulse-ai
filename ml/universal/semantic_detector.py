"""Semantic Column Detector using multi-signal heuristics."""

import re
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


class SemanticColumnDetector:
    """Infers semantic meaning of dataset columns using multi-signal heuristics:
    1. Header patterns & synonyms
    2. Data types
    3. Sample values & regex patterns
    4. Cardinality ratios & distributions
    5. Statistical characteristics
    """

    SEMANTIC_TYPES = [
        "CUSTOMER_ID",
        "USER_ID",
        "ACCOUNT_ID",
        "TRANSACTION_ID",
        "TIMESTAMP",
        "DATE",
        "PRODUCT_ID",
        "PRODUCT_CATEGORY",
        "EVENT_TYPE",
        "QUANTITY",
        "PRICE",
        "REVENUE",
        "ORDER_VALUE",
        "SESSION_ID",
        "CAMPAIGN_ID",
        "TREATMENT",
        "CONVERSION",
        "CHURN_LABEL",
        "TARGET",
        "COUNTRY",
        "DEVICE",
        "CHANNEL",
        "RATING",
        "RETURNED",
        "EMAIL",
        "PHONE",
        "NAME",
        "UNKNOWN",
    ]

    # Regex patterns for column names
    NAME_PATTERNS = {
        "CUSTOMER_ID": r"\b(canonical_?customer_?id|canonical_?cust_?id|customer_?id|cust_?id|client_?id|visitor_?id|shopper_?id|member_?id|patient_?id|pat_?id)\b",
        "USER_ID": r"\b(user_?id|uid|employee_?id|emp_?id|staff_?id)\b",
        "ACCOUNT_ID": r"\b(account_?id|acc_?id|acct_?no|org_?id)\b",
        "TRANSACTION_ID": r"\b(transaction_?id|tx_?id|order_?id|invoice_?id|bill_?id|receipt_?id)\b",
        "TIMESTAMP": r"\b(timestamp|event_?time|datetime|created_?at|occurred_?at|time_?stamp|ts)\b",
        "DATE": r"\b(date|order_?date|trans_?date|day|visit_?date|tx_?date)\b",
        "PRODUCT_ID": r"\b(product_?id|item_?id|sku|prod_?id|item_?code|asin|sku_code)\b",
        "PRODUCT_CATEGORY": r"\b(category|product_?category|cat_?id|department|dept|genre|item_?category|vertical)\b",
        "EVENT_TYPE": r"\b(event|event_?type|action|activity|event_?name|interaction_?type|step|action_?type)\b",
        "QUANTITY": r"\b(quantity|qty|units|item_?count|pieces|volume)\b",
        "PRICE": r"\b(price|unit_?price|cost|rate|msrp|unit_?price_?inr)\b",
        "REVENUE": r"\b(revenue|amount|amt|total_?amt|sales|total_?price|spend|gmv|monetary|sales_?amount_?inr|order_?value_?inr|cart_?value_?inr)\b",
        "ORDER_VALUE": r"\b(order_?value|order_?total|subtotal|basket_?size)\b",
        "SESSION_ID": r"\b(session_?id|sess_?id|visit_?id)\b",
        "CAMPAIGN_ID": r"\b(campaign_?id|camp_?id|promo_?id|offer_?id)\b",
        "TREATMENT": r"\b(treatment|is_?treated|treatment_?group|group|exposed|variant|test_?group)\b",
        "CONVERSION": r"\b(conversion|converted|is_?converted|outcome|purchased|bought|revisit)\b",
        "CHURN_LABEL": r"\b(churn|churn_?label|is_?churned|churned|inactivity_?target|cancelled|left)\b",
        "TARGET": r"\b(target|label|y|class|prediction_?target)\b",
        "COUNTRY": r"\b(country|nation|region|state|city|location|geo|customer_?city|user_?city|shipping_?city)\b",
        "DEVICE": r"\b(device|os|browser|platform|device_?type|user_?agent)\b",
        "CHANNEL": r"\b(channel|medium|source|traffic_?source|utm_?source|referral)\b",
        "RATING": r"\b(rating|customer_?rating|customer_?score|review_?score|stars|score)\b",
        "RETURNED": r"\b(returned|is_?returned|return_?flag|refunded|is_?refunded)\b",
        "EMAIL": r"\b(email|e_?mail|email_?address)\b",
        "PHONE": r"\b(phone|mobile|telephone|cell|phone_?num)\b",
        "NAME": r"\b(name|customer_?name|first_?name|last_?name|full_?name|product_?name|title|brand_?title)\b",
    }

    @classmethod
    def infer_column_semantic(
        cls,
        col_name: str,
        series: pd.Series,
        row_count: int,
    ) -> Tuple[str, float]:
        """Determine the most likely semantic type and associated confidence score (0.0 to 1.0)."""
        clean_name = col_name.strip().lower()
        num_unique = int(series.nunique(dropna=True))
        cardinality_ratio = num_unique / max(1, row_count)
        sample_vals = [str(v) for v in series.dropna().head(20).tolist()]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        scores: Dict[str, float] = {}

        # Check column name regex patterns
        for sem_type, pattern in cls.NAME_PATTERNS.items():
            if re.search(pattern, clean_name, re.I):
                scores[sem_type] = scores.get(sem_type, 0.0) + 0.65

        # Signal 2: Value Pattern & Regex Matching
        # Email pattern
        if any(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v) for v in sample_vals):
            scores["EMAIL"] = scores.get("EMAIL", 0.0) + 0.85

        # Datetime pattern in values
        if not is_numeric and len(sample_vals) > 0:
            try:
                parsed_dt = pd.to_datetime(sample_vals, errors="coerce")
                if parsed_dt.notnull().sum() >= len(sample_vals) * 0.7:
                    if "time" in clean_name or any(":" in v for v in sample_vals):
                        scores["TIMESTAMP"] = scores.get("TIMESTAMP", 0.0) + 0.80
                    else:
                        scores["DATE"] = scores.get("DATE", 0.0) + 0.75
            except Exception:
                pass

        # Treatment & Conversion binary values (0/1 or True/False)
        unique_vals_set = set(series.dropna().unique().tolist()[:10])
        if unique_vals_set.issubset({0, 1, 0.0, 1.0, True, False, "0", "1", "true", "false", "control", "treatment"}):
            if "treatment" in clean_name or "group" in clean_name or "variant" in clean_name:
                scores["TREATMENT"] = scores.get("TREATMENT", 0.0) + 0.90
            elif "convert" in clean_name or "outcome" in clean_name:
                scores["CONVERSION"] = scores.get("CONVERSION", 0.0) + 0.90
            elif "churn" in clean_name or "cancel" in clean_name:
                scores["CHURN_LABEL"] = scores.get("CHURN_LABEL", 0.0) + 0.90
            else:
                scores["TARGET"] = scores.get("TARGET", 0.0) + 0.35

        # Event Type values ('view', 'cart', 'buy', 'click', 'pageview', 'login')
        event_keywords = {"view", "click", "add_to_cart", "addtocart", "purchase", "transaction", "checkout", "login", "pageview", "search"}
        if any(any(kw in str(v).lower() for kw in event_keywords) for v in sample_vals):
            scores["EVENT_TYPE"] = scores.get("EVENT_TYPE", 0.0) + 0.85

        # Revenue / Numeric Value distribution
        if is_numeric:
            if series.dropna().min() >= 0:
                if "amt" in clean_name or "amount" in clean_name or "rev" in clean_name or "price" in clean_name or "spend" in clean_name or "sales" in clean_name:
                    scores["REVENUE"] = scores.get("REVENUE", 0.0) + 0.80
                elif "qty" in clean_name or "quantity" in clean_name or "count" in clean_name:
                    scores["QUANTITY"] = scores.get("QUANTITY", 0.0) + 0.80

        # Customer ID cardinality
        if cardinality_ratio > 0.05 and ("cust" in clean_name or "user" in clean_name or "visitor" in clean_name or "client" in clean_name):
            scores["CUSTOMER_ID"] = scores.get("CUSTOMER_ID", 0.0) + 0.85

        if not scores:
            return "UNKNOWN", 0.30

        # Find best scored semantic type
        best_type, best_score = max(scores.items(), key=lambda x: x[1])
        # Cap confidence between 0.30 and 0.99
        confidence = min(0.99, max(0.40, round(best_score, 2)))
        return best_type, confidence

    @classmethod
    def detect_all_columns(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Inspect all columns in dataframe and generate semantic mappings with confidence."""
        results = []
        row_count = len(df)
        for col in df.columns:
            sem_type, confidence = cls.infer_column_semantic(col, df[col], row_count)
            results.append({
                "column_name": col,
                "detected_semantic_type": sem_type,
                "confidence": confidence,
                "options": cls.SEMANTIC_TYPES,
                "raw_dtype": str(df[col].dtype),
                "unique_values": int(df[col].nunique(dropna=True)),
                "sample_values": [str(v) for v in df[col].dropna().head(3).tolist()],
            })
        return results
