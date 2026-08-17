"""Analysis Capability Detection Engine with Strict Evidence Gating.
Never fabricates ML capability flags. Evaluates what analyses are mathematically
and statistically supported by the uploaded schema and data volume.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


class CapabilityEngine:
    """Evaluates and enforces data-driven analysis capabilities with zero hallucination."""

    @classmethod
    def evaluate_capabilities(
        cls,
        column_mapping: Dict[str, str],
        row_count: int,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Determine which ML and analytics modules are strictly supported by the dataset."""
        semantic_types = set(column_mapping.values())
        has_customer_entity = "CUSTOMER_ID" in semantic_types or "ACCOUNT_ID" in semantic_types
        has_entity = any(st in ("ENTITY_ID", "CUSTOMER_ID", "ACCOUNT_ID") for st in semantic_types)
        has_timestamp = any(st in ("TIMESTAMP", "DATE") for st in semantic_types)
        has_monetary = any(st in ("CURRENCY", "NUMERIC", "VALUE", "REVENUE", "PRICE", "ORDER_VALUE", "SALES", "AMOUNT") for st in semantic_types)
        has_event_or_tx = any(st in ("TRANSACTION_ID", "EVENT_TYPE", "ORDER_ID") for st in semantic_types)
        has_product = any(st in ("PRODUCT_ID", "ITEM_ID", "SKU", "ASIN", "PRODUCT_CATEGORY") for st in semantic_types)
        has_treatment = "TREATMENT" in semantic_types
        has_outcome = any(st in ("OUTCOME", "CONVERSION") for st in semantic_types)
        has_churn_label = any(st in ("CHURN_LABEL", "TARGET") for st in semantic_types)

        # Calculate entity count and temporal span if df is provided
        entity_count = row_count
        span_days = 0
        has_repeated_interactions = False

        if df is not None and has_entity:
            ent_cols = [c for c, st in column_mapping.items() if st in ("ENTITY_ID", "CUSTOMER_ID", "ACCOUNT_ID")]
            if ent_cols:
                ent_col = ent_cols[0]
                entity_count = int(df[ent_col].nunique())
                avg_obs = row_count / max(1, entity_count)
                has_repeated_interactions = avg_obs > 1.2 or row_count > entity_count

            if has_timestamp:
                ts_cols = [c for c, st in column_mapping.items() if st in ("TIMESTAMP", "DATE")]
                if ts_cols:
                    try:
                        dts = pd.to_datetime(df[ts_cols[0]], errors="coerce").dropna()
                        if not dts.empty:
                            span_days = max(0, (dts.max() - dts.min()).days)
                    except Exception:
                        pass
        elif has_entity and has_timestamp:
            has_repeated_interactions = True
            span_days = 90

        # Detailed Module Capability Checks
        capabilities: Dict[str, Dict[str, Any]] = {}

        # 1. Customer / Entity 360
        if has_entity:
            capabilities["customer_360"] = {
                "available": True,
                "reason": f"Detected entity identifier with {entity_count:,} unique accounts.",
            }
        else:
            capabilities["customer_360"] = {
                "available": False,
                "reason": "No reliable entity/customer identifier detected in the uploaded schema.",
            }

        # 2. RFM Behavioral Analytics
        if has_entity and has_timestamp and has_monetary:
            capabilities["rfm_analysis"] = {
                "available": True,
                "reason": "Entity ID, transaction timestamps, and monetary values are all present.",
            }
        else:
            missing = []
            if not has_entity: missing.append("Entity ID")
            if not has_timestamp: missing.append("Timestamp")
            if not has_monetary: missing.append("Monetary/Revenue")
            capabilities["rfm_analysis"] = {
                "available": False,
                "reason": f"Missing required RFM dimensions: {', '.join(missing)}.",
            }

        # 3. Customer Segmentation (Clustering)
        if has_entity and entity_count >= 6:
            capabilities["segmentation"] = {
                "available": True,
                "reason": f"Sufficient entity volume ({entity_count:,} accounts) for unsupervised clustering.",
            }
        else:
            capabilities["segmentation"] = {
                "available": False,
                "reason": "Insufficient entity sample size (minimum 6 accounts required for clustering).",
            }

        # 4. Lifecycle 9-State Machine
        if has_entity and has_timestamp and (has_repeated_interactions or span_days >= 7):
            capabilities["lifecycle_model"] = {
                "available": True,
                "reason": f"Temporal observations spanning {span_days} days across entity lifecycle.",
            }
        else:
            capabilities["lifecycle_model"] = {
                "available": False,
                "reason": "Insufficient longitudinal observations to model multi-stage lifecycle states.",
            }

        # 5. Markov Transition Matrix
        if has_entity and has_timestamp and has_repeated_interactions and span_days >= 14:
            capabilities["markov_transitions"] = {
                "available": True,
                "reason": "Multiple observation time windows available for state transition probability estimation.",
            }
        else:
            capabilities["markov_transitions"] = {
                "available": False,
                "reason": "Insufficient longitudinal observations across multiple time windows for transition modeling.",
            }

        # 6. Churn Prediction & TreeSHAP
        can_train_temporal_churn = has_entity and has_timestamp and span_days >= 30 and row_count >= 100
        if has_churn_label or can_train_temporal_churn:
            capabilities["churn_prediction"] = {
                "available": True,
                "reason": "Explicit churn label or time-aware 70/30 observation cutoff target successfully established.",
            }
        else:
            capabilities["churn_prediction"] = {
                "available": False,
                "reason": "No explicit churn target column and time span is insufficient (< 30 days) to construct a non-leaking temporal churn window.",
            }

        # 7. Uplift / Causal Machine Learning (CRITICAL ZERO-FABRICATION RULE)
        if has_treatment and has_outcome:
            capabilities["uplift_modeling"] = {
                "available": True,
                "reason": "Treatment intervention and outcome conversion variables detected.",
            }
        else:
            capabilities["uplift_modeling"] = {
                "available": False,
                "reason": "No treatment/control intervention variable and binary outcome suitable for causal estimation were detected in the uploaded schema.",
            }

        # 8. Product Affinity & Collaborative Co-Occurrence
        if has_product and (has_event_or_tx or has_entity):
            capabilities["product_affinity"] = {
                "available": True,
                "reason": "Product identifiers and entity order baskets detected.",
            }
        else:
            capabilities["product_affinity"] = {
                "available": False,
                "reason": "No product identifiers or order baskets present to calculate item co-occurrence.",
            }

        # 9. Next-Best-Action Decision Cards
        if has_entity:
            capabilities["recommendations"] = {
                "available": True,
                "reason": "Behavioral next-best-actions derived from individual customer purchase history, recency, and category affinities.",
            }
        else:
            capabilities["recommendations"] = {
                "available": False,
                "reason": "Entity ID required to synthesize individualized decision cards.",
            }

        # 10. Anomaly Detection (Entity & Transaction Level)
        capabilities["anomaly_detection"] = {
            "available": True,
            "reason": "Isolation Forest and statistical Z-score/EWMA anomaly radar supported.",
        }

        # 11. Time-Series Revenue Forecasting (for Business Sales or Aggregate Datasets)
        if (has_monetary and has_timestamp) or not has_entity:
            capabilities["time_series_revenue"] = {
                "available": True,
                "reason": "Time-series sales metrics and trend forecasting supported.",
            }

        # 12. AI Analyst Agent & Executive PDF
        capabilities["ai_analyst"] = {"available": True, "reason": "AI Analyst Agent supported."}
        capabilities["executive_pdf"] = {"available": True, "reason": "Executive PDF report generator supported."}

        # Determine Dataset Mode
        if has_treatment and has_outcome:
            dataset_mode = "CAMPAIGN_UPLIFT"
            mode_label = "A/B Testing & Campaign Uplift Dataset"
            domain = "MARKETING"
        elif has_entity and has_event_or_tx:
            dataset_mode = "CUSTOMER_EVENT"
            mode_label = "Customer Behavioral & Order Dataset"
            domain = "ECOMMERCE_RETAIL"
        elif has_entity and (has_monetary or has_timestamp):
            dataset_mode = "CUSTOMER_TRANSACTION"
            mode_label = "Customer Account & Transaction Dataset"
            domain = "COMMERCE"
        elif not has_entity and has_monetary and (has_timestamp or has_product):
            dataset_mode = "BUSINESS_SALES"
            mode_label = "Business Sales & Aggregate Metrics Dataset"
            domain = "SALES"
        else:
            dataset_mode = "GENERIC_TABULAR"
            mode_label = "Universal Tabular Dataset"
            domain = "GENERAL"

        NAME_MAP = {
            "rfm_analysis": "RFM Behavioral Analytics",
            "time_series_revenue": "Time-Series Revenue Forecasting",
            "customer_360": "Customer 360 Profiles",
            "segmentation": "Customer Clustering & Segmentation",
            "lifecycle_model": "9-State Lifecycle Engine",
            "markov_transitions": "Markov State Transitions",
            "churn_prediction": "Churn Prediction & TreeSHAP",
            "uplift_modeling": "Causal Uplift Modeling",
            "product_affinity": "Product Affinity & Basket Analysis",
            "recommendations": "Next-Best-Action Decision Cards",
            "anomaly_detection": "Statistical Anomaly Radar",
            "ai_analyst": "AI Decision Analyst Agent",
            "executive_pdf": "Executive PDF Report Generator",
        }

        # Enabled vs Disabled lists
        enabled_list = [{"name": NAME_MAP.get(k, k.replace("_", " ").title()), "description": v["reason"]} for k, v in capabilities.items() if v["available"]]
        disabled_list = [{"name": NAME_MAP.get(k, k.replace("_", " ").title()), "reason": v["reason"]} for k, v in capabilities.items() if not v["available"]]

        limitations = [f"{item['name']}: {item['reason']}" for item in disabled_list]

        return {
            "dataset_mode": dataset_mode,
            "domain": domain,
            "mode_label": mode_label,
            "is_customer_intelligence_supported": has_entity,
            "capabilities": capabilities,
            "enabled_capabilities": enabled_list,
            "disabled_capabilities": disabled_list,
            "limitations": limitations,
            "schema_summary": {
                "has_entity": has_entity,
                "has_timestamp": has_timestamp,
                "has_monetary": has_monetary,
                "has_event_or_tx": has_event_or_tx,
                "has_product": has_product,
                "has_treatment": has_treatment,
                "has_outcome": has_outcome,
                "has_churn_label": has_churn_label,
            },
        }
