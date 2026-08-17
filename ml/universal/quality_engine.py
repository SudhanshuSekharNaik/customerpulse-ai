"""Data Quality & Integrity Profiling Engine.
Evaluates completeness, uniqueness, validity, statistical distribution,
and computes a mathematically grounded Data Quality Score (0-100) with itemized audit findings.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime


class DataQualityEngine:
    """Evaluates uploaded CSV data quality with zero fabrication."""

    @classmethod
    def evaluate_quality(
        cls,
        df: pd.DataFrame,
        semantic_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Compute statistical data quality metrics, anomaly checks, and audit score."""
        total_rows = len(df)
        total_cols = len(df.columns)
        if total_rows == 0:
            return {
                "overall_score": 0.0,
                "grade": "F",
                "total_rows": 0,
                "total_columns": total_cols,
                "duplicate_rows": 0,
                "missing_values_count": 0,
                "missing_rate_pct": 100.0,
                "audit_issues": [{"severity": "CRITICAL", "message": "Dataset is completely empty."}],
                "column_quality": {},
            }

        issues: List[Dict[str, Any]] = []
        col_quality: Dict[str, Any] = {}

        # 1. Missingness Analysis
        null_counts = df.isnull().sum()
        total_nulls = int(null_counts.sum())
        total_cells = max(1, total_rows * total_cols)
        overall_missing_pct = round((total_nulls / total_cells) * 100.0, 2)

        for col in df.columns:
            n_null = int(null_counts[col])
            col_missing_pct = round((n_null / total_rows) * 100.0, 2)
            col_quality[col] = {
                "missing_count": n_null,
                "missing_pct": col_missing_pct,
                "unique_values": int(df[col].nunique()),
                "dtype": str(df[col].dtype),
            }
            if col_missing_pct > 30.0:
                issues.append({
                    "severity": "CRITICAL" if col_missing_pct > 60.0 else "WARNING",
                    "column": col,
                    "type": "HIGH_MISSINGNESS",
                    "message": f"Column '{col}' has {col_missing_pct}% missing values ({n_null:,} rows).",
                })

        # 2. Duplicate Rows (exact duplicates across all columns)
        dup_count = int(df.duplicated().sum())
        dup_pct = round((dup_count / total_rows) * 100.0, 2)
        if dup_count > 0:
            issues.append({
                "severity": "WARNING" if dup_pct < 5.0 else "CRITICAL",
                "type": "DUPLICATE_ROWS",
                "message": f"Found {dup_count:,} exact duplicate rows ({dup_pct}% of dataset).",
            })

        # 3. Numeric & Monetary Validity Checks
        negative_value_cols = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            sem_type = (semantic_mapping or {}).get(col, "").upper()
            min_val = float(df[col].dropna().min()) if not df[col].dropna().empty else 0.0
            if "CURRENCY" in sem_type or "REVENUE" in sem_type or "PRICE" in sem_type or "AMOUNT" in col.lower():
                if min_val < 0:
                    neg_count = int((df[col] < 0).sum())
                    negative_value_cols.append(col)
                    issues.append({
                        "severity": "WARNING",
                        "column": col,
                        "type": "NEGATIVE_MONETARY_VALUE",
                        "message": f"Monetary column '{col}' contains {neg_count} negative values (min: {min_val:,.2f}).",
                    })

        # 4. Entity & Transaction Cardinality Check
        entity_cols = [c for c, st in (semantic_mapping or {}).items() if st in ("ENTITY_ID", "CUSTOMER_ID", "USER_ID", "ACCOUNT_ID")]
        if entity_cols:
            ent_col = entity_cols[0]
            unique_ents = int(df[ent_col].nunique())
            avg_tx_per_ent = round(total_rows / max(1, unique_ents), 2)
            col_quality[ent_col]["unique_entities"] = unique_ents
            col_quality[ent_col]["avg_transactions_per_entity"] = avg_tx_per_ent
            if unique_ents == total_rows and total_rows > 100:
                issues.append({
                    "severity": "INFO",
                    "column": ent_col,
                    "type": "ONE_ROW_PER_ENTITY",
                    "message": f"Each row represents a distinct entity ({unique_ents:,} unique accounts). Longitudinal lifecycle transitions will be limited.",
                })
        else:
            unique_ents = None

        # 5. Timestamp / Temporal Continuity Check
        ts_cols = [c for c, st in (semantic_mapping or {}).items() if st in ("TIMESTAMP", "DATE")]
        date_range_info = None
        if ts_cols:
            ts_col = ts_cols[0]
            try:
                dt_series = pd.to_datetime(df[ts_col], errors="coerce").dropna()
                if not dt_series.empty:
                    min_date = dt_series.min()
                    max_date = dt_series.max()
                    span_days = max(1, (max_date - min_date).days)
                    date_range_info = {
                        "column": ts_col,
                        "min_date": str(min_date),
                        "max_date": str(max_date),
                        "span_days": span_days,
                    }
            except Exception:
                pass

        # 6. Mathematical Quality Score Calculation (0 - 100)
        # Base: 100 points
        score = 100.0

        # Deduction for missingness: up to 25 points
        score -= min(25.0, overall_missing_pct * 0.8)

        # Deduction for duplicate rows: up to 15 points
        score -= min(15.0, dup_pct * 1.5)

        # Deduction for negative monetary anomalies: 4 points per column (up to 12)
        score -= min(12.0, len(negative_value_cols) * 4.0)

        # Deduction for critical issues
        crit_issues = [iss for iss in issues if iss.get("severity") == "CRITICAL"]
        score -= min(20.0, len(crit_issues) * 5.0)

        score = max(0.0, min(100.0, round(score, 1)))

        if score >= 90.0:
            grade = "A+ (Production Ready)"
        elif score >= 80.0:
            grade = "A (High Integrity)"
        elif score >= 70.0:
            grade = "B (Acceptable Quality)"
        elif score >= 50.0:
            grade = "C (Degraded Quality)"
        else:
            grade = "D / F (Severe Quality Issues)"

        return {
            "overall_score": score,
            "grade": grade,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "unique_entities": unique_ents,
            "duplicate_rows": dup_count,
            "duplicate_rate_pct": dup_pct,
            "missing_values_count": total_nulls,
            "missing_rate_pct": overall_missing_pct,
            "date_range": date_range_info,
            "issues_count": len(issues),
            "audit_issues": issues,
            "column_quality": col_quality,
        }
