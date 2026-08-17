"""Data and Feature Drift Monitoring for CustomerPulse AI.
Implements Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI)
to label model and feature health as GOOD / WARNING / CRITICAL from real statistical tests.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class DriftMonitor:
    """Calculates KS-statistic, p-values, and PSI to audit distribution drift."""

    FEATURE_COLS = [
        "recency_days",
        "frequency_30d",
        "monetary_total",
        "cart_to_view_ratio",
        "velocity_7d_30d",
    ]

    @staticmethod
    def compute_psi(reference: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
        """Compute Population Stability Index (PSI) between reference and current feature distributions."""
        ref_clean = reference[~np.isnan(reference)]
        curr_clean = current[~np.isnan(current)]

        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return 0.0

        # Create quantile buckets based on reference
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(ref_clean, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        bins = np.unique(bins)

        ref_counts, _ = np.histogram(ref_clean, bins=bins)
        curr_counts, _ = np.histogram(curr_clean, bins=bins)

        ref_pct = ref_counts / max(1, len(ref_clean))
        curr_pct = curr_counts / max(1, len(curr_clean))

        # Handle 0 counts with epsilon smoothing
        eps = 1e-4
        ref_pct = np.where(ref_pct == 0, eps, ref_pct)
        curr_pct = np.where(curr_pct == 0, eps, curr_pct)

        psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(np.clip(psi_val, 0.0, 10.0))

    def evaluate_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute KS test and PSI for each core feature and determine global model health."""
        drift_results = []
        max_psi = 0.0
        drifted_features_count = 0

        for col in self.FEATURE_COLS:
            if col in reference_df.columns and col in current_df.columns:
                ref_vals = reference_df[col].dropna().values
                curr_vals = current_df[col].dropna().values

                # KS test
                ks_stat, p_val = ks_2samp(ref_vals, curr_vals)
                psi_val = self.compute_psi(ref_vals, curr_vals)

                # Drift condition: p_val < 0.05 and PSI > 0.10
                is_drifted = bool(p_val < 0.05 and psi_val > 0.10)
                if is_drifted:
                    drifted_features_count += 1
                max_psi = max(max_psi, psi_val)

                status_label = "CRITICAL" if psi_val > 0.25 else "WARNING" if psi_val > 0.10 else "GOOD"

                drift_results.append({
                    "feature": col,
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "psi": round(float(psi_val), 4),
                    "is_drifted": is_drifted,
                    "status": status_label,
                })

        # Overall model health
        if max_psi > 0.25 or drifted_features_count >= 2:
            model_health = "CRITICAL"
        elif max_psi > 0.10 or drifted_features_count >= 1:
            model_health = "WARNING"
        else:
            model_health = "GOOD"

        return {
            "model_health": model_health,
            "max_psi": round(max_psi, 4),
            "drifted_features_count": drifted_features_count,
            "total_monitored_features": len(drift_results),
            "feature_metrics": drift_results,
        }
