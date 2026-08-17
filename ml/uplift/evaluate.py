"""Evaluation metrics and curves for Uplift Modeling: Qini Score, AUUC, Deciles."""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def compute_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute Qini curve, Qini score, AUUC, and uplift by decile."""
    df = pd.DataFrame({
        "y": y_true,
        "t": treatment,
        "score": uplift_scores,
    }).sort_values(by="score", ascending=False).reset_index(drop=True)

    n_total = len(df)
    n_treated_total = (df["t"] == 1).sum()
    n_control_total = (df["t"] == 0).sum()
    y_treated_total = df.loc[df["t"] == 1, "y"].sum()
    y_control_total = df.loc[df["t"] == 0, "y"].sum()

    df["decile"] = pd.qcut(df.index, q=n_bins, labels=False) + 1

    decile_stats = []
    cum_treated_conv = 0
    cum_control_conv = 0
    cum_treated_n = 0
    cum_control_n = 0

    qini_points = [(0.0, 0.0)]

    for d in range(1, n_bins + 1):
        d_df = df[df["decile"] == d]
        
        n_t = (d_df["t"] == 1).sum()
        n_c = (d_df["t"] == 0).sum()
        y_t = d_df.loc[d_df["t"] == 1, "y"].sum()
        y_c = d_df.loc[d_df["t"] == 0, "y"].sum()

        r_t = y_t / max(1, n_t)
        r_c = y_c / max(1, n_c)
        decile_uplift = r_t - r_c

        cum_treated_conv += y_t
        cum_control_conv += y_c
        cum_treated_n += n_t
        cum_control_n += n_c

        # Qini formula: n_t_conv - (n_c_conv * (n_t / n_c))
        qini_val = cum_treated_conv - (cum_control_conv * (n_treated_total / max(1, n_control_total)))
        frac_pop = d / float(n_bins)
        qini_points.append((frac_pop, float(qini_val)))

        decile_stats.append({
            "decile": d,
            "population_fraction": round(frac_pop, 2),
            "treated_conversion_rate": round(float(r_t), 4),
            "control_conversion_rate": round(float(r_c), 4),
            "estimated_uplift": round(float(decile_uplift), 4),
            "cumulative_qini": round(float(qini_val), 2),
        })

    # AUUC / Qini area calculation (numerical integration)
    x_vals = [p[0] for p in qini_points]
    y_vals = [p[1] for p in qini_points]
    
    # Compatible with numpy 2.0+ (trapezoid) and numpy 1.x (trapz)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    qini_auc = float(trapz_fn(y_vals, x_vals))
    
    # Baseline random Qini area
    random_y = [x * y_vals[-1] for x in x_vals]
    random_auc = float(trapz_fn(random_y, x_vals))
    normalized_qini_score = float((qini_auc - random_auc) / max(1.0, abs(random_auc)))

    return {
        "qini_score": round(normalized_qini_score, 4),
        "auuc": round(qini_auc, 2),
        "deciles": decile_stats,
        "qini_curve_points": [{"fraction": p[0], "value": p[1]} for p in qini_points],
    }
