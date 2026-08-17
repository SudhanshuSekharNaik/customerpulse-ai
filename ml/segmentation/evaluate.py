"""Customer Segmentation evaluation module for CustomerPulse AI."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_clusters(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive clustering metrics."""
    if len(np.unique(labels)) < 2:
        return {"silhouette_score": 0.0, "davies_bouldin_score": 0.0, "calinski_harabasz_score": 0.0}

    return {
        "silhouette_score": float(silhouette_score(X, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
    }
