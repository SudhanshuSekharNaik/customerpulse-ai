"""SHAP explainability utilities for CustomerPulse AI."""

import os
import joblib
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import shap


class ShapExplainer:
    """Computes global feature importance and individual prediction attributions."""

    def __init__(self, model_dir: str = "ml/models"):
        self.explainer_path = os.path.join(model_dir, "churn_shap_explainer.joblib")
        self.explainer = None
        if os.path.exists(self.explainer_path):
            self.explainer = joblib.load(self.explainer_path)

    def explain_instance(self, feature_vector: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Compute SHAP attribution values for a single customer."""
        if self.explainer is None:
            return {}

        shap_vals = self.explainer.shap_values(feature_vector.reshape(1, -1))
        if isinstance(shap_vals, list):
            arr = shap_vals[1][0]
        elif len(shap_vals.shape) == 2:
            arr = shap_vals[0]
        else:
            arr = shap_vals[0, :, 1] if shap_vals.ndim == 3 else shap_vals[0]

        return {name: round(float(val), 4) for name, val in zip(feature_names, arr)}
