"""Customer Segmentation inference module for CustomerPulse AI."""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from ml.segmentation.train import CustomerSegmentationTrainer


class CustomerSegmentPredictor:
    """Assigns segment IDs, labels, and centroid distances for inference."""

    def __init__(self, model_dir: str = "ml/models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "segmentation_kmeans.joblib")
        self.scaler_path = os.path.join(model_dir, "segmentation_scaler.joblib")
        self.labels_path = os.path.join(model_dir, "segmentation_labels.json")
        self.model = None
        self.scaler = None
        self.labels = {}
        self._load()

    def _load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            if os.path.exists(self.labels_path):
                with open(self.labels_path, "r") as f:
                    raw_labels = json.load(f)
                    self.labels = {int(k): v for k, v in raw_labels.items()}

    def predict(self, feature_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict segment assignment and distance to centroid for input feature rows."""
        if self.model is None or self.scaler is None:
            return [{"segment_id": -1, "segment_label": "Unassigned", "distance_to_centroid": 0.0} for _ in range(len(feature_df))]

        X = feature_df[CustomerSegmentationTrainer.FEATURE_COLS].fillna(0).values
        X_scaled = self.scaler.transform(X)
        cluster_preds = self.model.predict(X_scaled)
        centroids = self.model.cluster_centers_

        results = []
        for i, c_id in enumerate(cluster_preds):
            c_vec = X_scaled[i]
            dist = float(np.linalg.norm(c_vec - centroids[c_id]))
            label = self.labels.get(int(c_id), f"Cluster {c_id}")
            results.append({
                "segment_id": int(c_id),
                "segment_label": label,
                "distance_to_centroid": round(dist, 3),
            })
        return results
