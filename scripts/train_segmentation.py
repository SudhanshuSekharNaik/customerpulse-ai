"""CLI script to train customer segmentation."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from backend.app.database.session import SessionLocal
from backend.app.database.models import CustomerFeature
from ml.segmentation.train import CustomerSegmentationTrainer
from ml.data.ingestion import compute_dataframe_hash


def main():
    print("=== Training Customer Segmentation Model ===")
    db = SessionLocal()
    try:
        feats = db.query(CustomerFeature).all()
        if not feats:
            print("No features found. Run `python scripts/build_features.py` first.")
            return

        feat_dicts = [
            {col: getattr(f, col) for col in CustomerSegmentationTrainer.FEATURE_COLS + ["customer_id", "is_cold_start"]}
            for f in feats
        ]
        feat_df = pd.DataFrame(feat_dicts)
        dataset_hash = compute_dataframe_hash(feat_df)

        trainer = CustomerSegmentationTrainer()
        results = trainer.train(feat_df, dataset_hash=dataset_hash)
        print("Segmentation Results Summary:")
        print(f"  - Optimal K: {results['selected_k']}")
        print(f"  - Silhouette Score: {results['silhouette_score']:.4f}")
        print(f"  - Davies-Bouldin: {results['davies_bouldin_score']:.4f}")
        print("  - Auto-generated Segment Labels:")
        for k, v in results["segment_labels"].items():
            print(f"     [{k}] {v}")
        print("=== Segmentation Training Complete ===")
    finally:
        db.close()


if __name__ == "__main__":
    main()
