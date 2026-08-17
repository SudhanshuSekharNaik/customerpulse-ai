"""CLI script to train Uplift models on Criteo data with --sample support."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from backend.app.database.session import SessionLocal
from backend.app.database.models import CustomerFeature
from ml.data.ingestion import DataIngestion
from ml.uplift.train import UpliftTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Uplift Models (Two-Model vs X-Learner) on Criteo dataset.")
    parser.add_argument("--sample", type=float, default=1.0, help="Sample fraction for development (e.g. 0.1 for 10%%)")
    args = parser.parse_args()

    print(f"=== Training Uplift Models on Criteo Dataset (Sample fraction: {args.sample}) ===")
    ingestion = DataIngestion()
    criteo_df, criteo_hash = ingestion.load_or_generate_criteo_uplift(sample_fraction=args.sample)

    trainer = UpliftTrainer()
    trainer.train(criteo_df, dataset_hash=criteo_hash)

    # Score RetailRocket customer population
    db = SessionLocal()
    try:
        feats = db.query(CustomerFeature).all()
        if feats:
            feat_dicts = [
                {
                    "customer_id": f.customer_id,
                    "cart_to_view_ratio": f.cart_to_view_ratio,
                    "recency_days": f.recency_days,
                    "frequency_30d": f.frequency_30d,
                    "velocity_7d_30d": f.velocity_7d_30d,
                }
                for f in feats
            ]
            feat_df = pd.DataFrame(feat_dicts)
            trainer.score_customer_population_and_save(feat_df)
    finally:
        db.close()

    print("=== Uplift Modeling & Population Scoring Complete ===")


if __name__ == "__main__":
    main()
