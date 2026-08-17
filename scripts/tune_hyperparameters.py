"""CLI script to run systematic Optuna hyperparameter searches."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from backend.app.database.session import SessionLocal
from backend.app.database.models import Event
from ml.prediction.churn import ChurnPredictorTrainer
from ml.data.ingestion import compute_dataframe_hash


def main():
    parser = argparse.ArgumentParser(description="Tune LightGBM Churn Predictor Hyperparameters with Optuna.")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna trials")
    args = parser.parse_args()

    print(f"=== Starting Optuna Hyperparameter Optimization ({args.trials} trials) ===")
    db = SessionLocal()
    try:
        events = db.query(Event).all()
        if not events:
            print("No events found in database. Ingest first.")
            return

        events_data = [
            {
                "event_id": e.event_id,
                "customer_id": e.customer_id,
                "visitor_id": e.visitor_id,
                "event_type": e.event_type,
                "item_id": e.item_id,
                "category_id": e.category_id,
                "timestamp": e.timestamp,
                "transaction_id": e.transaction_id,
                "revenue": e.revenue,
            }
            for e in events
        ]
        events_df = pd.DataFrame(events_data)
        dataset_hash = compute_dataframe_hash(events_df)

        trainer = ChurnPredictorTrainer()
        results = trainer.train(events_df, dataset_hash=dataset_hash, n_trials=args.trials)
        print("Optuna Optimization Completed.")
        print(f"Best PR-AUC achieved: {results['pr_auc']:.4f}")
        print("Best Hyperparameters:")
        print(results["hyperparameters"])
    finally:
        db.close()


if __name__ == "__main__":
    main()
