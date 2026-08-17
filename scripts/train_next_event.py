"""CLI script to train multi-class next event prediction model."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from backend.app.database.session import SessionLocal
from backend.app.database.models import Event, CustomerFeature
from ml.prediction.next_event import NextEventPredictorTrainer
from ml.data.ingestion import compute_dataframe_hash


def main():
    print("=== Training Multi-Class Next-Event Prediction Model ===")
    db = SessionLocal()
    try:
        events = db.query(Event).all()
        feats = db.query(CustomerFeature).all()
        if not events or not feats:
            print("Events or features missing. Run `python scripts/ingest.py` first.")
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

        trainer = NextEventPredictorTrainer()
        trainer.train(events_df, dataset_hash=dataset_hash)

        feat_dicts = [
            {col: getattr(f, col) for col in NextEventPredictorTrainer.FEATURE_COLS + ["customer_id", "is_cold_start"]}
            for f in feats
        ]
        feat_df = pd.DataFrame(feat_dicts)
        trainer.score_all_customers_and_save(feat_df)
        print("=== Next-Event Model Training & Scoring Complete ===")
    finally:
        db.close()


if __name__ == "__main__":
    main()
