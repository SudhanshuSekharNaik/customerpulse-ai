"""CLI script for Customer 360 feature engineering and state classification."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from backend.app.database.session import SessionLocal
from backend.app.database.models import Event
from ml.data.features import FeatureEngineer
from ml.behavior.change_detection import StateEngine, BehaviorChangeDetector
from ml.behavior.anomaly import AnomalyDetector, OpportunityScoreEngine


def main():
    print("=== Building Customer 360 Feature Store & Customer States ===")
    db = SessionLocal()
    try:
        events = db.query(Event).all()
        if not events:
            print("No events found in database. Run `python scripts/ingest.py` first.")
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
        print(f"Loaded {len(events_df):,} events from database.")

        # 1. Feature Engineering
        fe = FeatureEngineer()
        features_df = fe.compute_customer_features(events_df)
        print(f"Computed features for {len(features_df):,} customers.")
        fe.save_features_to_db(features_df)

        # 2. State Engine & Transition Matrix
        se = StateEngine()
        states_df, transition_matrix = se.compute_states_and_transitions(features_df)
        se.save_states_to_db(states_df)
        print("Customer states and Markov transition matrix updated.")

        # 3. Behavior Changes Detection
        bcd = BehaviorChangeDetector()
        changes = bcd.detect_changes(features_df)
        bcd.save_changes_to_db(changes)
        print(f"Detected and logged {len(changes):,} behavior change alerts.")

        print("=== Customer 360 Features & States Complete ===")
    finally:
        db.close()


if __name__ == "__main__":
    main()
