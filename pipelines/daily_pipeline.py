"""End-to-end Daily Batch Pipeline Orchestrator for CustomerPulse AI.
Orchestrates: Ingest -> Validate -> Features -> Segments -> States -> Predictions -> Uplift -> NBA -> Summary.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import argparse
from ml.data.ingestion import DataIngestion
from ml.data.features import FeatureEngineer
from ml.segmentation.train import CustomerSegmentationTrainer
from ml.behavior.change_detection import StateEngine, BehaviorChangeDetector
from ml.prediction.churn import ChurnPredictorTrainer
from ml.prediction.next_event import NextEventPredictorTrainer
from ml.uplift.train import UpliftTrainer
from ml.recommendation.next_best_action import NextBestActionEngine
from backend.app.database.session import SessionLocal
from backend.app.database.models import CustomerFeature, Event
import pandas as pd


def run_daily_pipeline(sample_fraction: float = 1.0):
    start_time = time.time()
    print("=" * 70)
    print(f"   STARTING CUSTOMERPULSE AI DAILY PIPELINE (Sample: {sample_fraction})   ")
    print("=" * 70)

    # 1. Ingest Data & Validate
    print("\n[Stage 1/7] Ingesting & Validating Datasets...")
    ingestion = DataIngestion()
    events_df, rr_hash = ingestion.load_or_generate_retailrocket(sample_fraction=sample_fraction)
    ingestion.ingest_to_database(events_df, sample_fraction=sample_fraction)

    criteo_df, criteo_hash = ingestion.load_or_generate_criteo_uplift(sample_fraction=sample_fraction)

    # 2. Customer 360 Feature Store
    print("\n[Stage 2/7] Computing Point-in-Time Customer 360 Features...")
    fe = FeatureEngineer()
    features_df = fe.compute_customer_features(events_df)
    fe.save_features_to_db(features_df)

    # 3. State Engine & Behavior Monitor
    print("\n[Stage 3/7] Updating Customer Lifecycle States & Detecting Behavior Changes...")
    se = StateEngine()
    states_df, _ = se.compute_states_and_transitions(features_df)
    se.save_states_to_db(states_df)

    bcd = BehaviorChangeDetector()
    changes = bcd.detect_changes(features_df)
    bcd.save_changes_to_db(changes)

    # 4. Customer Segmentation
    print("\n[Stage 4/7] Training Segmentation & Auto-Generating Statistical Labels...")
    seg_trainer = CustomerSegmentationTrainer()
    seg_trainer.train(features_df, dataset_hash=rr_hash)

    # 5. Supervised Predictions (Churn & Next Event)
    print("\n[Stage 5/7] Training Churn (PR-AUC Optimized) and Next-Event Models...")
    churn_trainer = ChurnPredictorTrainer()
    churn_trainer.train(events_df, dataset_hash=rr_hash, n_trials=10)
    churn_trainer.score_all_customers_and_save(features_df)

    nxt_trainer = NextEventPredictorTrainer()
    nxt_trainer.train(events_df, dataset_hash=rr_hash)
    nxt_trainer.score_all_customers_and_save(features_df)

    # 6. Uplift Modeling
    print("\n[Stage 6/7] Training X-Learner vs Two-Model Uplift on Criteo Experimental Data...")
    uplift_trainer = UpliftTrainer()
    uplift_trainer.train(criteo_df, dataset_hash=criteo_hash)
    uplift_trainer.score_customer_population_and_save(features_df)

    # 7. Next-Best-Action Engine
    print("\n[Stage 7/7] Generating Next-Best-Actions & Performing Offline Policy Backtest...")
    nba = NextBestActionEngine()
    nba.generate_and_save_all_recommendations()

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"   CUSTOMERPULSE AI DAILY PIPELINE COMPLETED SUCCESSFULLY IN {elapsed}s   ")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily CustomerPulse batch ML pipeline.")
    parser.add_argument("--sample", type=float, default=1.0, help="Sample fraction for development")
    args = parser.parse_args()
    run_daily_pipeline(sample_fraction=args.sample)
