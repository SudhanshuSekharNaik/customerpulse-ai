"""CLI script for dataset ingestion with --sample support."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from ml.data.ingestion import DataIngestion


def main():
    parser = argparse.ArgumentParser(description="Ingest RetailRocket behavioral events and Criteo Uplift datasets.")
    parser.add_argument("--sample", type=float, default=1.0, help="Sample fraction for development (e.g. 0.1 for 10%%)")
    args = parser.parse_args()

    print(f"=== Starting CustomerPulse AI Data Ingestion (Sample fraction: {args.sample}) ===")
    ingestion = DataIngestion()
    
    # 1. RetailRocket behavioral dataset
    events_df, rr_hash = ingestion.load_or_generate_retailrocket(sample_fraction=args.sample)
    print(f"RetailRocket loaded: {len(events_df):,} events (SHA-256: {rr_hash})")
    
    # Ingest into database
    res = ingestion.ingest_to_database(events_df, sample_fraction=args.sample)
    print(f"Database Ingest Result: {res['customers_count']:,} customers, {res['events_count']:,} events.")

    # 2. Criteo Uplift benchmark dataset
    criteo_df, criteo_hash = ingestion.load_or_generate_criteo_uplift(sample_fraction=args.sample)
    print(f"Criteo Uplift loaded: {len(criteo_df):,} rows (SHA-256: {criteo_hash})")
    print("=== Ingestion Finished Successfully ===")


if __name__ == "__main__":
    main()
