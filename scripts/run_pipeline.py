"""CLI entrypoint to run end-to-end pipeline."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pipelines.daily_pipeline import run_daily_pipeline


def main():
    parser = argparse.ArgumentParser(description="CustomerPulse AI End-to-End Pipeline CLI Runner.")
    parser.add_argument("--sample", type=float, default=0.2, help="Sample fraction for development (default: 0.2)")
    args = parser.parse_args()
    run_daily_pipeline(sample_fraction=args.sample)


if __name__ == "__main__":
    main()
