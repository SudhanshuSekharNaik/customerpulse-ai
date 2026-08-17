#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "=== [1/4] Installing Python dependencies ==="
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "=== [2/4] Building React Vite Frontend ==="
cd frontend
npm install
npm run build
cd ..

echo "=== [3/4] Generating Benchmark Datasets & Seed Data ==="
python scripts/generate_ecommerce_benchmarks.py
python scripts/ingest.py
python scripts/build_features.py

echo "=== [4/4] Training Baseline AI / ML Models ==="
python scripts/train_churn.py
python scripts/train_segmentation.py
python scripts/train_uplift.py
python scripts/train_next_event.py
python scripts/generate_recommendations.py

echo "=== CustomerPulse AI Build Complete & Ready for Render Deployment ==="
