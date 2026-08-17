"""CLI script to generate Next-Best-Actions and run offline policy backtest."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.recommendation.next_best_action import NextBestActionEngine
from backend.app.services.recommendation_service import RecommendationService
from backend.app.database.session import SessionLocal


def main():
    print("=== Generating Next-Best-Action Recommendations ===")
    nba = NextBestActionEngine()
    nba.generate_and_save_all_recommendations()

    # Run Offline Policy Evaluation
    db = SessionLocal()
    try:
        backtest_report = RecommendationService.get_offline_policy_backtest(db)
        print("=" * 60)
        print("       OFFLINE POLICY BACKTEST REPORT       ")
        print("=" * 60)
        print(f"Status:                      {backtest_report.get('status')}")
        print(f"Total Evaluated Customers:   {backtest_report.get('total_evaluated_customers'):,}")
        print(f"Action Diversity Index:      {backtest_report.get('action_diversity_index')} / 1.0")
        print(f"Degenerate Policy Flag:      {backtest_report.get('is_degenerate_policy')}")
        print(f"Top Action Share:            {backtest_report.get('top_action_share'):.1%}")
        print(f"Avg Expected Impact:         INR {backtest_report.get('average_expected_impact_inr'):,.2f}")
        print(f"Total Portfolio Uplift:      INR {backtest_report.get('total_portfolio_uplift_inr'):,.2f}")
        print(f"Conclusion:                  {backtest_report.get('offline_backtest_conclusion')}")
        print("=" * 60)
    finally:
        db.close()


if __name__ == "__main__":
    main()
