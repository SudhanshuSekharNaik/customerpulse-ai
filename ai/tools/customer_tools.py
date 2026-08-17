"""Customer domain tools for AI Analyst Agent."""

from typing import Dict, Any, List, Optional
from backend.app.database.session import SessionLocal
from backend.app.services.customer_service import CustomerService


class CustomerTools:
    @staticmethod
    def get_customer_360(customer_id: str) -> Dict[str, Any]:
        """Fetch 360-degree customer profile, feature metrics, state, segment, and recent events."""
        db = SessionLocal()
        try:
            detail = CustomerService.get_customer_360_detail(db, customer_id)
            if not detail:
                return {"error": f"Customer '{customer_id}' not found in database."}
            return detail
        finally:
            db.close()

    @staticmethod
    def search_customers(
        state: Optional[str] = None,
        min_revenue: Optional[float] = None,
        min_recency_days: Optional[float] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search and filter customer population by state, revenue, and inactivity recency."""
        db = SessionLocal()
        try:
            res = CustomerService.get_customers(
                db,
                limit=limit,
                state=state,
                min_revenue=min_revenue,
            )
            return res.get("items", [])
        finally:
            db.close()
