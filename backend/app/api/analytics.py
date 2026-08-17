"""Executive Analytics & E-Commerce Traffic Surge Forecasting API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.database.session import get_db
from backend.app.database.models import Customer, CustomerState, CustomerFeature, Prediction, Recommendation, Event, Product
from ml.ecommerce.traffic_forecast import EcommerceTrafficEngine

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/overview")
def get_executive_overview(db: Session = Depends(get_db)):
    total_custs = db.query(Customer).count()
    total_events = db.query(Event).count()
    total_orders = db.query(func.sum(Customer.total_orders)).scalar() or total_events
    total_revenue = db.query(func.sum(Customer.total_revenue)).scalar() or 0.0
    total_products = db.query(Product).count()
    total_recs = db.query(Recommendation).count()

    at_risk_count = db.query(CustomerState).filter(CustomerState.current_state.in_(["AT_RISK", "DORMANT", "DECLINING"])).count()
    at_risk_cids = db.query(CustomerState.customer_id).filter(CustomerState.current_state.in_(["AT_RISK", "DECLINING"]))
    rev_at_risk = db.query(func.sum(Customer.total_revenue)).filter(Customer.customer_id.in_(at_risk_cids)).scalar() or 0.0

    total_uplift_impact = db.query(func.sum(Recommendation.expected_impact)).scalar() or 0.0

    return {
        "total_customers": total_custs,
        "total_events": total_events,
        "total_orders": int(total_orders),
        "total_products": total_products,
        "total_recommendations": total_recs,
        "total_revenue": round(float(total_revenue), 2),
        "total_revenue_inr": round(float(total_revenue), 2),
        "at_risk_customers_count": at_risk_count,
        "at_risk_percentage": round((at_risk_count / max(1, total_custs)) * 100, 1),
        "revenue_at_risk": round(float(rev_at_risk), 2),
        "revenue_at_risk_inr": round(float(rev_at_risk), 2),
        "addressable_portfolio_uplift": round(float(total_uplift_impact), 2),
        "portfolio_actionable_uplift_inr": round(float(total_uplift_impact), 2),
    }


@router.get("/traffic-forecast")
def get_traffic_forecast(db: Session = Depends(get_db)):
    """Hourly traffic curve, next peak surge window, day-of-week demand, and festive sale multipliers."""
    return EcommerceTrafficEngine.analyze_traffic_and_peaks(db=db)


@router.get("/customer-next-action/{customer_id}")
def get_customer_next_action(customer_id: str, db: Session = Depends(get_db)):
    """Predict customer next activity timing, expected action, cart recovery, and voucher sensitivity."""
    return EcommerceTrafficEngine.get_customer_next_action_prediction(customer_id=customer_id, db=db)
