"""Health and readiness check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
from backend.app.database.models import Customer, ModelRun

router = APIRouter(tags=["Health"])


@router.get("/api/health")
def health_check(db: Session = Depends(get_db)):
    cust_count = db.query(Customer).count()
    model_runs_count = db.query(ModelRun).count()
    return {
        "status": "HEALTHY",
        "service": "CustomerPulse AI",
        "version": "1.0.0",
        "database": "CONNECTED",
        "total_customers": cust_count,
        "trained_models": model_runs_count,
    }
