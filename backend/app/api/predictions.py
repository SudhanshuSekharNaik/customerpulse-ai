"""Predictions and SHAP explainability API endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
from backend.app.services.prediction_service import PredictionService

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])


@router.get("/churn/overview")
def get_churn_overview(db: Session = Depends(get_db)):
    return PredictionService.get_churn_predictions_overview(db)


@router.get("/churn/top-risk")
def get_top_churn_risk(
    limit: int = Query(50, ge=1, le=200),
    search: str = Query(None),
    db: Session = Depends(get_db)
):
    return PredictionService.get_top_churn_customers(db, limit=limit, search=search)
