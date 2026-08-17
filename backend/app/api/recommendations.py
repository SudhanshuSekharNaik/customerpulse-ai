"""Next-Best-Action recommendation API endpoints."""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
from backend.app.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/api/recommendations", tags=["Recommendations"])


@router.get("")
def list_recommendations(
    limit: int = Query(50, ge=1, le=200),
    action_type: Optional[str] = None,
    confidence: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return RecommendationService.get_recommendations(
        db,
        limit=limit,
        action_type=action_type,
        confidence=confidence,
    )


@router.get("/offline-backtest")
def get_offline_backtest(db: Session = Depends(get_db)):
    return RecommendationService.get_offline_policy_backtest(db)
