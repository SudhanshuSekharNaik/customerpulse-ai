"""Uplift modeling API endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
from backend.app.services.uplift_service import UpliftService

router = APIRouter(prefix="/api/uplift", tags=["Uplift"])


@router.get("/overview")
def get_uplift_overview(db: Session = Depends(get_db)):
    return UpliftService.get_uplift_overview(db)


@router.get("/top-persuadables")
def get_top_persuadables(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return UpliftService.get_top_persuadables(db, limit=limit)
