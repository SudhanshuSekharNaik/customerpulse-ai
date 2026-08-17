"""Behavior and Customer State API endpoints."""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
from backend.app.services.behavior_service import BehaviorService

router = APIRouter(prefix="/api/behavior", tags=["Behavior"])


@router.get("/states")
def get_states(db: Session = Depends(get_db)):
    return BehaviorService.get_state_distribution(db)


@router.get("/transitions")
def get_transitions(db: Session = Depends(get_db)):
    return BehaviorService.get_transition_matrix(db)


@router.get("/changes")
def get_changes(
    limit: int = Query(50, ge=1, le=200),
    severity: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return BehaviorService.get_behavior_changes(db, limit=limit, severity=severity)
