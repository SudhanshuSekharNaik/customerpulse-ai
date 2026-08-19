"""Segmentation API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
from backend.app.services.segment_service import SegmentService

router = APIRouter(prefix="/api/segments", tags=["Segments"])


@router.get("")
def list_segments(db: Session = Depends(get_db)):
    return SegmentService.get_segment_summaries(db)


@router.get("/scatter")
def get_cluster_scatter(db: Session = Depends(get_db)):
    return SegmentService.get_cluster_scatter_data(db)

