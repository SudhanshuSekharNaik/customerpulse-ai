"""Customer 360 API endpoints."""

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.services.customer_service import CustomerService

router = APIRouter(prefix="/api/customers", tags=["Customers"])


@router.get("")
def list_customers(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    state: Optional[str] = None,
    segment_id: Optional[int] = None,
    min_revenue: Optional[float] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return CustomerService.get_customers(
        db,
        limit=limit,
        offset=offset,
        state=state,
        segment_id=segment_id,
        min_revenue=min_revenue,
        search=search,
    )


@router.get("/{customer_id}")
def get_customer_detail(customer_id: str, db: Session = Depends(get_db)):
    detail = CustomerService.get_customer_360_detail(db, customer_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")
    return detail
