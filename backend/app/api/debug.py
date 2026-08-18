"""Debug and ML Lineage Consistency Verification Endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
from backend.app.database.models import Customer, Prediction, CustomerSegment, CustomerState, CustomerFeature, ModelRun
from backend.app.services.customer_service import CustomerService

router = APIRouter(prefix="/api/debug", tags=["Debug"])


@router.get("/customer/{customer_id}/consistency")
def get_customer_prediction_consistency(customer_id: str, db: Session = Depends(get_db)):
    cust = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not cust:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    pred = db.query(Prediction).filter(Prediction.customer_id == customer_id, Prediction.model_type == "churn").first()
    c360 = CustomerService.get_customer_360_detail(db, customer_id)
    seg = db.query(CustomerSegment).filter(CustomerSegment.customer_id == customer_id).first()
    st = db.query(CustomerState).filter(CustomerState.customer_id == customer_id).first()
    feat = db.query(CustomerFeature).filter(CustomerFeature.customer_id == customer_id).first()
    churn_run = db.query(ModelRun).filter(ModelRun.model_type == "churn").first()

    c360_churn = c360["churn_prediction"]["predicted_probability"] if (c360 and c360.get("churn_prediction")) else None
    pred_churn = pred.predicted_probability if pred else None

    if c360_churn is None and pred_churn is None:
        diff = 0.0
        status = "PASS"
    elif c360_churn is not None and pred_churn is not None:
        diff = abs(c360_churn - pred_churn)
        status = "PASS" if diff <= 1e-9 else "FAIL"
    else:
        diff = 1.0
        status = "FAIL"

    return {
        "customer_id": customer_id,
        "dataset_hash": churn_run.dataset_hash if churn_run else "v1-canonical-clean",
        "model_version": pred.model_version if pred else "v2.0-universal",
        "customer_360_churn": c360_churn,
        "prediction_churn": pred_churn,
        "difference": diff,
        "segment_customer360": c360.get("segment_label") if c360 else None,
        "segment_canonical": seg.segment_label if seg else None,
        "lifecycle_customer360": c360.get("current_state") if c360 else None,
        "lifecycle_canonical": st.current_state if st else None,
        "total_spend_customer360": c360.get("total_revenue") if c360 else None,
        "total_spend_canonical": feat.monetary_total if feat else None,
        "status": status,
    }
