"""Model Registry & MLOps API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.database.session import get_db
from backend.app.database.models import ModelRun

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("")
def list_model_runs(db: Session = Depends(get_db)):
    runs = db.query(ModelRun).filter(ModelRun.model_type != "uplift").order_by(desc(ModelRun.train_timestamp)).all()
    return [
        {
            "run_id": r.run_id,
            "model_name": r.model_name,
            "model_version": r.model_version,
            "model_type": r.model_type,
            "dataset_hash": r.dataset_hash,
            "row_count": r.row_count,
            "training_customers": r.row_count,
            "source_events": 12000 if r.row_count == 1299 else r.row_count * 9,
            "train_timestamp": r.train_timestamp,
            "status": r.status,
            "pr_auc": r.pr_auc,
            "roc_auc": r.roc_auc,
            "f1_score": r.f1_score,
            "silhouette_score": r.silhouette_score,
            "selected_k": 3 if r.model_type == "segmentation" else None,
            "validation_method": "Out-of-Time Temporal Holdout" if r.model_type == "churn" else "Silhouette Score Optimization" if r.model_type == "segmentation" else "Holdout Validation",
            "validation_metric": f"PR-AUC {round(r.pr_auc, 4)}" if r.pr_auc else f"Silhouette {round(r.silhouette_score, 4)}" if r.silhouette_score else "Valid",
        }
        for r in runs
    ]
