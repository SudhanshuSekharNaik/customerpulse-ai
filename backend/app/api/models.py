"""Model Registry & MLOps API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.database.session import get_db
from backend.app.database.models import ModelRun

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("")
def list_model_runs(db: Session = Depends(get_db)):
    runs = db.query(ModelRun).order_by(desc(ModelRun.train_timestamp)).all()
    return [
        {
            "run_id": r.run_id,
            "model_name": r.model_name,
            "model_version": r.model_version,
            "model_type": r.model_type,
            "dataset_hash": r.dataset_hash,
            "row_count": r.row_count,
            "train_timestamp": r.train_timestamp,
            "status": r.status,
            "pr_auc": r.pr_auc,
            "roc_auc": r.roc_auc,
            "f1_score": r.f1_score,
            "qini_score": r.qini_score,
            "silhouette_score": r.silhouette_score,
        }
        for r in runs
    ]
