"""FastAPI router for Universal CSV Intelligence Mode."""

import os
import json
import uuid
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from backend.app.database.session import get_db, SessionLocal
from backend.app.database.models import UploadedDataset
from ml.universal.profiler import DatasetProfiler
from ml.universal.semantic_detector import SemanticColumnDetector
from ml.universal.capability_engine import CapabilityEngine
from ml.universal.pipeline import UniversalPipelineRunner

router = APIRouter(prefix="/api/universal", tags=["Universal Data Mode"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ConfirmAnalysisRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset_id: str
    column_mapping: Dict[str, str]  # column_name -> semantic_type


class SwitchModeRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: str  # "DEMO" or "UPLOAD"
    dataset_id: Optional[str] = None


@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload raw CSV file, sanitize, profile, infer semantic columns, and return preliminary capabilities."""
    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid CSV file.")

    contents = await file.read()
    if len(contents) > DatasetProfiler.MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds maximum allowed size (100MB).")

    try:
        profile_res = DatasetProfiler.load_and_profile_csv(contents, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    profile = profile_res["profile"]
    df = profile_res["dataframe"]

    # Infer semantic columns
    detected_columns = SemanticColumnDetector.detect_all_columns(df)
    initial_mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected_columns}

    # Evaluate capabilities
    capabilities = CapabilityEngine.evaluate_capabilities(initial_mapping, len(df))

    dataset_id = f"ds_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    raw_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_{file.filename}")
    with open(raw_path, "wb") as f:
        f.write(contents)

    # Save initial record to DB
    ds_record = UploadedDataset(
        dataset_id=dataset_id,
        filename=file.filename,
        dataset_mode=capabilities["dataset_mode"],
        domain=capabilities["domain"],
        mode_label=capabilities["mode_label"],
        row_count=profile["total_rows"],
        col_count=profile["total_columns"],
        profile_json=json.dumps(profile),
        column_mapping_json=json.dumps(initial_mapping),
        capabilities_json=json.dumps(capabilities),
        raw_csv_path=raw_path,
        status="PROFILED",
        is_active=False,
    )
    db.add(ds_record)
    db.commit()
    db.refresh(ds_record)

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "profile": profile,
        "detected_columns": detected_columns,
        "initial_mapping": initial_mapping,
        "capabilities": capabilities,
        "status": "READY_FOR_CONFIRMATION",
    }


@router.post("/confirm-and-analyze")
def confirm_and_analyze(
    req: ConfirmAnalysisRequest,
    db: Session = Depends(get_db),
):
    """Confirm column mapping, run adaptive pipeline, train models, and generate full report."""
    ds = db.query(UploadedDataset).filter(UploadedDataset.dataset_id == req.dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset record not found.")

    if not ds.raw_csv_path or not os.path.exists(ds.raw_csv_path):
        raise HTTPException(status_code=400, detail="Underlying raw CSV file is unavailable.")

    # Load dataframe
    try:
        profile_res = DatasetProfiler.load_and_profile_csv(ds.raw_csv_path, filename=ds.filename)
        df = profile_res["dataframe"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to reload dataset: {str(e)}")

    # Run complete adaptive pipeline
    report = UniversalPipelineRunner.run_pipeline(
        raw_df=df,
        column_mapping=req.column_mapping,
        dataset_name=ds.filename,
        dataset_id=ds.dataset_id,
    )

    # Deactivate other datasets
    db.query(UploadedDataset).update({UploadedDataset.is_active: False})

    # Update dataset record
    ds.dataset_mode = report["dataset_mode"]
    ds.domain = report["domain"]
    ds.mode_label = report["mode_label"]
    ds.column_mapping_json = json.dumps(req.column_mapping)
    ds.capabilities_json = json.dumps(report["capabilities"])
    ds.report_json = json.dumps(report)
    ds.status = "ANALYZED"
    ds.is_active = True
    ds.analyzed_at = datetime.utcnow()

    db.commit()
    db.refresh(ds)

    return {
        "dataset_id": ds.dataset_id,
        "filename": ds.filename,
        "report": report,
        "status": "COMPLETED",
    }


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    """List all previously uploaded and analyzed datasets."""
    datasets = db.query(UploadedDataset).order_by(UploadedDataset.created_at.desc()).all()
    results = []
    for d in datasets:
        results.append({
            "dataset_id": d.dataset_id,
            "filename": d.filename,
            "dataset_mode": d.dataset_mode,
            "domain": d.domain,
            "mode_label": d.mode_label,
            "row_count": d.row_count,
            "col_count": d.col_count,
            "status": d.status,
            "is_active": d.is_active,
            "created_at": d.created_at,
            "analyzed_at": d.analyzed_at,
        })
    return results


@router.get("/dataset/{dataset_id}")
def get_dataset_detail(dataset_id: str, db: Session = Depends(get_db)):
    """Retrieve full analysis report and profile for a specific dataset."""
    ds = db.query(UploadedDataset).filter(UploadedDataset.dataset_id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    report = json.loads(ds.report_json) if ds.report_json else None
    profile = json.loads(ds.profile_json) if ds.profile_json else None
    capabilities = json.loads(ds.capabilities_json) if ds.capabilities_json else None
    column_mapping = json.loads(ds.column_mapping_json) if ds.column_mapping_json else None

    return {
        "dataset_id": ds.dataset_id,
        "filename": ds.filename,
        "dataset_mode": ds.dataset_mode,
        "domain": ds.domain,
        "mode_label": ds.mode_label,
        "row_count": ds.row_count,
        "col_count": ds.col_count,
        "status": ds.status,
        "is_active": ds.is_active,
        "profile": profile,
        "column_mapping": column_mapping,
        "capabilities": capabilities,
        "report": report,
        "created_at": ds.created_at,
        "analyzed_at": ds.analyzed_at,
    }


from fastapi.responses import FileResponse

PRESETS = [
    {
        "preset_id": "AMAZON",
        "name": "Amazon E-Commerce Benchmark",
        "filename": "amazon_ecommerce_demo.csv",
        "row_count": 12000,
        "customer_count": 3000,
        "domain": "E-Commerce & Retail",
        "features": ["order_id", "customer_id", "order_date", "product_name", "category", "sales_amount_inr", "city", "rating"],
        "description": "12,000 order transactions across 3,000 unique customers (cust_1 to cust_3000) with RFM, TreeSHAP, and Causal Uplift.",
    },
    {
        "preset_id": "FLIPKART",
        "name": "Flipkart Multi-Category Benchmark",
        "filename": "flipkart_ecommerce_demo.csv",
        "row_count": 10000,
        "customer_count": 1109,
        "domain": "Electronics & SuperMart",
        "features": ["order_id", "user_id", "order_timestamp", "title", "vertical", "order_value_inr", "payment_type", "user_city"],
        "description": "10,000 orders covering Mobiles, Laptops, Gadgets, and Apparel with SuperCoins & Pay Later payments.",
    },
    {
        "preset_id": "MYNTRA",
        "name": "Myntra Lifestyle & Beauty Benchmark",
        "filename": "myntra_lifestyle_demo.csv",
        "row_count": 8000,
        "customer_count": 928,
        "domain": "Fashion & Cosmetics",
        "features": ["transaction_id", "account_id", "purchase_date", "brand_title", "product_category", "cart_value_inr", "coupon_discount"],
        "description": "8,000 orders across Western & Ethnic Wear, Sneakers, and Luxury Beauty with coupon discount dynamics.",
    },
]


@router.get("/presets")
def get_presets():
    """List available pre-configured e-commerce and marketing benchmark datasets."""
    return PRESETS


@router.get("/presets/{preset_id}/download")
def download_preset_csv(preset_id: str):
    """Download a standardized benchmark CSV file."""
    preset = next((p for p in PRESETS if p["preset_id"] == preset_id.upper()), None)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset dataset not found.")
    file_path = os.path.join("data", preset["filename"])
    if not os.path.exists(file_path):
        from scripts.generate_ecommerce_benchmarks import generate_all_ecommerce_benchmarks
        generate_all_ecommerce_benchmarks()
    return FileResponse(file_path, filename=preset["filename"], media_type="text/csv")


@router.post("/switch-mode")
def switch_analysis_mode(
    req: SwitchModeRequest,
    db: Session = Depends(get_db),
):
    """Switch active platform mode between Demo Benchmark presets and Uploaded Datasets."""
    preset_match = next((p for p in PRESETS if p["preset_id"] == req.mode.upper() or p["filename"] == req.dataset_id), None)
    
    if req.mode == "DEMO" or preset_match is not None:
        db.query(UploadedDataset).update({UploadedDataset.is_active: False})
        db.commit()

        target_preset = preset_match or PRESETS[0]
        csv_path = os.path.join("data", target_preset["filename"])
        if not os.path.exists(csv_path):
            from scripts.generate_ecommerce_benchmarks import generate_all_ecommerce_benchmarks
            generate_all_ecommerce_benchmarks()

        try:
            df_preset = pd.read_csv(csv_path)
            detected = SemanticColumnDetector.detect_all_columns(df_preset)
            mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}
            dataset_id = f"ds_preset_{target_preset['preset_id'].lower()}"
            report = UniversalPipelineRunner.run_pipeline(df_preset, mapping, target_preset["filename"], dataset_id=dataset_id)
            
            # Upsert this preset dataset as active in UploadedDataset table
            db.query(UploadedDataset).update({UploadedDataset.is_active: False})
            ds_rec = db.query(UploadedDataset).filter(UploadedDataset.dataset_id == dataset_id).first()
            if not ds_rec:
                ds_rec = UploadedDataset(
                    dataset_id=dataset_id,
                    filename=target_preset["filename"],
                    dataset_mode=report.get("dataset_mode", "CUSTOMER_TRANSACTION"),
                    domain=target_preset.get("domain", "E-Commerce & Retail"),
                    mode_label=target_preset["name"],
                    row_count=len(df_preset),
                    col_count=len(df_preset.columns),
                    column_mapping_json=json.dumps(mapping),
                    capabilities_json=json.dumps(report.get("capabilities", {})),
                    report_json=json.dumps(report),
                    raw_csv_path=csv_path,
                    status="ANALYZED",
                    is_active=True,
                    analyzed_at=datetime.utcnow(),
                )
                db.add(ds_rec)
            else:
                ds_rec.is_active = True
                ds_rec.mode_label = target_preset["name"]
                ds_rec.filename = target_preset["filename"]
                ds_rec.row_count = len(df_preset)
                ds_rec.report_json = json.dumps(report)
                ds_rec.capabilities_json = json.dumps(report.get("capabilities", {}))
                ds_rec.status = "ANALYZED"
                ds_rec.analyzed_at = datetime.utcnow()
            db.commit()
        except Exception as e:
            print(f"Warning running preset pipeline for {target_preset['name']}: {e}")

        return {
            "active_mode": "DEMO",
            "preset_id": target_preset["preset_id"],
            "dataset_name": target_preset["filename"],
            "mode_label": target_preset["name"],
            "message": f"Switched to {target_preset['name']} ({target_preset['row_count']:,} records).",
        }
    else:
        if not req.dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id is required when mode is UPLOAD.")
        ds = db.query(UploadedDataset).filter(UploadedDataset.dataset_id == req.dataset_id).first()
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        db.query(UploadedDataset).update({UploadedDataset.is_active: False})
        ds.is_active = True
        db.commit()

        # Re-sync this dataset's database tables
        if ds.raw_csv_path and os.path.exists(ds.raw_csv_path) and ds.column_mapping_json:
            try:
                mapping = json.loads(ds.column_mapping_json)
                profile_res = DatasetProfiler.load_and_profile_csv(ds.raw_csv_path, filename=ds.filename)
                report = UniversalPipelineRunner.run_pipeline(profile_res["dataframe"], mapping, ds.filename, dataset_id=ds.dataset_id)
                ds.report_json = json.dumps(report)
                ds.analyzed_at = datetime.utcnow()
                db.commit()
            except Exception as e:
                print(f"Warning re-syncing uploaded dataset: {e}")

        return {
            "active_mode": "UPLOAD",
            "dataset_id": ds.dataset_id,
            "filename": ds.filename,
            "dataset_mode": ds.dataset_mode,
            "mode_label": ds.mode_label,
        }


@router.get("/active-context")
def get_active_context(db: Session = Depends(get_db)):
    """Retrieve currently active dataset context for dashboard and AI agent."""
    active_ds = db.query(UploadedDataset).filter(UploadedDataset.is_active == True).first()
    if not active_ds:
        # Fallback to first available preset
        demo_ds = db.query(UploadedDataset).filter(UploadedDataset.dataset_id.like("ds_preset_%")).first()
        demo_report = json.loads(demo_ds.report_json) if (demo_ds and demo_ds.report_json) else {}
        return {
            "active_mode": "DEMO",
            "mode_label": demo_ds.mode_label if demo_ds else "Amazon E-Commerce Benchmark (12,000 Orders)",
            "dataset_name": demo_ds.filename if demo_ds else "amazon_ecommerce_demo.csv",
            "domain": demo_ds.domain if demo_ds else "ECOMMERCE_RETAIL",
            "row_count": demo_ds.row_count if demo_ds else 12000,
            "is_customer_intelligence": True,
            "report": demo_report,
        }

    report = json.loads(active_ds.report_json) if active_ds.report_json else {}
    is_preset = active_ds.dataset_id.startswith("ds_preset_")

    return {
        "active_mode": "DEMO" if is_preset else "UPLOAD",
        "preset_id": active_ds.dataset_id.replace("ds_preset_", "").upper() if is_preset else None,
        "dataset_id": active_ds.dataset_id,
        "dataset_name": active_ds.filename,
        "dataset_mode": active_ds.dataset_mode,
        "domain": active_ds.domain,
        "mode_label": active_ds.mode_label,
        "row_count": active_ds.row_count,
        "col_count": active_ds.col_count,
        "is_customer_intelligence": active_ds.dataset_mode in ("CUSTOMER_EVENT", "CUSTOMER_TRANSACTION", "CAMPAIGN_UPLIFT") or is_preset,
        "report": report,
    }
