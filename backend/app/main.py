"""CustomerPulse AI — Main FastAPI Application Entrypoint."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.database.session import init_db
from backend.app.api import (
    auth,
    customers,
    segments,
    behavior,
    predictions,
    uplift,
    recommendations,
    analytics,
    models,
    agent,
    health,
    universal,
    reports,
)

app = FastAPI(
    title="CustomerPulse AI",
    description="Full-stack AI-powered Customer Decision-Intelligence Platform API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(customers.router)
app.include_router(segments.router)
app.include_router(behavior.router)
app.include_router(predictions.router)
app.include_router(uplift.router)
app.include_router(recommendations.router)
app.include_router(analytics.router)
app.include_router(models.router)
app.include_router(agent.router)
app.include_router(health.router)
app.include_router(universal.router)
app.include_router(reports.router)

# Mount and serve built frontend React SPA if dist/ exists
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))

if os.path.exists(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith(("api", "docs", "redoc", "openapi.json")):
            return None
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        index_file = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "CustomerPulse AI API is running. Build frontend to view UI."}


@app.on_event("startup")
def on_startup():
    init_db()
    from backend.app.database.session import SessionLocal
    from backend.app.database.models import Customer
    db = SessionLocal()
    try:
        c_count = db.query(Customer).count()
        if c_count == 0:
            print("Populating initial Amazon E-Commerce Benchmark demo dataset...")
            demo_csv = "data/amazon_ecommerce_demo.csv"
            if not os.path.exists(demo_csv):
                from scripts.generate_amazon_demo import generate_amazon_demo_csv
                generate_amazon_demo_csv()
            import pandas as pd
            from ml.universal.semantic_detector import SemanticColumnDetector
            from ml.universal.pipeline import UniversalPipelineRunner
            df_demo = pd.read_csv(demo_csv)
            detected = SemanticColumnDetector.detect_all_columns(df_demo)
            mapping = {c["column_name"]: c["detected_semantic_type"] for c in detected}
            UniversalPipelineRunner.run_pipeline(df_demo, mapping, "amazon_ecommerce_demo.csv", dataset_id="ds_amazon_demo")
            print("Amazon demo dataset initialized successfully.")
    except Exception as e:
        print(f"Startup data init warning: {e}")
    finally:
        db.close()
    print("CustomerPulse AI database initialized successfully.")
