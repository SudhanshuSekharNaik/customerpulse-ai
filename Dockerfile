# ==========================================
# STAGE 1: Build React Frontend (Vite)
# ==========================================
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ==========================================
# STAGE 2: Python Backend & ML Environment
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install system dependencies needed by LightGBM and C/C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY ai/ ./ai/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Copy built frontend assets from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Initialize datasets and baseline models during build
RUN python scripts/generate_ecommerce_benchmarks.py && \
    python scripts/ingest.py && \
    python scripts/build_features.py && \
    python scripts/train_churn.py && \
    python scripts/train_segmentation.py && \
    python scripts/train_uplift.py && \
    python scripts/train_next_event.py && \
    python scripts/generate_recommendations.py

# Expose default port
EXPOSE 8000
ENV PORT=8000
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Start FastAPI server
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
