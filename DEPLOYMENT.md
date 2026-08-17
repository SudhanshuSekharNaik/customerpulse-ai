# CustomerPulse AI — Render Deployment Guide 🚀

CustomerPulse AI is configured for **zero-configuration, 1-click deployment on [Render](https://render.com/)** as a unified full-stack web service (FastAPI Backend + React Vite SPA + ML Engines).

---

## 📁 Deployment Configuration Files Included

- **[`render.yaml`](./render.yaml)**: Render Infrastructure-as-Code (Blueprint) specifying runtime, environment variables, build steps, and health check.
- **[`build.sh`](./build.sh)**: Executable build script that installs Python dependencies, builds the Vite frontend bundle, generates benchmark datasets, and trains baseline ML models.
- **[`requirements.txt`](./requirements.txt)**: Pinned production dependencies for FastAPI, LightGBM, SHAP, Optuna, Scikit-Learn, and SQLAlchemy.
- **[`Dockerfile`](./Dockerfile)**: Production-ready multi-stage Dockerfile for containerized deployment.
- **[`Procfile`](./Procfile)**: Process declaration for Render web process.

---

## ⚡ Method 1: 1-Click Render Blueprint (Recommended)

1. **Push your code to GitHub / GitLab**:
   ```bash
   git init
   git add .
   git commit -m "Deploy CustomerPulse AI to Render"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/customerpulse-ai.git
   git push -u origin main
   ```

2. **Open Render Dashboard**:
   - Navigate to [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints).
   - Click **New Blueprint Instance**.
   - Connect your GitHub repository.
   - Render will automatically detect `render.yaml` and configure the service.
   - Click **Apply**.

---

## 🛠️ Method 2: Manual Render Web Service Setup

1. Go to [dashboard.render.com](https://dashboard.render.com/) and click **New +** > **Web Service**.
2. Connect your repository.
3. Configure the settings:
   - **Name**: `customerpulse-ai`
   - **Region**: Nearest to your users (e.g., `Oregon`, `Frankfurt`, `Singapore`)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `./build.sh` (or `chmod +x build.sh && ./build.sh`)
   - **Start Command**: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables**:
   - `PYTHON_VERSION`: `3.11.9`
   - `NODE_VERSION`: `20.12.0`
   - `ENVIRONMENT`: `production`
   - `SECRET_KEY`: *(Generate any random 32-character string)*
5. **Health Check Path**:
   - Set to `/api/health`
6. Click **Create Web Service**.

---

## 🐳 Method 3: Containerized Docker Deployment

If you prefer containerized deployment:
1. Go to **New +** > **Web Service**.
2. Select **Runtime**: `Docker`.
3. Render will automatically build the multi-stage `Dockerfile` (compiling the React frontend and setting up the Python ML environment).
4. Click **Create Web Service**.

---

## 🌐 Live Verification Endpoints

Once deployed, your service will be live at `https://customerpulse-ai.onrender.com` (or your chosen URL):

| URL Path | Description |
| :--- | :--- |
| `/` | Interactive React + Vite Dashboard & Decision Intelligence Platform |
| `/docs` | Interactive Swagger API Documentation |
| `/api/health` | Automated System & Database Health Check |
| `/api/reports/pdf` | Audit-ready Executive Decision PDF Export |
