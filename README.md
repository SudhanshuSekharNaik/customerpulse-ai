# CustomerPulse AI

### Customer Decision Intelligence Dashboard

CustomerPulse AI turns raw e-commerce behavioral data into customer-level predictions, segments, lifecycle states, anomaly signals, and recommended actions — in a single dashboard.

🔗 **Live Demo:** [customerpulse-ai-ve3h.onrender.com](https://customerpulse-ai-ve3h.onrender.com/)

Rather than stopping at "what happened," the platform connects:

> **Behavior → Segmentation → Lifecycle State → Churn Prediction → Explanation → Recommended Action**

into one consistent customer view.

---

## 🚀 What It Does

Given a CSV of customer orders and events, CustomerPulse AI answers four questions for every customer:

1. **Who are they?** — behavioral segment, lifecycle stage, spend/order history
2. **Are they at risk?** — churn probability from a trained classifier
3. **Why?** — SHAP-based feature attribution for that prediction
4. **What should we do?** — a recommended next action based on risk, value, and lifecycle state

All of this rolls up into an executive view of total revenue, revenue in at-risk states, and lifecycle distribution across the customer base.

---

## ✨ Core Modules

| Module | What it shows |
|---|---|
| **Executive Pulse** | Revenue, active customers, lifecycle risk, recent behavior alerts |
| **Customer 360** | Per-customer segment, lifecycle state, spend, churn risk, opportunity score |
| **Segmentation** | K-Means clustering of customers into behavioral cohorts with a data-driven cluster-count selection |
| **State Machine** | 9-stage customer lifecycle with an empirical Markov transition matrix |
| **Predictions & SHAP** | Churn model outputs, validation metrics, and per-customer explanations |
| **Anomaly Radar** | Behavior-change detection (recency spikes, engagement collapse, spend outliers) with severity tiers |
| **Next-Best-Action** | Rule-based recommendation engine mapping customer state to a suggested action |
| **AI Analyst Studio** | Natural-language interface for querying customer data |
| **MLOps & Quality** | Model/data lineage and audit information |

---

## 📊 Example Dataset Profile

The dashboard is dataset-agnostic (upload any customer behavioral CSV), but the reference demo dataset used for development looks like this:

```text
Source Events        12,000
Customers             1,299
Purchase Events       1,284
Revenue           ₹36,638,759.90
```

---

## 🧩 Behavioral Segmentation

Customers are clustered on lifetime spend, visit recency, and purchase frequency using K-Means. The pipeline evaluates multiple candidate cluster counts (3–6) and reports silhouette score and cluster compactness for each, selecting the best-fit option rather than a fixed K.

On the reference dataset, 3 clusters were selected as best fit (silhouette 0.864):

```text
Segment #1 — Ultra-High-Value Spenders (Tier 2)   ~12% of customers
Segment #2 — Core Steady Customers                ~88% of customers
Segment #3 — Ultra-High-Value Spenders (Tier 1)    <1% of customers (strategic outliers)
```

Each segment includes its clustering basis, average spend/recency/frequency, top category affinity, and a recommended action strategy.

---

## 🔄 Customer Lifecycle & State Machine

Customers are assigned to one of 9 lifecycle stages based on recency, frequency, and spend:

```text
NEW → EXPLORING → ENGAGED → CONVERTING → LOYAL
                     ↓
                DECLINING → AT_RISK → DORMANT → RECOVERING
```

Each stage reports population share, average spend, average recency, and mean churn risk. A transition matrix (calculated from the dataset's own history, where sufficient temporal data exists) estimates the probability of a customer moving from one stage to another.

> **Note:** transition probabilities are only as reliable as the temporal depth of the underlying dataset. Where a dataset has limited purchase history, these should be read as illustrative rather than statistically robust Markov estimates.

---

## 🔮 Churn Prediction

A gradient-boosted classifier estimates churn probability per customer, validated on a temporal (out-of-time) holdout to avoid lookahead leakage.

Reference dataset validation:

```text
Validation PR-AUC     0.7281  (Good)
Validation ROC-AUC    0.6624  (High Discrimination)
Operating Threshold   20% (cost-calibrated at an assumed 5:1 cost ratio)
```

The operating threshold is adjustable, trading off between flagging more accounts (lower threshold) and flagging only extreme risk (higher threshold).

### SHAP Explanations

Each flagged customer includes the top features driving their score (e.g. `aov`, `recency_days`, `frequency_30d`) with signed contribution values, so risk scores are not black-box outputs.

---

## 🚨 Anomaly Radar

A rolling feed of behavioral deviations, each tagged with severity (LOW / MEDIUM / HIGH / CRITICAL), the metric involved (e.g. recency inactivity spike, engagement velocity collapse, monetary outlier), baseline vs. current value, percent deviation, and the detection method used.

---

## 🎯 Next-Best-Action

Maps each customer's segment, lifecycle state, and churn risk to a recommended action (e.g. `DISCOUNT`, `WIN_BACK`, `LOYALTY_REWARD`, VIP tier enrollment).

**This is currently a rule/eligibility-based recommendation engine, not an uplift model.** Action values (e.g. discount amounts) are template-driven rather than derived from causal treatment-effect estimation. If causal uplift modeling is added later, it should independently estimate `P(outcome | treated) − P(outcome | untreated)` per customer rather than being inferred from the README description alone — the roadmap below reflects this as a planned extension, not a current one.

---

## 🏗️ Architecture

```text
CSV Upload (orders / events / behavior)
              │
              ▼
     Feature Engineering
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
Segmentation  Churn ML   Anomaly
(K-Means)     (GBM)      Detection
   │          │          │
   └──────────┼──────────┘
              ▼
      Customer 360
  (segment + lifecycle + risk + SHAP)
              │
              ▼
      Next-Best-Action
     (rule-based recommender)
              │
              ▼
    Executive Dashboard
```

---

## ⚙️ Data Consistency

Every customer's churn score, segment, and lifecycle state should be read from the same canonical prediction table across the Customer 360, Predictions, Segmentation, and Actions views. If you're extending this project, it's worth adding an internal consistency check — e.g. an admin-only endpoint that compares the stored prediction against what each view renders for a given customer ID — since divergent per-page calculations are a common bug in dashboards that compute the same score in multiple places.

---

## 🔐 Design Principles

- **Read-only analytics** — the dashboard surfaces predictions and recommendations; it does not autonomously execute business actions.
- **Explainability over opacity** — every churn score ships with a SHAP-based explanation rather than a bare probability.
- **Transparent model quality** — validation metrics (PR-AUC, ROC-AUC) and clustering diagnostics (silhouette, compactness) are shown in-app rather than hidden.

---

## 📌 Known Limitations

- Next-Best-Action is rule-based, not causal — described honestly above rather than labeled "predicted uplift."
- Markov transition probabilities depend on the temporal depth of the uploaded dataset and may not be statistically robust for small or short-history datasets.
- Churn probabilities, segments, and lifecycle labels must stay in sync across all views (Customer 360, Predictions, Segmentation, Actions) — verify this after any pipeline change.
- This is a portfolio/prototype system and has not been hardened for production traffic, authentication, or multi-tenant use.

---

## 🔮 Roadmap

- True causal/uplift modeling to replace rule-based Next-Best-Action
- Probability calibration and cost-sensitive threshold tuning
- Automated cross-module consistency checks (debug endpoint comparing stored vs. rendered predictions)
- Role-based access control
- Model monitoring and drift detection

---

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/CustomerPulse-AI.git
cd CustomerPulse-AI

python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows

pip install -r requirements.txt
```

Configure environment variables in `.env` (only include what your implementation actually needs, e.g. database URL, model config).

```bash
# backend
python app.py

# frontend
npm install
npm run dev
```

Upload a CSV of customer orders/events through the dashboard to generate segments, lifecycle states, and predictions.

---

## 👨‍💻 Author

**Sudhanshu Sekhar Naik**
B.Tech — Information Technology

---

*A prediction is only useful when it's explainable, consistent, and tied to a concrete next step. CustomerPulse AI is built around that principle rather than around any single flashy metric.*
