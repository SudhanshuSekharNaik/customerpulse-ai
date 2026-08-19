<div align="center">

# CustomerPulse AI

### your customers, decoded before they churn.

upload a csv. get segments, lifecycle stages, churn risk, *why* the model
thinks that, and what to actually do about it — in one dashboard, not four.

[![live demo](https://img.shields.io/badge/demo-customerpulse--ai--ve3h.onrender.com-B4FF39?style=for-the-badge&logo=render&logoColor=black)](https://customerpulse-ai-ve3h.onrender.com/)
[![pr-auc](https://img.shields.io/badge/PR--AUC-0.7281-B4FF39?style=for-the-badge)](#-churn-prediction)
[![roc-auc](https://img.shields.io/badge/ROC--AUC-0.6624-8A8A9E?style=for-the-badge)](#-churn-prediction)
[![silhouette](https://img.shields.io/badge/silhouette-0.864-8A8A9E?style=for-the-badge)](#-behavioral-segmentation)
[![explainability](https://img.shields.io/badge/every_score-SHAP--explained-FF5C7A?style=for-the-badge)](#-churn-prediction)

**[▶ try it live](https://customerpulse-ai-ve3h.onrender.com/)** · [what it does](#-what-it-does) · [modules](#-core-modules) · [architecture](#-architecture) · [run it locally](#-getting-started)

</div>

<br>

> render free tier — first load after idle takes a beat to spin up.
> the model's not slow, the server's just waking up.

<br>

## 🧠 the pitch

most churn dashboards give you a red number and call it a day. that's not
a decision, that's anxiety.

CustomerPulse AI connects the whole chain instead of stopping at "here's a
score":

```
behavior  →  segmentation  →  lifecycle state  →  churn prediction  →  explanation  →  recommended action
```

every customer gets a *who they are*, a *where they're headed*, a *why the
model flagged them*, and a *what to do about it* — read from one consistent
record, not recomputed differently on every page.

<br>

## ⚙️ what it does

given a csv of customer orders/events, four questions get answered per customer:

| question | answer comes from |
|:--|:--|
| **who are they?** | behavioral segment + lifecycle stage + spend history |
| **are they at risk?** | a trained churn classifier |
| **why?** | SHAP feature attribution — not a black box |
| **what now?** | a recommended action based on risk, value, and lifecycle state |

all of it rolls up into one executive view — total revenue, revenue sitting
in at-risk states, lifecycle distribution across the whole base.

<br>

## 📦 core modules

<details>
<summary><b>📈 executive_pulse</b> — the 10-second read</summary>
<br>

revenue, active customers, lifecycle risk exposure, recent behavior alerts.
the page you actually open every morning.

</details>

<details>
<summary><b>👤 customer_360</b> — everything about one person, one place</summary>
<br>

segment, lifecycle state, spend, churn risk, opportunity score — per
customer, pulled from the same canonical prediction table every other
module reads from. no page shows a different number than another.

</details>

<details>
<summary><b>🧩 segmentation</b> — cohorts, not guesses</summary>
<br>

K-Means clustering on lifetime spend, visit recency, and purchase
frequency. tests 3–6 cluster counts, scores each on silhouette + cluster
compactness, and picks the best fit instead of hardcoding a K.

on the reference dataset: **3 clusters, silhouette 0.864**

```
segment #1 — ultra-high-value spenders (tier 2)   ~12% of customers
segment #2 — core steady customers                ~88% of customers
segment #3 — ultra-high-value spenders (tier 1)    <1%  (strategic outliers)
```

</details>

<details>
<summary><b>🔄 state_machine</b> — 9-stage lifecycle</summary>
<br>

```
NEW → EXPLORING → ENGAGED → CONVERTING → LOYAL
                     ↓
                DECLINING → AT_RISK → DORMANT → RECOVERING
```

each stage reports population share, avg spend, avg recency, mean churn
risk. transition probabilities are computed from the dataset's own
history — **and are only as reliable as how much history you actually
gave it.** short-history datasets get illustrative numbers, not robust
Markov estimates, and this repo says so instead of pretending otherwise.

</details>

<details>
<summary><b>🔮 predictions_&_shap</b> — scores you can argue with</summary>
<br>

gradient-boosted churn classifier, validated on a temporal holdout
(no lookahead leakage). every flagged customer ships with the top
features driving their score — `aov`, `recency_days`, `frequency_30d`,
signed contributions — so "why" is never a mystery.

```
validation PR-AUC     0.7281   (good)
validation ROC-AUC    0.6624   (high discrimination)
operating threshold   20%      (cost-calibrated, 5:1 assumed cost ratio)
```

threshold's adjustable — lower it to catch more accounts, raise it to
flag only the extreme cases.

</details>

<details>
<summary><b>🚨 anomaly_radar</b> — behavior drift, tagged and timestamped</summary>
<br>

rolling feed of behavioral deviations — recency spikes, engagement
collapse, spend outliers — each tagged LOW / MEDIUM / HIGH / CRITICAL,
with baseline vs. current value, % deviation, and detection method shown.

</details>

<details>
<summary><b>🎯 next_best_action</b> — rule-based, and honest about it</summary>
<br>

maps segment + lifecycle state + churn risk → a recommended action
(`DISCOUNT`, `WIN_BACK`, `LOYALTY_REWARD`, VIP enrollment, etc.).

**this is a rule/eligibility engine, not an uplift model.** action values
are template-driven, not derived from causal treatment-effect estimation.
that distinction matters and this repo doesn't blur it — true uplift
modeling is on the [roadmap](#-roadmap), not pretending to already exist.

</details>

<details>
<summary><b>💬 ai_analyst_studio</b> — ask it instead of filtering it</summary>
<br>

natural-language interface over the customer data — skip the filter
dropdowns, just ask.

</details>

<details>
<summary><b>🔍 mlops_&_quality</b> — the receipts</summary>
<br>

model/data lineage and audit trail. because "trust me" isn't a validation
strategy.

</details>

<br>

## 📊 reference dataset profile

dashboard is dataset-agnostic — upload any customer behavioral csv. the
reference dataset used during development:

```
source events        12,000
customers              1,299
purchase events         1,284
revenue           ₹36,638,759.90
```

<br>

## 🏗️ architecture

```
                    CSV upload (orders / events / behavior)
                                    │
                                    ▼
                          feature engineering
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
            segmentation        churn ML          anomaly
             (K-Means)            (GBM)           detection
                  │                 │                 │
                  └─────────────────┼─────────────────┘
                                    ▼
                             customer 360
                (segment + lifecycle + risk + SHAP)
                                    │
                                    ▼
                            next-best-action
                          (rule-based recommender)
                                    │
                                    ▼
                          executive dashboard
```

<br>

## 🔐 design principles

- **read-only analytics.** the dashboard surfaces predictions and recommendations — it does not autonomously execute business actions
- **explainability over opacity.** every churn score ships with a SHAP-based explanation, never a bare probability
- **transparent model quality.** validation metrics and clustering diagnostics are shown in-app, not hidden behind a "trust the AI" wall

<br>

## ⚠️ known limitations

- next-best-action is rule-based, not causal — labeled honestly, not dressed up as "predicted uplift"
- Markov transition probabilities depend on dataset temporal depth — small/short-history datasets get illustrative numbers, not statistically robust ones
- churn scores, segments, and lifecycle labels must stay in sync across every view (Customer 360, Predictions, Segmentation, Actions) — worth an internal consistency check (e.g. an admin endpoint comparing stored vs. rendered predictions per customer ID) if you extend this, since divergent per-page calculations are a classic dashboard bug
- portfolio/prototype system — not hardened for production traffic, auth, or multi-tenant use

<br>

## 🗺️ roadmap

- [ ] true causal/uplift modeling to replace the rule-based recommender
- [ ] probability calibration + cost-sensitive threshold tuning
- [ ] automated cross-module consistency checks
- [ ] role-based access control
- [ ] model monitoring + drift detection

<br>

## 🚀 getting started

```bash
git clone https://github.com/<your-username>/CustomerPulse-AI.git
cd CustomerPulse-AI

python -m venv venv
source venv/bin/activate       # windows: venv\Scripts\activate

pip install -r requirements.txt
```

drop what your implementation actually needs into `.env` (db url, model config, etc).

```bash
# backend
python app.py

# frontend
npm install
npm run dev
```

upload a customer orders/events csv → get segments, lifecycle states, and
predictions back.

or skip all that → **[customerpulse-ai-ve3h.onrender.com](https://customerpulse-ai-ve3h.onrender.com/)**

<br>

---

<div align="center">

**Sudhanshu Sekhar Naik**
b.tech — information technology

*a prediction only matters if it's explainable, consistent, and tied to a
concrete next step. that's the whole design brief.*

</div>
