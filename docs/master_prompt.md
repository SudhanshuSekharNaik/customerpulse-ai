# CustomerPulse AI — Master Build Prompt (Refined)

> This is the original master prompt with five additions applied at the end
> (marked `ADDED`). Everything above is unchanged in substance from the
> source spec; only formatting was normalized to Markdown.

**Tagline:** "Understand what customers are doing. Predict what happens next. Decide what to do."

## Positioning

CustomerPulse AI is a full-stack AI-powered customer decision-intelligence
platform that learns behavioral customer states, detects meaningful changes,
predicts future customer behavior, estimates heterogeneous intervention
response, and recommends next-best-actions through an explainable ML
pipeline and a tool-using AI analytics agent.

Do NOT describe it as "a customer churn prediction system." Do NOT build a
notebook, a generic chatbot, or a static dashboard. Do NOT copy an existing
project's architecture.

## Core loop

```
CUSTOMER STATE → UNDERSTAND → PREDICT → DETECT CHANGE →
ESTIMATE INTERVENTION IMPACT → SELECT NEXT-BEST-ACTION →
EXPLAIN WITH AI → MEASURE OUTCOME
```

## Datasets

- **RetailRocket** — primary behavioral dataset (views, cart adds,
  transactions). Drives segmentation, states, journeys, next-event and churn
  prediction, behavior-change detection, opportunity scoring.
- **Criteo Uplift** — treatment/control dataset. Drives the uplift model
  only. **Never join RetailRocket customers to Criteo customers** — they are
  different populations. Label every uplift output "estimated," trained on
  Criteo treatment/control data.

## Architecture

```
DATA SOURCES → INGESTION → VALIDATION → PostgreSQL → CUSTOMER 360
→ FEATURE ENGINEERING → {SEGMENTATION, STATE ENGINE, PREDICTION}
→ BEHAVIOR MONITOR → CHANGE/ANOMALY DETECTION → OPPORTUNITY SCORING
→ UPLIFT MODEL → NEXT-BEST-ACTION ENGINE → {WEB DASHBOARD, AI ANALYST AGENT}
```

## Database (PostgreSQL)

Tables: `customers`, `events`, `products`, `customer_features`,
`customer_segments`, `customer_states`, `predictions`, `behavior_changes`,
`uplift_predictions`, `recommendations`, `model_runs`, `model_metrics`,
`agent_runs`, `agent_tool_calls`, `audit_logs`. (Full column lists: see
schema section retained from original spec — unchanged.)

## Modeling requirements

- **Segmentation:** K-Means primary, evaluate HDBSCAN and GMM. Try multiple
  K, report Silhouette / Davies-Bouldin / Calinski-Harabasz for each — never
  hardcode K=5. Auto-generate segment labels from cluster statistics, don't
  hand-write "Cluster 0 = VIP."
- **State engine:** NEW / EXPLORING / ENGAGED / CONVERTING / LOYAL /
  DECLINING / AT_RISK / DORMANT / RECOVERING, derived from real feature
  thresholds/statistics, with transition probabilities computed from actual
  data.
- **Next-event & churn prediction:** Logistic Regression baseline →
  LightGBM/XGBoost primary. Time-aware validation, no random splits that
  leak the future into the past. Report ROC-AUC, PR-AUC, precision, recall,
  F1, calibration, confusion matrix.
- **Explainability:** SHAP for every material prediction — computed, never
  hand-written.
- **Behavior-change detection:** rolling stats / z-score / EWMA /
  change-point / Isolation Forest against each customer's own baseline.
- **Uplift:** two-model (or better) approach on Criteo data, uplift/Qini/AUUC
  metrics, explicit "estimated treatment uplift" labeling, no causal claims
  beyond what the experimental data supports.
- **Next-best-action:** rank candidate actions (NO_ACTION, PRODUCT_
  RECOMMENDATION, REMINDER, EMAIL, DISCOUNT, LOYALTY_REWARD, WIN_BACK,
  CROSS_SELL, UPSELL) from state + value + risk + next-event + uplift + cost.
- **No fabrication, anywhere.** If data is missing: "Insufficient data to
  determine this," not an invented number.

## AI Agent — "CustomerPulse Analyst"

Real tool-using agent, not a wrapper chatbot. Tool families: customer,
segment, ML, analytics, decision, and one **read-only** SQL tool that
rejects INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE/GRANT/REVOKE outright
and returns "Unsafe SQL operation rejected."

Every agent answer must separate **Observed Data / Model Prediction / Model
Estimate / Recommendation**, show concise analysis steps (not raw
chain-of-thought), and never invent customers, revenue, predictions,
segments, or campaign outcomes.

## Stack

React+TypeScript frontend · FastAPI backend · PostgreSQL · Python ML ·
Prefect orchestration · MLflow registry · Evidently drift monitoring ·
Docker Compose · GitHub Actions CI.

## Build order

1. Data layer 2. Customer 360 3. Segmentation 4. State engine
5. Behavior + churn models 6. Uplift model 7. Next-best-action
8. FastAPI 9. React UI 10. AI agent 11. MLOps 12. Deployment

Do not advance a phase until the previous one produces real, validated
output. Anything unimplemented must be labeled `NOT IMPLEMENTED` in the UI —
never mocked to look finished.

## Acceptance tests (abbreviated)

Find customer → view 360 → segment → state → behavior changes → next-event →
churn → SHAP → uplift → next-best-action; ask the agent "why is this
customer at risk," "which customers should we target," "why are high-value
customers declining," "compare two segments," a filtered SQL-style question,
and finally attempt a destructive SQL statement through the agent — it must
be rejected.

---

## ADDED — fixes applied in this review

### A. Agent model configuration (was unspecified)

Name the LLM and tool-calling framework explicitly rather than leaving the
builder to guess:

- Model: Claude (via the Anthropic Messages API), using native tool use for
  the agent's tool calls.
- Keep a system prompt file (`ai/prompts/system_prompt.txt`) as the single
  source of truth for agent behavior/guardrails — the backend loads it, it
  is never inlined ad hoc in code.
- Set an explicit per-query tool-call budget (e.g., max 8 tool calls per
  agent turn) and a max-token budget, and log both to `agent_runs` so cost
  is auditable.
- Cache tool results within a single agent turn to avoid redundant calls
  (e.g., don't re-fetch the same customer twice in one reasoning chain).

### B. Dataset scale guidance (was unspecified)

RetailRocket (~2.75M events) is workable locally as-is. Criteo Uplift
(~25M rows) is not — training on the full file on every dev iteration will
dominate build time.

- Local dev/test: stratified sample of Criteo (document the sample size and
  sampling method in `docs/ml_methodology.md`).
- Final/reported metrics: train on the full dataset at least once and record
  that run's `model_runs` entry as the canonical one used for reported
  numbers.
- Never silently swap which run's numbers are shown in the UI — the Models
  page must show which dataset size backs the displayed metrics.

### C. Licensing / ToS note (was missing)

Both datasets carry usage restrictions (RetailRocket: Kaggle terms,
non-commercial; Criteo Uplift: released for research use). Add a line to the
README's "Data limitations" section stating the datasets are used for
research/portfolio purposes only, not for training a production commercial
model, and link the original dataset pages rather than redistributing the
raw files in the repo.

### D. Phased prompt delivery (structural risk, not a spec gap)

Don't paste all ~85 sections into one agent turn — at that length even a
capable coding agent tends to skim the back half. Deliver this document in
the same 12 phases listed under "Build order," pasting only the relevant
section(s) plus this file's architecture/positioning header each time, and
require the agent to report actual outputs (row counts, metric values,
cluster scores) before you hand it the next phase.

### E. Dev cap or scope switch (added for clarity)

Add a `--sample` flag to `scripts/ingest.py` and `scripts/train_uplift.py`
(e.g., `--sample 0.1`) so the same scripts serve both fast local dev and the
full canonical training run — rather than maintaining two separate code
paths.

---

## ADDED — round 2: model-quality fixes

These target the modeling itself, not just the architecture. The original
spec says "don't fabricate metrics" but doesn't say enough about *getting
good metrics in the first place*. Six gaps closed:

### F. Class imbalance in churn/inactivity prediction

Churn/inactivity is almost always a minority class (often 5–15% positive
in retail event data). Left unhandled, a model can hit 90%+ accuracy by
predicting "not churned" for everyone — useless in practice.

- Report **PR-AUC as the primary metric**, not accuracy or ROC-AUC alone;
  ROC-AUC is optimistic under imbalance.
- Try class weighting (`scale_pos_weight` in XGBoost/LightGBM) before
  resampling. If resampling, use SMOTE/undersampling only on the *training*
  fold, never on validation/test — resampling before a time-based split
  leaks information.
- Pick the decision threshold by business cost (cost of a missed churner vs.
  cost of an unnecessary intervention), not by defaulting to 0.5. Document
  the chosen threshold and the cost assumption behind it.

### G. Cold-start customers

New customers (first purchase < 30 days ago, or fewer than N events) don't
have stable RFM/velocity features — models trained mostly on established
customers will misfire on them.

- Segment customers into `established` vs `cold_start` before feature
  generation. Cold-start customers get a separate, simpler rule-based or
  low-feature-count model (or explicit `INSUFFICIENT_HISTORY` state) rather
  than being scored by the main churn/next-event model with garbage
  features.
- Never silently impute a cold-start customer's missing history as zero —
  zero recency/frequency looks identical to "engaged and about to buy" in
  some features. Flag it instead.

### H. Uplift modeling: go beyond the two-model baseline

The two-model approach (P(Y|T) − P(Y|C)) is a reasonable baseline but is
known to compound the error of both models. After the baseline is working
and evaluated:

- Add at least one purpose-built uplift method — X-learner or a causal
  forest (e.g., via `causalml` or `econml`) — and compare against the
  two-model baseline on Qini/AUUC before picking a production model.
- Report **uplift by decile**, not just an average uplift number — this is
  what actually drives "who do we target" decisions and is far more
  convincing in a demo than a single aggregate percentage.
- State the randomization assumption the Criteo experiment relies on
  (treatment was randomly assigned) explicitly wherever uplift numbers are
  shown — that assumption is exactly what makes uplift ≠ correlation.

### I. Confidence intervals, not just point estimates

Sections 20–24 of the spec show single-number outputs ("+22% uplift").
Point estimates without uncertainty overstate confidence, especially for
customers with thin behavioral history.

- Every churn probability, uplift estimate, and revenue-impact figure
  should carry an interval (bootstrap CI for uplift; predicted-probability
  calibration bands for churn) or, at minimum, a low/medium/high confidence
  label derived from prediction variance / sample size backing that
  customer's segment — not a fixed "Medium" placeholder.

### J. Hyperparameter tuning + reproducibility

The original spec says "evaluate candidate cluster counts" but doesn't
require systematic tuning for the supervised models.

- Use Optuna (or equivalent) for LightGBM/XGBoost tuning with a fixed
  random seed and a logged search space, tracked in MLflow alongside the
  resulting metrics — not hand-picked hyperparameters.
- Pin dataset version (row count + a content hash) in `model_runs` so a
  metric reported on the Models page is reproducible from that exact data
  snapshot, not "whatever ingestion produced most recently."

### K. Offline policy evaluation for the recommendation engine

Section 56 says to "evaluate" recommendations but the only concrete method
listed is comparing to actual outcomes after the fact — which only works
once you have live data. Add an **offline** check before anything ships:

- Backtest the next-best-action policy against the historical record: for
  customers who happened to receive something equivalent to a modeled
  action, check whether the policy would have picked the same action, and
  whether predicted uplift correlates with their actual subsequent
  behavior.
- This is weaker evidence than a live A/B test, but it catches an
  obviously-broken policy (e.g., one that always recommends the same
  action regardless of input) before it's demoed or deployed.
