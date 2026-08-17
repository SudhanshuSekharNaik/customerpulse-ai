# CustomerPulse AI

Full-stack AI-powered customer decision-intelligence platform: behavioral
customer states → prediction → uplift → next-best-action → explainable AI
agent.

This repository is a **starter scaffold**, not a finished implementation.
Every file under `backend/`, `ml/`, `ai/`, `pipelines/`, `monitoring/`,
`scripts/`, and `tests/` is a stub marked `# TODO`. The build spec — what
each module must do, in what order — lives in
[`docs/master_prompt.md`](docs/master_prompt.md).

## Why it's structured this way

The full spec is long by design (it covers data ingestion through MLOps).
Build it in the 12 phases listed in `docs/master_prompt.md` under
"Build order," validating real output at each phase before moving to the
next. Don't hand the whole spec to a coding agent in one shot — see
Addendum D in that file.

## Data

- **RetailRocket** (e-commerce events: view / add-to-cart / transaction) —
  primary behavioral dataset.
- **Criteo Uplift** (treatment/control) — uplift modeling only.

These are **not** the same customers and must never be joined as if they
were. Both datasets are used here for research/portfolio purposes; see
`docs/master_prompt.md` Addendum C before any commercial use.

## Repository layout

```
frontend/       React + TypeScript UI
backend/        FastAPI app (api/, services/, database/, security/)
ml/             ingestion, features, segmentation, prediction, uplift, etc.
ai/             agent + tools + prompts
pipelines/      Prefect/Airflow orchestration
monitoring/     drift monitoring (Evidently)
tests/          unit / integration / API / agent-tool / SQL-safety tests
docs/           master_prompt.md, evaluation_protocol.md, ml_methodology.md, model_cards.md
scripts/        reproducible CLI entry points (ingest, train, run_pipeline)
notebooks/      exploratory only — nothing production-critical lives here
docker/         Dockerfiles
```

## Getting started

```bash
cp .env.example .env
docker compose up
```

(`docker-compose.yml` currently only stands up Postgres — add `backend`,
`frontend`, and optional `mlflow`/`prefect` services as those phases are
built.)

## Status

Nothing is implemented yet — this commit is the skeleton plus the reviewed
spec. See `docs/master_prompt.md` for the full requirements, including two
rounds of additions:

- **Round 1** (architecture/process): agent model config, dataset scale
  guidance, licensing note, phased delivery, dev-sample flag.
- **Round 2** (model quality): class imbalance handling, cold-start
  customers, uplift beyond the two-model baseline, confidence intervals on
  every estimate, hyperparameter tuning + reproducibility, offline policy
  evaluation for recommendations.

Before treating any model's numbers as final, run them through
`docs/evaluation_protocol.md` — it's the checklist version of round 2.
