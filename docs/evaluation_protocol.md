# Evaluation Protocol

Concrete checklist to run before any model's metrics are shown on the
Models page or used by the agent/recommendation engine. See
`docs/master_prompt.md` (round 2 additions, F–K) for the reasoning behind
each item.

## Churn / next-event models

- [ ] Time-based split used (no random shuffling across the churn label's
      observation window).
- [ ] Class balance reported (positive rate in train/val/test).
- [ ] PR-AUC reported as primary metric; ROC-AUC reported alongside, not
      instead of it.
- [ ] Decision threshold chosen from a stated cost assumption, not 0.5 by
      default.
- [ ] Cold-start customers excluded from the main model's training/eval
      set, or excluded and handled separately — never scored with
      undefined/zero-imputed history.
- [ ] Hyperparameters came from a logged Optuna (or equivalent) search, not
      hand-picked.
- [ ] `model_runs` row includes a dataset content hash + row count.

## Uplift model

- [ ] Two-model baseline implemented and evaluated first.
- [ ] At least one purpose-built method (X-learner / causal forest)
      compared against the baseline on Qini/AUUC.
- [ ] Uplift reported by decile, not only as a single average.
- [ ] Randomization assumption stated wherever uplift numbers are surfaced.
- [ ] Confidence interval or confidence band attached to each customer-level
      uplift estimate.

## Segmentation

- [ ] Multiple K evaluated with Silhouette / Davies-Bouldin /
      Calinski-Harabasz, not a hardcoded K.
- [ ] Cluster labels generated from actual cluster statistics vs. global
      population, not hand-written.

## Recommendation / next-best-action

- [ ] Offline policy backtest run before first demo (does the policy
      recommend a plausible action given historical context, does it vary
      across customers rather than defaulting to one action).
- [ ] Feedback loop (`recommendations` → actual outcome) wired even if
      thin at first — this is what eventually replaces the offline backtest
      with real A/B evidence.

## General

- [ ] Every number shown in the UI traces to a `model_runs` /
      `predictions` / `uplift_predictions` row — nothing hand-typed.
- [ ] Anything not yet meeting this checklist is marked `NOT IMPLEMENTED`
      in the UI rather than shown with placeholder numbers.
