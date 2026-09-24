# CH'27 — Celebal Technologies × VIT Vellore: Dark-Store Demand Forecasting

Predict daily `OrderVolume` for each `HubID` over a 42-day future window
(2015-06-20 → 2015-07-31). **Metric: RMSLE.** Individual assessment — only the
provided datasets may be used, no external data.

## Problem in one line
Genuine **future time-series forecasting** (not random regression): the test
period is strictly after all training data, so validation must be
**chronological** — never a random split.

## Repo layout
```
data/                 raw CSVs (never modified)          [live on main branch]
src/
  data.py             reproducible loaders + structural constants   [Phase 1 ✓]
  validation.py       chronological CV + safe RMSLE                 [Phase 3]
  features.py         leakage-safe feature pipeline                 [Phase 5]
  models.py           LightGBM / CatBoost / XGBoost wrappers        [Phase 6]
  train.py            training driver + experiment logging          [Phase 6+]
  predict.py          inference + submission builder                [Phase 12]
  leakage_checks.py   automated leakage assertions                  [Phase 5/10]
outputs/
  data_audit.md              Phase-1 audit report                   ✓
  feature_risk_analysis.md   Phase-1 leakage matrix                 ✓
  validation_strategy.md     Phase-3 methodology
  eda/ error_analysis/ metrics/
models/               saved models
submissions/          submission_v1.csv, ...
notebooks/            exploratory notebooks
```

## Headline findings (Phase 1 — see `outputs/data_audit.md`)
1. **`IsOpen == 0` ⟹ `OrderVolume == 0`** deterministically → hard-set closed
   test rows to 0 (free, leakage-safe accuracy).
2. **`AppSessions` is train-only (absent from test)** → banned as a direct
   feature; it is the #1 leakage trap (log-log corr with target = 0.996).
3. Test is a clean **1115 hubs × 42 days** grid, 1 day after train ends.
4. Metadata joins 1:1 with no missing values introduced.

## Reproducibility
* Python 3.11, versions pinned in `requirements-lock.txt`.
* Fixed random seeds throughout; chronological (never random) splits.
* Raw CSVs are read-only; all artifacts are written under `outputs/`,
  `models/`, `submissions/`.

## Workflow phases
Audit → EDA → chronological validation → naive baseline → leakage-safe
features → benchmark LGBM/CatBoost/XGB (raw vs log1p) → importance/error
analysis → tune strongest → ensemble → strict leakage audit → train on all
eligible data → `submission_v1.csv` → post-leaderboard analysis.
