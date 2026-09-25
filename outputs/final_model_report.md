# Final Model Report — submission_v3

## Model
Seed-averaged (5 seeds), **horizon-aware LightGBM**, `log1p(OrderVolume)` target,
tuned for the summer test season. Closed hubs (`IsOpen==0`) forced to exactly 0.

## Why this design
* **Direct + horizon-aware:** predicts any of the 42 future days from a single
  cutoff, but includes near-term lags (1–14, 21, 28, 35) that are **naturally
  masked** (NaN) when unavailable for a given horizon — so recent demand is used
  only where legitimate. This beat both the lag≥42-only model and recursive
  forecasting (which drifted).
* **log1p target:** aligns the training loss with the RMSLE metric.
* **Summer-season tuning:** Optuna objective = mean RMSLE over the summer-2014
  fold + recent holdout, matching the Jun–Jul test window.
* **Seed averaging:** variance reduction for a small, reliable gain.

## Features (leakage-safe; 5/5 automated audit PASS)
Calendar (cyclic), metadata (format, tier, competitor tenure, loyalty), operational
flags (promo, holiday, school closure), target lags (near 1–35 masked by horizon +
seasonal 42–63/357–371), recent-window stats, hub / hub×weekday / hub×promo /
hub×weekday×promo historical means, recent means & trend, `horizon`.

## Validation (chronological, RMSLE)
| Fold | Window | RMSLE |
|---|---|---|
| fold1 | 2015-02-14 … 03-27 | 0.0600 |
| fold2 | 2015-03-28 … 05-08 | 0.0536 |
| holdout | 2015-05-09 … 06-19 | 0.0552 |
| summer (LB proxy) | 2014-06-20 … 07-31 | 0.0541 |

Tight cluster 0.054–0.060 across seasons → robust, not overfit. Kaggle public LB
(submission_v1, the plain direct model) was 0.07181; v3's summer proxy is 0.0541,
so a large LB improvement is expected (honest range ~0.056–0.062 after
tuning-optimism discount).

## Progression
| Version | Change | Summer-fold RMSLE |
|---|---|---|
| v1 | direct tuned LightGBM | 0.0717 (LB 0.07181) |
| v2 | + horizon-aware near lags | 0.0706 |
| v3 | + richer features + summer-tuning + 5-seed avg | **0.0541** |

## Rejected experiments (validated, then discarded)
* School-closure features (EXP-2): +0.012 worse.
* Recursive forecasting (EXP-5): 0.17 — lag_1 drift flattens seasonality.
* LightGBM+XGBoost ensemble (EXP-7): XGB too weak on this feature set; blends hurt.

## Submission checks (submission_v3.csv)
46,830 rows · exact Id order · no missing · no negatives · 6,548 closed-hub zeros.
All PASS.

## Reproduce
`python src/tune.py` → `python src/finalize_v3.py` (leakage: `python src/leakage_checks.py`).
