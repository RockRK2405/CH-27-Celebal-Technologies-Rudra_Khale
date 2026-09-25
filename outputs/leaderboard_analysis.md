# Phase 13 — Leaderboard vs Validation Analysis

**Kaggle public LB RMSLE:** 0.07181
**Local CV:** 0.06422 (main holdout, May 9–Jun 19) · 0.06248 (mean of folds 2+3)
**Gap:** +0.008 (LB slightly worse than CV) — small, positive, healthy.

## Is it better / worse / substantially different?
Slightly **worse** than validation, by a small margin. This is the *expected*
direction and magnitude. Crucially, it is **not** substantially different — an
LB far worse than CV would signal leakage; an LB far better would signal a test
overlap. Neither happened, so validation was honest and the model is not overfit.

## Root cause (evidence-based): summer school-holiday distribution shift

| Feature | Local holdout (May 9–Jun 19) | Kaggle test (Jun 20–Jul 31) |
|---|---|---|
| School-closure rows | 7.5% | **28.5%** (≈4×) |
| Test *dates* with any closure | 45% | 64% |
| Regional holidays | 6.4% | 0% |
| Promo active | 35.7% | 35.7% (same) |
| Closed (IsOpen=0) | 20.0% | 14.0% |
| Weekday mix | uniform | uniform (same) |

From Phase-7 error analysis, **school-closure days have RMSLE 0.161 vs 0.085**
on normal days. The test window is dominated by this weakest segment, which
accounts for essentially the entire CV→LB gap. Promo/weekday composition are
identical, so they are not contributors.

## Ranked experiments (one change each — awaiting approval before implementing)

### EXP-1 — Add a summer validation fold (highest priority)
* **Hypothesis:** our spring holdout under-represents the summer regime, so CV is
  optimistic for the real test. A same-season fold will track the LB.
* **Change:** add a chronological fold with validation = 2014-06-20 … 2014-07-31
  (train = everything before), mirroring the test calendar exactly.
* **Expected effect:** CV on that fold ≈ LB (~0.072); every later change is then
  judged against a faithful proxy. No model change yet.
* **Risk:** none (validation-only). Slightly less training history for that fold.
* **Evaluation:** compare fold RMSLE to the 0.07181 LB; expect close agreement.

### EXP-2 — School-closure-aware features (highest expected LB gain)
* **Hypothesis:** the model lacks signal specific to school-holiday demand.
* **Change:** add per-hub historical school-closure uplift (mean OrderVolume on
  SchoolClosureFlag==1 vs ==0), school-closure run-length, and
  school-closure×weekend — all knowable from the test calendar.
* **Expected effect:** lower RMSLE on the 28.5% closure rows → direct LB gain.
* **Risk:** low; small overfit risk on sparse per-hub closure history (mitigate
  with shrinkage/global fallback).
* **Evaluation:** school-closure-segment RMSLE on the EXP-1 summer fold.

### EXP-3 — Retune LightGBM on summer + recent folds
* **Hypothesis:** params tuned only on spring folds are mildly mis-specified for
  summer.
* **Change:** re-run the 12-trial Optuna with the objective = mean of the summer
  fold (EXP-1) and the main holdout.
* **Expected effect:** small, robust LB improvement; less season overfit.
* **Risk:** low; controlled search only.
* **Evaluation:** mean RMSLE across both folds; accept only if it also holds on
  the summer fold.

### EXP-4 — Leverage lag_364 for seasonality
* **Hypothesis:** last-July demand (lag ~364) is the best guide to this July.
* **Change:** add lag_364 interacted with SchoolClosureFlag, and a
  same-week-last-year mean.
* **Expected effect:** better level for the holiday weeks.
* **Risk:** low (lag_364 already leakage-audited).
* **Evaluation:** summer-fold RMSLE.

### EXP-5 — Recency-weighted / recursive short lags (higher effort)
* **Hypothesis:** the direct model can only use lags ≥ 42; nearer history would
  help early test days.
* **Change:** iterative/recursive forecasting to unlock lag-1…7 safely.
* **Expected effect:** possible gain on early test days; uncertain.
* **Risk:** higher — error accumulation, more complex leakage surface.
* **Evaluation:** summer-fold RMSLE vs the direct model; keep only if clearly better.

## Recommendation
Do **EXP-1 first** (free, makes CV honest for the test season), then **EXP-2**
(targets the dominant weak segment), then **EXP-3**. EXP-4/5 only if a gap
remains. One change at a time, each judged on the summer fold.
