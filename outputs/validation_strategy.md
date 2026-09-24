# Phase 3 & 4 — Validation Strategy + Naive Baseline

## Why chronological validation
The Kaggle test set is the **42 days immediately after** the last training day
(2015-06-20 → 2015-07-31). The model will be judged on its ability to forecast
the *future*. A random split (or K-fold) would leak future information into
training and produce an optimistic, meaningless score. Therefore **every split
places validation strictly after its training slice**, and the validation window
length equals the real horizon (42 days).

Implemented in `src/validation.py`.

## The metric — safe RMSLE
```
RMSLE = sqrt( mean( (log1p(pred) - log1p(true))^2 ) )
```
Defensive implementation (`rmsle`):
* predictions clipped to **≥ 0** before scoring (engineering rule #15);
* targets asserted **non-negative** (no invalid labels);
* shape mismatch / NaN raise instead of silently corrupting the score.

RMSLE penalises **relative** error and under-prediction more than
over-prediction — so getting small/medium hubs right matters as much as large
hubs, and log-space modelling (`log1p` target) is well aligned with it.

## Split design

### Main holdout (primary reporting window — your choice)
| | Train | Validation |
|---|---|---|
| dates | 2013-01-01 … **2015-05-08** | **2015-05-09 … 2015-06-19** (42 days) |
| rows | 923,549 | 46,830 |

The validation block is a full 1115 × 42 grid — **the same size and shape as the
Kaggle test set** — so local RMSLE is a faithful proxy for the leaderboard
(modulo the seasonal shift noted below).

### Rolling / expanding folds (stability check)
Three contiguous, non-overlapping 42-day validation windows with expanding
training history:

| Fold | Train ≤ | Validation |
|---|---|---|
| rolling_fold1 | 2015-02-13 | 2015-02-14 … 2015-03-27 |
| rolling_fold2 | 2015-03-27 | 2015-03-28 … 2015-05-08 |
| rolling_fold3 | 2015-05-08 | 2015-05-09 … 2015-06-19 (= main holdout) |

These tell us whether a model's edge is stable across different seasons rather
than a fluke of one window.

## Closed-row handling (your choice: hard-set to 0)
Because `IsOpen == 0 ⟹ OrderVolume == 0` deterministically in train, and
`IsOpen` is provided in the test file, predictions for closed rows are forced to
**exactly 0**, and models are trained on **open rows only**. Scoring is still
done over **all** validation rows (mirroring Kaggle), so the closed rows
contribute 0 error — a free, fully leakage-safe gain.

## Naive baseline results (Phase 4)
Statistics fit on each fold's **train slice, open rows only**; closed val rows → 0.

| Strategy | Main holdout RMSLE (all) | open-only | rolling range |
|---|---|---|---|
| global_median | 0.3655 | 0.4086 | 0.365–0.389 |
| hub_median | 0.2365 | 0.2644 | 0.236–0.277 |
| **hub_weekday_median** | **0.1838** | 0.2054 | **0.184–0.229** |
| hub_recent_mean (28d) | 0.2272 | 0.2540 | 0.225–0.262 |

**Anchor to beat: `hub_weekday_median` ≈ 0.1838.** Every subsequent model must
improve on this to earn its complexity. Closed-row RMSLE = 0.0 on all folds
(confirms the deterministic-zero handling).

Artifacts: `outputs/metrics/baseline_metrics.csv`.

## Why this resembles the Kaggle test
* Validation is the **last 42 days**, immediately preceding the true test window,
  same grid shape (1115 hubs × 42 days).
* No random shuffling; training never sees validation-period or future rows.
* Same metric (RMSLE), same closed-row convention we'll use at submission.

## Limitations / caveats
1. **Seasonal offset:** our holdout is May–June; the real test is late-June →
   July (school-holiday season). Demand seasonality differs slightly, so the
   leaderboard may sit a bit above local CV. The rolling folds partly probe this.
2. **No regional holidays in the test window** (all 0) — but our May–June holdout
   also has few, so the mismatch is small.
3. **Refurbishment gaps** (180 hubs) mean some rolling windows include hubs with
   sparse recent history; handled by gap-aware features in Phase 5.
4. Rolling folds share the *same* future-most window as the main holdout by
   construction (fold3 == holdout) — treat fold1/fold2 as the independent
   stability evidence.
