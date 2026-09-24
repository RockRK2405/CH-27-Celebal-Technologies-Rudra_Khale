# Phase 1 — Feature Risk & Leakage Analysis

The single rule for this competition: **to predict `OrderVolume` at date *t*, no
feature may use any information that would only be known on or after *t*.**
Every column below is classified against that rule.

---

## A. Safe features (known at prediction time, use freely)

These appear in **both** train and test, or are pure calendar / static metadata:

| Feature | Why safe |
|---|---|
| `HubID` | identity; static |
| `Weekday` | calendar (= `Date.dayofweek`+1); deterministic for any future date |
| `Date`-derived calendar: day-of-week, day-of-month, week-of-year, month, quarter, weekend flag, cyclic sin/cos | pure function of the date being forecast |
| `IsOpen` | **provided in test** for every row |
| `PromoActive` | **provided in test** (promotions are planned in advance) |
| `RegionalHoliday` | **provided in test** (calendar; all 0 in the test window) |
| `SchoolClosureFlag` | **provided in test** (calendar) |
| `HubFormat`, `AssortmentTier`, `LoyaltyProgram` | static per-hub metadata |
| `CompetitorDistance` (imputed) | static per-hub metadata |
| Competitor tenure at *t* (from `CompetitorOpenSinceMonth/Year`) | static anchor + the forecast date → known |
| Loyalty tenure / "is renewal month" at *t* (from Loyalty since-fields + `LoyaltyProgramInterval`) | static anchor + forecast date → known |

---

## B. Potentially risky features (usable ONLY with strict shifting)

| Feature | Risk | Safe construction |
|---|---|---|
| **Lag of `OrderVolume`** (lag-1…lag-42, seasonal 7/14/28) | uses the target | only lags with horizon ≥ prediction step; for the 42-day test, only lags ≥ 42 are available for *every* test day without recursion. Shorter lags require recursive/iterative forecasting or must be dropped for the direct-multi-step setup. |
| **Rolling stats of `OrderVolume`** (mean/median/std/min/max over 7/14/28) | uses the target | must be computed on `shift(≥horizon)` history so the current and future targets never enter the window |
| **Trend features** (recent-vs-long-run averages, growth) | derived from target | same shifting discipline as rolling |

> For a 42-day block forecast, the cleanest leakage-safe design is to build
> lag/rolling features from history **shifted by at least the forecast gap**,
> or to forecast recursively while never reading a future/unobserved target.
> This is the core of the Phase 5 pipeline and will be unit-tested.

---

## C. Features requiring special handling

| Feature | Issue | Plan |
|---|---|---|
| `CompetitorDistance` | 3 NaN; heavy right skew | impute (median or "no-competitor" sentinel) + `log1p`; add missing-flag |
| `CompetitorOpenSinceMonth/Year` | 31.7% NaN | derive `competitor_open_months = (year_t-Y)*12 + (month_t-M)`; NaN → treat as competitor absent/very-long-standing + missing-flag |
| `LoyaltyProgram*Since*` / `Interval` | 48.8% NaN = LP off | derive `loyalty_active_at_t`, `loyalty_tenure_weeks`, `is_loyalty_renewal_month`; NaN → 0 / off |
| `RegionalHoliday` | test has only value 0 | keep, but expect no signal from it on the leaderboard |
| Refurbishment gaps (180 hubs) | rolling windows straddle gaps | gap-aware windowing; count observed days in window |
| 54 open-but-zero rows | mild RMSLE stress | leave as-is; revisit in error analysis |

---

## D. Possible leakage sources (DO NOT use directly)

| Source | Verdict |
|---|---|
| **`AppSessions` (same-day)** | **BANNED as a direct feature** — absent from test, so it is not knowable at prediction time. log-log corr with target = 0.996, so it *will* fake a great local score and fail on Kaggle. Allowed only as *historical* aggregates (per hub / weekday) built from the training past. |
| **`OrderVolume` (same-day or future)** | the target — never a feature except via properly-shifted lags/rollings |
| Any statistic computed over the **full** series (global means, target encodings) without time-aware folding | would leak validation/future info → compute inside each fold's train slice only |
| Standardization / imputation fit on train+test together | fit transforms on training data only, then apply to test |

---

## E. Automated leakage checks to implement (Phase 5)

1. **Column presence:** assert no feature depends on a column missing from test (esp. `AppSessions`, `OrderVolume`).
2. **Temporal shift assertion:** for every lag/rolling feature, assert the max source date used for row *t* is `< t` (strictly).
3. **Fold isolation:** assert validation rows' features never read training rows dated `≥ validation start` when that would violate the horizon.
4. **Fit-on-train-only:** assert imputers/encoders are fit on the fold's train slice.
5. **Prediction sanity:** clip predictions to ≥ 0 (rule #15) and force 0 where `IsOpen == 0`.
