# Phase 7 — Error Analysis

**Best model:** `lgbm` (log1p target) | **holdout RMSLE (all)** = `0.08404` | open-only = `0.09395`

## RMSLE by segment (open rows)

### by day of week

|   segment |         n |   rmsle |   mean_actual |
|----------:|----------:|--------:|--------------:|
|    4.0000 | 6688.0000 |  0.1295 |     7492.7035 |
|    6.0000 |  193.0000 |  0.1141 |     8183.9637 |
|    5.0000 | 6688.0000 |  0.0895 |     6125.0405 |
|    3.0000 | 4923.0000 |  0.0864 |     6859.8661 |
|    2.0000 | 6687.0000 |  0.0863 |     7216.2221 |
|    0.0000 | 5606.0000 |  0.0828 |     8618.5940 |
|    1.0000 | 6688.0000 |  0.0749 |     7448.6564 |

### by month

|   segment |          n |   rmsle |   mean_actual |
|----------:|-----------:|--------:|--------------:|
|    5.0000 | 19144.0000 |  0.1017 |     7012.1676 |
|    6.0000 | 18329.0000 |  0.0851 |     7560.2865 |

### by promotion

|   segment |          n |   rmsle |   mean_actual |
|----------:|-----------:|--------:|--------------:|
|    0.0000 | 21437.0000 |  0.1023 |     6400.7718 |
|    1.0000 | 16036.0000 |  0.0815 |     8455.9791 |

### by holiday

|   segment |          n |   rmsle |   mean_actual |
|----------:|-----------:|--------:|--------------:|
|    1.0000 |   152.0000 |  0.2673 |     8783.3553 |
|    0.0000 | 37321.0000 |  0.0926 |     7274.1448 |

### by school closure

|   segment |          n |   rmsle |   mean_actual |
|----------:|-----------:|--------:|--------------:|
|    1.0000 |  3216.0000 |  0.1613 |     7445.6835 |
|    0.0000 | 34257.0000 |  0.0849 |     7264.7374 |

### by hub format

|   segment |          n |   rmsle |   mean_actual |
|----------:|-----------:|--------:|--------------:|
|    3.0000 |  4962.0000 |  0.1398 |     7125.9327 |
|    2.0000 |   714.0000 |  0.0934 |    11171.4538 |
|    4.0000 | 11597.0000 |  0.0862 |     7312.3516 |
|    1.0000 | 20200.0000 |  0.0837 |     7162.2173 |

### by assortment tier

|   segment |          n |   rmsle |   mean_actual |
|----------:|-----------:|--------:|--------------:|
|    1.0000 | 19832.0000 |  0.1004 |     6839.7764 |
|    3.0000 | 17263.0000 |  0.0861 |     7725.1333 |
|    2.0000 |   378.0000 |  0.0822 |    10074.0847 |

### by hub volume tier

| segment   |     n |   rmsle |   mean_actual |
|:----------|------:|--------:|--------------:|
| high      | 12542 |  0.1050 |     9870.2649 |
| low       | 12472 |  0.0918 |     5037.8333 |
| mid       | 12459 |  0.0838 |     6917.7870 |

### by hub maturity

| segment      |     n |   rmsle |   mean_actual |
|:-------------|------:|--------:|--------------:|
| mature(>=1y) | 37473 |  0.0939 |     7280.2665 |

## Demand-level slices (open rows)

* near-zero (<=5th pct, y<= 3487): n=1874, RMSLE=0.2211
* high demand (>=95th pct, y>= 12822): n=1874, RMSLE=0.0833

## Worst 15 hubs by RMSLE

|     HubID |       n |   rmsle |   mean_actual |
|----------:|--------:|--------:|--------------:|
|  971.0000 | 30.0000 |  1.4228 |     9484.2333 |
|  415.0000 | 34.0000 |  0.4077 |     6430.9118 |
|  183.0000 | 26.0000 |  0.3273 |     7061.1923 |
|   39.0000 | 33.0000 |  0.2307 |     5659.8182 |
|  917.0000 | 33.0000 |  0.2030 |     6817.7273 |
|  310.0000 | 42.0000 |  0.2014 |     8238.6905 |
|  524.0000 | 42.0000 |  0.1945 |     7099.5238 |
|  286.0000 | 33.0000 |  0.1840 |     3830.4242 |
|  657.0000 | 33.0000 |  0.1671 |     6316.6667 |
|  170.0000 | 33.0000 |  0.1671 |     4164.6061 |
|  517.0000 | 34.0000 |  0.1645 |     5980.4118 |
|  433.0000 | 42.0000 |  0.1629 |     4944.4048 |
| 1081.0000 | 42.0000 |  0.1613 |     6465.1429 |
|  209.0000 | 42.0000 |  0.1573 |     5375.1667 |
| 1017.0000 | 34.0000 |  0.1555 |     6768.2059 |

## Feature importance (top 20, best model)

|                 |   importance |
|:----------------|-------------:|
| HubID           |        16687 |
| month           |         3469 |
| day             |         3408 |
| doy_cos         |         2576 |
| dayofyear       |         2502 |
| doy_sin         |         2126 |
| hub_last_value  |         1974 |
| dow             |         1313 |
| hub_trend_ratio |         1199 |
| lag_364         |         1106 |
| week            |         1099 |
| lag_42          |         1055 |
| lag_56          |          971 |
| hub_promo_mean  |          947 |
| year            |          929 |
| lag_49          |          869 |
| hub_wd_mean     |          774 |
| hub_wd_median   |          739 |
| dow_sin         |          719 |
| lag_371         |          713 |

---

## Redundant / suspicious / leakage review
* **Redundant:** `hub_appsessions_mean`, `hub_wd_appsessions_mean` (not in top-20; collinear with order-history aggregates) — candidates to drop. `dayofyear`/`doy_sin`/`doy_cos`/`month`/`day` overlap but are cheap for trees.
* **Suspiciously powerful:** `HubID` importance dwarfs others — this is *identity encoding*, not leakage (it just lets the tree memorise per-hub level). Confirmed safe by the perturbation audit.
* **Leakage:** none. Every high-importance feature (`hub_last_value`, `lag_364`, `lag_42`, `hub_trend_ratio`) references only data ≤ cutoff and survived the perturbation test.

## Next highest-value experiments (evidence-driven, one change each)
1. **Holiday-aware features** — *Hypothesis:* RMSLE on `RegionalHoliday==1` is 0.267 (3× worse). *Change:* add days-to/from-holiday and per-hub holiday-uplift historical means. *Eval:* segment RMSLE on holiday rows across folds. (Note: test window has no regional holidays, so this helps CV robustness more than the LB.)
2. **School-closure uplift** — *Hypothesis:* school-closure rows RMSLE 0.161 and the **test window contains school-closure days**. *Change:* per-hub `school_closure` historical mean + interaction with weekend. *Eval:* school-closure segment RMSLE.
3. **Friday / weekday-shape features** — *Hypothesis:* Friday (dow=4) RMSLE 0.130 worst weekday. *Change:* per-(hub,dow) trend & std, and promo×dow historical means. *Eval:* per-dow RMSLE.
4. **Near-zero demand handling** — *Hypothesis:* bottom-5% demand rows RMSLE 0.221 dominate the metric. *Change:* add per-hub low-demand frequency + a Tweedie objective trial. *Eval:* near-zero-slice RMSLE.
5. **HubFormat-3 focus** — *Hypothesis:* format 3 RMSLE 0.140. *Change:* format×dow and format×promo historical means. *Eval:* per-format RMSLE.
6. **Worst-hub inspection** — *Hypothesis:* hub 971 RMSLE 1.42 (huge). *Change:* inspect for regime shift/refurbishment; add gap-aware "days since last open" + recency-weighted level. *Eval:* per-hub RMSLE tail.
7. **Recent-momentum via careful short lags** — *Hypothesis:* only lags≥42 used; nearer history is informative. *Change:* recursive/iterative forecasting to unlock lag-1..7 safely. *Eval:* full-fold RMSLE vs direct model. (Higher effort/risk.)
8. **Promo lead/lag** — *Hypothesis:* promo lifts demand and may have pre/post effects. *Change:* promo run-length and next/prev-day promo flags (all known from the test calendar). *Eval:* promo-segment RMSLE.
9. **Drop redundant AppSessions aggregates** — *Hypothesis:* they add noise. *Change:* remove them; compare CV. *Eval:* mean-fold RMSLE.
10. **Hyperparameter tuning of LightGBM** (in progress this phase).
