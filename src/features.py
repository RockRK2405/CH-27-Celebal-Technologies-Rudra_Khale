"""
features.py — leakage-safe feature engineering (Phase 5).

Forecasting setup that drives every design choice
-------------------------------------------------
The test set is a *block* of 42 future days that all depend on the same fixed
information cutoff (the last training day, 2015-06-19). We therefore build a
**direct multi-step** model: one model predicts any day in the 42-day block from
features known at the cutoff.

The single safety rule:
    A feature for a target row at date t may only use data with Date <= CUTOFF,
    where CUTOFF is the last day we are allowed to observe.
        * test:       CUTOFF = 2015-06-19 (train max)
        * validation: CUTOFF = val_start - 1 day
Because the block spans up to 42 days after CUTOFF, any lag of the *target* must
be >= 42 days (otherwise the last block days would need an unobserved value).
This makes validation and test have IDENTICAL feature availability — the whole
point of a faithful CV.

`build_features(history, targets, hub, cutoff)` is the one entry point:
    * `history` = rows with Date <= cutoff  (source of all target-derived stats)
    * `targets` = rows to featurize (val or test), all with Date > cutoff
    * every target-derived statistic is computed from `history` ONLY.

AppSessions
-----------
`AppSessions` is train-only (absent from test), so its same-day value is BANNED.
We use it *only* as a historical per-hub average (known at the cutoff), never the
contemporaneous value. See outputs/feature_risk_analysis.md.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGET = "OrderVolume"

# Target-lag horizons: all >= 42 (the block length) so every block day is safe.
SAFE_LAGS = [42, 49, 56, 63, 364, 371]

# Near-term lags for the HORIZON-AWARE model. These are computed ONLY from
# history <= cutoff, so for a target at date t=cutoff+h, lag_k is naturally
# available iff k >= h (t-k <= cutoff) and NaN otherwise. We deliberately DO NOT
# fill these NaNs — "unavailable" is a real signal, and LightGBM handles NaN. A
# `horizon` feature lets the model learn when each near lag is trustworthy. This
# is the leakage-safe, drift-free alternative to recursive forecasting.
NEAR_LAGS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35]

MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}

CATEGORICAL = ["HubID", "HubFormat", "AssortmentTier", "dow", "month"]


# ---------------------------------------------------------------------------
# calendar (pure function of the forecast date — always knowable)
# ---------------------------------------------------------------------------
def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["Date"]
    out = pd.DataFrame(index=df.index)
    out["dow"] = d.dt.dayofweek                    # 0=Mon
    out["day"] = d.dt.day
    out["week"] = d.dt.isocalendar().week.astype("int32")
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter
    out["year"] = d.dt.year
    out["dayofyear"] = d.dt.dayofyear
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["is_month_start"] = d.dt.is_month_start.astype("int8")
    out["is_month_end"] = d.dt.is_month_end.astype("int8")
    # cyclic encodings
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365)
    return out


# ---------------------------------------------------------------------------
# metadata (static + time-derived that is still calendar-knowable)
# ---------------------------------------------------------------------------
def _metadata(df: pd.DataFrame, hub: pd.DataFrame) -> pd.DataFrame:
    # merge resets the index, so work positionally with numpy arrays and only
    # re-attach df.index at the end (avoids silent index-alignment NaNs).
    m = df.merge(hub, on="HubID", how="left").reset_index(drop=True)
    out = pd.DataFrame(index=range(len(df)))
    out["HubFormat"] = m["HubFormat"].to_numpy()
    out["AssortmentTier"] = m["AssortmentTier"].to_numpy()
    out["LoyaltyProgram"] = m["LoyaltyProgram"].to_numpy()

    year = df["Date"].dt.year.to_numpy()
    month = df["Date"].dt.month.to_numpy()
    week = df["Date"].dt.isocalendar().week.to_numpy().astype("float64")

    # competitor distance: heavy right skew -> log; 3 NaN -> flag + median fill
    dist = m["CompetitorDistance"]
    out["competitor_distance_missing"] = dist.isna().to_numpy().astype("int8")
    out["log_competitor_distance"] = np.log1p(dist.fillna(dist.median()).to_numpy())

    # competitor tenure in months at the forecast date (calendar + static anchor)
    cy, cm = m["CompetitorOpenSinceYear"].to_numpy(), m["CompetitorOpenSinceMonth"].to_numpy()
    tenure = (year - cy) * 12 + (month - cm)
    out["competitor_open_months"] = np.where(np.isnan(tenure), -1.0, np.clip(tenure, 0, None))
    out["competitor_unknown"] = np.isnan(cy).astype("int8")

    # loyalty program tenure/active at the forecast date
    ly, lw = m["LoyaltyProgramSinceYear"].to_numpy(), m["LoyaltyProgramSinceWeek"].to_numpy()
    lp = m["LoyaltyProgram"].to_numpy()
    tenure_w = (year - ly) * 52 + (week - lw)
    active = (lp == 1) & (tenure_w > 0)
    active = np.where(np.isnan(tenure_w), False, active)
    out["loyalty_active"] = active.astype("int8")
    out["loyalty_tenure_weeks"] = np.where(active, np.clip(np.nan_to_num(tenure_w), 0, None), 0.0)

    # is the forecast month a loyalty-renewal month for this hub?
    month_abbr = pd.Series(month).map(MONTH_ABBR).to_numpy()
    interval = m["LoyaltyProgramInterval"].fillna("").to_numpy()
    out["is_loyalty_renewal_month"] = np.array(
        [int(bool(ma) and (ma in iv)) for ma, iv in zip(month_abbr, interval)], dtype="int8")

    out.index = df.index
    return out


# ---------------------------------------------------------------------------
# target-history aggregates (computed from history <= cutoff ONLY)
# ---------------------------------------------------------------------------
def _history_aggregates(history: pd.DataFrame, targets: pd.DataFrame,
                        cutoff: pd.Timestamp) -> pd.DataFrame:
    open_h = history[history["IsOpen"] == 1]
    global_med = float(open_h[TARGET].median())
    out = pd.DataFrame(index=targets.index)

    # ---- per-hub level (open rows only) ----
    g = open_h.groupby("HubID")[TARGET]
    hub_stats = pd.DataFrame({
        "hub_mean": g.mean(), "hub_median": g.median(),
        "hub_std": g.std(), "hub_max": g.max(),
    })
    for c in hub_stats:
        out[c] = targets["HubID"].map(hub_stats[c]).astype("float64")

    # ---- per (hub, weekday) — the strongest naive signal ----
    open_h = open_h.assign(_dow=open_h["Date"].dt.dayofweek)
    hw = open_h.groupby(["HubID", "_dow"])[TARGET].agg(["mean", "median"])
    tgt_dow = targets["Date"].dt.dayofweek
    keys = list(zip(targets["HubID"], tgt_dow))
    out["hub_wd_mean"] = pd.Series(hw["mean"].reindex(keys).to_numpy(), index=targets.index)
    out["hub_wd_median"] = pd.Series(hw["median"].reindex(keys).to_numpy(), index=targets.index)

    # ---- per (hub, promo) — encodes each hub's promo lift ----
    hp = open_h.groupby(["HubID", "PromoActive"])[TARGET].mean()
    keys_p = list(zip(targets["HubID"], targets["PromoActive"]))
    out["hub_promo_mean"] = pd.Series(hp.reindex(keys_p).to_numpy(), index=targets.index)
    # NOTE: per-(hub, school-closure) means were tried (EXP-2) and REJECTED — they
    # hurt the summer fold by +0.012 (see outputs/leaderboard_analysis.md).

    # ---- recent level & trend (windows ending at cutoff) ----
    for win in (28, 91, 182):
        recent = open_h[open_h["Date"] > (cutoff - pd.Timedelta(days=win))]
        rm = recent.groupby("HubID")[TARGET].mean()
        out[f"hub_recent{win}_mean"] = targets["HubID"].map(rm).astype("float64")
    out["hub_trend_ratio"] = out["hub_recent91_mean"] / out["hub_mean"]

    # ---- last observed value at/before cutoff (per hub, open) ----
    last = (open_h.sort_values("Date").groupby("HubID")[TARGET].last())
    out["hub_last_value"] = targets["HubID"].map(last).astype("float64")

    # ---- historical AppSessions (per hub / per hub-weekday) — SAFE aggregate ----
    if "AppSessions" in history.columns:
        oa = history[history["IsOpen"] == 1]
        out["hub_appsessions_mean"] = targets["HubID"].map(
            oa.groupby("HubID")["AppSessions"].mean()).astype("float64")
        oa = oa.assign(_dow=oa["Date"].dt.dayofweek)
        hwa = oa.groupby(["HubID", "_dow"])["AppSessions"].mean()
        out["hub_wd_appsessions_mean"] = pd.Series(
            hwa.reindex(keys).to_numpy(), index=targets.index)

    # ---- target lags via date-shift merge (>=42 days => block-safe) ----
    hist_small = history[["HubID", "Date", TARGET]]
    for k in SAFE_LAGS:
        shifted = hist_small.copy()
        shifted["Date"] = shifted["Date"] + pd.Timedelta(days=k)
        shifted = shifted.rename(columns={TARGET: f"lag_{k}"})
        merged = targets[["HubID", "Date"]].merge(shifted, on=["HubID", "Date"], how="left")
        out[f"lag_{k}"] = merged[f"lag_{k}"].to_numpy()
    # a short "rolling" of the four ~6-13 week-old lags (recent-ish, still safe)
    out["lag_42_63_mean"] = out[["lag_42", "lag_49", "lag_56", "lag_63"]].mean(axis=1)

    # fill remaining NaNs (hubs with sparse history) with hub or global median
    hub_med_map = targets["HubID"].map(hub_stats["hub_median"])
    for c in out.columns:
        out[c] = out[c].fillna(hub_med_map).fillna(global_med)
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _sc_runlength(history: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    """Consecutive school-closure days ending at each target date (0 if the day
    is not a closure). Uses ONLY SchoolClosureFlag (a known calendar signal from
    history + the test file), never OrderVolume — so it is leakage-safe."""
    cols = ["HubID", "Date", "SchoolClosureFlag"]
    flags = pd.concat([history[cols], targets[cols]], ignore_index=True)
    flags = flags.drop_duplicates(["HubID", "Date"]).sort_values(["HubID", "Date"])
    changed = flags["SchoolClosureFlag"] != flags.groupby("HubID")["SchoolClosureFlag"].shift()
    seg = changed.cumsum()
    run = flags.groupby(seg).cumcount() + 1
    flags["run"] = (run * flags["SchoolClosureFlag"]).astype("int32")
    lut = flags.set_index(["HubID", "Date"])["run"]
    keys = list(zip(targets["HubID"], targets["Date"]))
    return lut.reindex(keys).fillna(0).to_numpy()


def build_features(history: pd.DataFrame, targets: pd.DataFrame,
                   hub: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Assemble all leakage-safe features for `targets`, using only `history`
    (Date <= cutoff). Returns a feature frame aligned to `targets.index`.
    """
    assert (targets["Date"] > cutoff).all(), "targets must be strictly after cutoff"
    assert (history["Date"] <= cutoff).all(), "history must be at/before cutoff"

    cal = _calendar(targets)
    meta = _metadata(targets, hub)
    hist = _history_aggregates(history, targets, cutoff)

    feats = pd.concat([cal, meta, hist], axis=1)
    # raw known-at-prediction operational flags
    for c in ["PromoActive", "SchoolClosureFlag", "RegionalHoliday"]:
        feats[c] = targets[c].to_numpy()
    feats["HubID"] = targets["HubID"].to_numpy()

    # a couple of justified interactions
    feats["promo_x_weekend"] = feats["PromoActive"] * feats["is_weekend"]
    feats["promo_x_dow"] = feats["PromoActive"] * (feats["dow"] + 1)
    # NOTE: sc_run_length / sc_x_weekend were tried (EXP-2) and REJECTED (see above).

    # ---- HORIZON-AWARE near-term lags (leakage-safe, NOT filled) ----
    feats["horizon"] = (targets["Date"] - cutoff).dt.days.to_numpy()
    hist_small = history[["HubID", "Date", TARGET]]
    near = {}
    for k in NEAR_LAGS:
        s = hist_small.copy(); s["Date"] = s["Date"] + pd.Timedelta(days=k)
        s = s.rename(columns={TARGET: f"nlag_{k}"})
        m = targets[["HubID", "Date"]].merge(s, on=["HubID", "Date"], how="left")
        col = m[f"nlag_{k}"].to_numpy()          # NaN where t-k > cutoff (unavailable)
        feats[f"nlag_{k}"] = col
        near[k] = col
    # recent 7-day mean from available near lags (NaN only if none available)
    week = np.vstack([near[k] for k in (1, 2, 3, 4, 5, 6, 7)]).T
    feats["nlag_roll7"] = np.nanmean(week, axis=1)

    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns]


def make_supervised(df: pd.DataFrame, hub: pd.DataFrame, cutoff: pd.Timestamp,
                    horizon: int = 42, step: int = 42,
                    min_history_days: int = 182, open_only: bool = True
                    ) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a training matrix (X, y) for a direct `horizon`-day model.

    We slide a cutoff `c` backward from `cutoff` in `step`-day increments. For
    each `c`, features come from history (Date <= c) and targets are the next
    `horizon` days (c < Date <= c+horizon). This reuses `build_features`
    verbatim, so training examples are produced by the exact same code path as
    inference — leakage cannot sneak in, and train/serve are symmetric.

    * `open_only=True` keeps only IsOpen==1 target rows (closed rows are the
      deterministic zero we hard-set at predict time, so they carry no signal).
    * `min_history_days` skips the earliest cutoffs that have too little history.
    """
    start = df["Date"].min()
    earliest_cutoff = start + pd.Timedelta(days=min_history_days)
    Xs, ys = [], []
    c = pd.Timestamp(cutoff)
    while c >= earliest_cutoff:
        hist = df[df["Date"] <= c]
        blk = df[(df["Date"] > c) & (df["Date"] <= c + pd.Timedelta(days=horizon))]
        if open_only:
            blk = blk[blk["IsOpen"] == 1]
        if len(blk) and len(hist):
            X = build_features(hist, blk, hub, c)
            Xs.append(X)
            ys.append(blk[TARGET].to_numpy())
        c = c - pd.Timedelta(days=step)
    X = pd.concat(Xs, ignore_index=True)
    y = np.concatenate(ys)
    return X, y


def as_model_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to pandas 'category' dtype for the boosters."""
    X = X.copy()
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


if __name__ == "__main__":
    import sys, pathlib, time
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from src.data import load_train, load_hub_metadata
    from src.validation import holdout_split

    tr = load_train(); hub = load_hub_metadata()
    fold = holdout_split(tr)
    cutoff = fold.train_end
    history = tr.iloc[fold.train_idx]
    targets = tr.iloc[fold.val_idx]
    t0 = time.time()
    X = build_features(history, targets, hub, cutoff)
    print(f"built {X.shape} features in {time.time()-t0:.1f}s")
    print("columns:", list(X.columns))
    print("NaNs:", int(X.isna().sum().sum()))
    print(X.memory_usage(deep=True).sum() / 1e6, "MB")
