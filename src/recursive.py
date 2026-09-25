"""
recursive.py — EXP-5: recursive (iterative) forecasting to unlock near-term lags.

The direct model (features.py) could only use lags >= 42 days, discarding the
strongest signal in demand forecasting: *recent* demand. Here we forecast the
42-day block one day at a time. When we predict day t we already hold predictions
for days cutoff+1 .. t-1, so lag-1/7/14 are available (as pseudo-history).

Leakage safety
--------------
* At TRAINING time, near-term lags come from ACTUAL past demand (Date < row date),
  which is legitimately knowable one step ahead.
* At INFERENCE time, the "panel" of history is seeded with actual demand up to the
  cutoff and then extended ONLY with our own predictions — never with actual
  future targets. `forecast_recursive` asserts this.
The same `compute_features` builds both train and inference matrices, so they are
symmetric.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import _calendar, _metadata, as_model_frame, CATEGORICAL
from src.models import SEED

TARGET = "OrderVolume"
DAY = pd.Timedelta(days=1)
LAGS = [1, 7, 14, 21, 28, 364]
ROLLS = [7, 14, 28]
OPS = ["PromoActive", "SchoolClosureFlag", "RegionalHoliday"]


def _panel_rollings(panel: pd.DataFrame) -> pd.DataFrame:
    """Rolling means/std up to & INCLUDING each panel date (per hub)."""
    p = panel.sort_values(["HubID", "Date"])
    g = p.groupby("HubID")[TARGET]
    out = p[["HubID", "Date"]].copy()
    for n in ROLLS:
        out[f"r{n}_mean"] = g.rolling(n, min_periods=1).mean().reset_index(level=0, drop=True).values
    out["r7_std"] = g.rolling(7, min_periods=2).std().reset_index(level=0, drop=True).values
    return out


def compute_features(panel: pd.DataFrame, rows: pd.DataFrame, hub: pd.DataFrame,
                     global_med: float) -> pd.DataFrame:
    """Features for `rows` using only `panel` (history strictly before each row)."""
    feats = pd.concat([_calendar(rows), _metadata(rows, hub)], axis=1)
    for c in OPS:
        feats[c] = rows[c].to_numpy()
    feats["HubID"] = rows["HubID"].to_numpy()

    # near-term + seasonal target lags (date-shift merge => only past values)
    small = panel[["HubID", "Date", TARGET]]
    for k in LAGS:
        s = small.copy(); s["Date"] = s["Date"] + pd.Timedelta(days=k)
        s = s.rename(columns={TARGET: f"lag_{k}"})
        m = rows[["HubID", "Date"]].merge(s, on=["HubID", "Date"], how="left")
        feats[f"lag_{k}"] = m[f"lag_{k}"].to_numpy()

    # rolling stats ending at t-1  (look up panel rolling at date = t-1)
    rollmap = _panel_rollings(panel)
    tmp = rows[["HubID", "Date"]].copy(); tmp["Date"] = tmp["Date"] - DAY
    mm = tmp.merge(rollmap, on=["HubID", "Date"], how="left")
    for c in ["r7_mean", "r14_mean", "r28_mean", "r7_std"]:
        feats[c] = mm[c].to_numpy()

    # fill gaps: recent mean, then global median
    for c in feats.columns:
        if feats[c].isna().any():
            feats[c] = feats[c].fillna(feats.get("r28_mean")).fillna(global_med)
    return feats


def train_recursive(df: pd.DataFrame, hub: pd.DataFrame, cutoff: pd.Timestamp,
                    params: dict, warmup_days: int = 60):
    """Train a one-step model on actual history <= cutoff (open rows only)."""
    import lightgbm as lgb
    panel = df[df["Date"] <= cutoff][["HubID", "Date", TARGET, "IsOpen"]]
    start = df["Date"].min() + pd.Timedelta(days=warmup_days)
    rows = df[(df["Date"] <= cutoff) & (df["Date"] >= start) & (df["IsOpen"] == 1)].copy()
    global_med = float(panel.loc[panel.IsOpen == 1, TARGET].median())
    X = compute_features(panel, rows, hub, global_med)
    y = np.log1p(rows[TARGET].to_numpy())
    model = lgb.LGBMRegressor(objective="regression", random_state=SEED,
                              n_jobs=-1, verbose=-1, **params)
    model.fit(as_model_frame(X), y)
    return model, global_med


def forecast_recursive(model, df: pd.DataFrame, hub: pd.DataFrame,
                       cutoff: pd.Timestamp, future_rows: pd.DataFrame,
                       global_med: float) -> np.ndarray:
    """Predict `future_rows` (all Date > cutoff) day-by-day, feeding predictions
    back. Closed rows are forced to 0 (and appended as 0 history)."""
    assert (future_rows["Date"] > cutoff).all()
    panel = df[df["Date"] <= cutoff][["HubID", "Date", TARGET]].copy()
    fr = future_rows.copy()
    preds = pd.Series(index=fr.index, dtype="float64")
    for t in sorted(fr["Date"].unique()):
        day = fr[fr["Date"] == t]
        X = compute_features(panel, day, hub, global_med)
        p = np.clip(np.expm1(model.predict(as_model_frame(X))), 0, None)
        p = np.where(day["IsOpen"].to_numpy() == 1, p, 0.0)
        preds.loc[day.index] = p
        # append predictions as pseudo-history (NEVER actual future targets)
        panel = pd.concat([panel, pd.DataFrame({
            "HubID": day["HubID"].to_numpy(), "Date": t, TARGET: p})], ignore_index=True)
    return preds.to_numpy()


if __name__ == "__main__":
    import json
    from src.data import load_train, load_hub_metadata
    from src.validation import window_fold, rolling_splits, rmsle
    params = json.loads((ROOT / "outputs" / "hyperparameter_results" /
                         "best_params.json").read_text())["params"]
    df = load_train(); hub = load_hub_metadata()
    for fold in [window_fold(df, "2014-06-20", "2014-07-31", "summer2014"),
                 rolling_splits(df, n_folds=3)[-1]]:
        model, gmed = train_recursive(df, hub, fold.train_end, params)
        val = df.iloc[fold.val_idx]
        pred = forecast_recursive(model, df, hub, fold.train_end, val, gmed)
        y, iso = val[TARGET].to_numpy(), val["IsOpen"].to_numpy()
        print(f"{fold.name}: recursive RMSLE = {rmsle(y, pred):.5f}", flush=True)
