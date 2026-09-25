"""Final model v4: near-lag features with CONSERVATIVE regularised params.

v3 overfit hyperparameters to the validation folds (CV 0.054 vs LB 0.072). v4 keeps
the (leakage-safe, present-in-test) near-lag features but uses strongly regularised
LightGBM params that are far less prone to overfit, so the CV/LB gap should shrink.
Also reports the genuine-extrapolation holdout RMSLE as an honest gut-check.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_train, load_test, load_hub_metadata, load_sample_submission
from src.validation import rolling_splits, rmsle
from src.features import make_supervised, build_features, as_model_frame

SEEDS = [42, 7, 123, 2024, 99]
CONSERVATIVE = dict(n_estimators=1400, learning_rate=0.03, num_leaves=63,
                    max_depth=-1, min_child_samples=200, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.7,
                    reg_lambda=5.0, reg_alpha=1.0)


def _fit_seeds(Xtr_m, y, Xte_m, params):
    import lightgbm as lgb
    preds = np.zeros(len(Xte_m))
    for s in SEEDS:
        m = lgb.LGBMRegressor(objective="regression", n_jobs=-1, verbose=-1,
                              random_state=s, **params)
        m.fit(Xtr_m, y)
        preds += np.clip(np.expm1(m.predict(Xte_m)), 0, None)
    return preds / len(SEEDS)


def main():
    df = load_train(); test = load_test(); hub = load_hub_metadata()
    ss = load_sample_submission()

    # honest gut-check on the genuine-extrapolation holdout (last 42 train days)
    fold = rolling_splits(df, n_folds=3)[-1]
    Xtr, ytr = make_supervised(df, hub, fold.train_end)
    val = df.iloc[fold.val_idx]
    Xval = build_features(df.iloc[fold.train_idx], val, hub, fold.train_end)
    yv, iso = val["OrderVolume"].to_numpy(), val["IsOpen"].to_numpy()
    p = _fit_seeds(as_model_frame(Xtr), np.log1p(ytr), as_model_frame(Xval), CONSERVATIVE)
    p = np.where(iso == 1, p, 0.0)
    print(f"holdout (extrapolation) RMSLE = {rmsle(yv, p):.5f}", flush=True)

    # final: train on ALL data -> predict test
    cutoff = df["Date"].max()
    XtrA, ytrA = make_supervised(df, hub, cutoff)
    Xte = build_features(df, test, hub, cutoff)
    final = _fit_seeds(as_model_frame(XtrA), np.log1p(ytrA), as_model_frame(Xte), CONSERVATIVE)
    final = np.where(test["IsOpen"].to_numpy() == 1, final, 0.0)
    final = np.clip(final, 0, None)

    out = pd.DataFrame({"Id": test["Id"].to_numpy(), "OrderVolume": final})
    out = out.set_index("Id").reindex(ss["Id"]).reset_index()
    checks = {
        "rows": len(out) == len(ss),
        "ids": bool((out["Id"].to_numpy() == ss["Id"].to_numpy()).all()),
        "no_nan": int(out["OrderVolume"].isna().sum()) == 0,
        "no_neg": bool((out["OrderVolume"] >= 0).all()),
        "cols": list(out.columns) == list(ss.columns),
    }
    assert all(checks.values()), checks
    out.to_csv(ROOT / "submissions" / "submission_v4.csv", index=False)
    print("checks:", checks, "| zeros:", int((out.OrderVolume == 0).sum()),
          "| mean:", round(out.OrderVolume.mean(), 1))
    print("wrote submissions/submission_v4.csv")


if __name__ == "__main__":
    main()
