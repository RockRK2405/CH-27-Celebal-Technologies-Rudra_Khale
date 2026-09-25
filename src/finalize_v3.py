"""Final model v3: seed-averaged, retuned, horizon-aware LightGBM -> submission_v3.

Seed averaging (bagging over random seeds) reduces variance for a small, reliable
gain. Trains K LightGBMs on ALL training data with the summer-tuned params and
averages their test predictions in expm1 space. Closed hubs forced to 0.
"""
from __future__ import annotations
import sys, pathlib, json, datetime
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import (load_train, load_test, load_hub_metadata, load_sample_submission)
from src.features import make_supervised, build_features, as_model_frame

SEEDS = [42, 7, 123, 2024, 99]
SUB = ROOT / "submissions"


def train_predict_all(df, test, hub, params):
    import lightgbm as lgb
    cutoff = df["Date"].max()
    Xtr, ytr = make_supervised(df, hub, cutoff)
    Xte = build_features(df, test, hub, cutoff)
    Xtr_m, Xte_m = as_model_frame(Xtr), as_model_frame(Xte)
    y = np.log1p(ytr)
    preds = np.zeros(len(Xte_m))
    for s in SEEDS:
        m = lgb.LGBMRegressor(objective="regression", n_jobs=-1, verbose=-1,
                              random_state=s, **params)
        m.fit(Xtr_m, y)
        preds += np.clip(np.expm1(m.predict(Xte_m)), 0, None)
    return preds / len(SEEDS)


def main():
    params = json.loads((ROOT / "outputs" / "hyperparameter_results" /
                         "best_params.json").read_text())["params"]
    df = load_train(); test = load_test(); hub = load_hub_metadata()
    ss = load_sample_submission()

    final = train_predict_all(df, test, hub, params)
    final = np.where(test["IsOpen"].to_numpy() == 1, final, 0.0)
    final = np.clip(final, 0, None)

    out = pd.DataFrame({"Id": test["Id"].to_numpy(), "OrderVolume": final})
    out = out.set_index("Id").reindex(ss["Id"]).reset_index()
    checks = {
        "rows_match_sample": len(out) == len(ss),
        "ids_match_sample_order": bool((out["Id"].to_numpy() == ss["Id"].to_numpy()).all()),
        "no_missing": int(out["OrderVolume"].isna().sum()) == 0,
        "no_negative": bool((out["OrderVolume"] >= 0).all()),
        "columns_ok": list(out.columns) == list(ss.columns),
    }
    assert all(checks.values()), f"checks failed: {checks}"
    out.to_csv(SUB / "submission_v3.csv", index=False)
    print("checks:", checks)
    print("zeros:", int((out.OrderVolume == 0).sum()), "| mean:", round(out.OrderVolume.mean(), 1))
    print("wrote submissions/submission_v3.csv")


if __name__ == "__main__":
    main()
