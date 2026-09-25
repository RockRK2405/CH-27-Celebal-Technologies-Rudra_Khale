"""EXP-6: horizon-aware model (near-term lags + horizon feature, no recursion).
Evaluate on the summer fold and main holdout vs the direct baselines
(summer 0.0717 / holdout 0.0642). Also break RMSLE down by horizon bucket.
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_train, load_hub_metadata
from src.validation import window_fold, rolling_splits, rmsle
from src.features import make_supervised, build_features
from src.models import fit_predict

BASE = {"summer2014": 0.07166, "rolling_fold3": 0.06422}


def main():
    params = json.loads((ROOT / "outputs" / "hyperparameter_results" /
                         "best_params.json").read_text())["params"]
    df = load_train(); hub = load_hub_metadata()
    for fold in [window_fold(df, "2014-06-20", "2014-07-31", "summer2014"),
                 rolling_splits(df, n_folds=3)[-1]]:
        Xtr, ytr = make_supervised(df, hub, fold.train_end)
        val = df.iloc[fold.val_idx]
        Xval = build_features(df.iloc[fold.train_idx], val, hub, fold.train_end)
        yv, iso = val["OrderVolume"].to_numpy(), val["IsOpen"].to_numpy()
        r = fit_predict("lgbm", Xtr, ytr, Xval, "log", params)
        pred = np.where(iso == 1, np.clip(r["pred"], 0, None), 0.0)
        score = rmsle(yv, pred)
        print(f"\n=== {fold.name} ===  horizon-aware RMSLE = {score:.5f}  "
              f"(direct baseline {BASE[fold.name]:.5f}, delta {score-BASE[fold.name]:+.5f})",
              flush=True)
        # by horizon bucket (open rows)
        hz = Xval["horizon"].to_numpy(); m = iso == 1
        for lo, hi in [(1, 7), (8, 14), (15, 28), (29, 42)]:
            b = m & (hz >= lo) & (hz <= hi)
            if b.any():
                print(f"    h {lo:2d}-{hi:2d}: RMSLE={rmsle(yv[b], pred[b]):.5f}  n={int(b.sum())}")


if __name__ == "__main__":
    main()
