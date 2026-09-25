"""Honest eval: params tuned on SPRING folds, tested on the UNTOUCHED summer-2014
proxy (which matched the LB for v1). Compares near-lag model vs direct-only so we
know if near lags genuinely transfer to the July test BEFORE spending a submission.
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_train, load_hub_metadata
from src.validation import window_fold, rolling_splits, rmsle
from src.features import make_supervised, build_features, NEAR_LAGS
from src.models import fit_predict

NEAR_COLS = (["horizon", "nlag_roll7", "nlag_roll14", "nlag_std7", "nlag_min7", "nlag_max7"]
             + [f"nlag_{k}" for k in NEAR_LAGS])


def main():
    params = json.loads((ROOT / "outputs" / "hyperparameter_results" /
                         "best_params.json").read_text())["params"]
    df = load_train(); hub = load_hub_metadata()
    folds = [window_fold(df, "2014-06-20", "2014-07-31", "summer2014(UNTOUCHED proxy)"),
             rolling_splits(df, n_folds=3)[-1]]
    for fold in folds:
        Xtr, ytr = make_supervised(df, hub, fold.train_end)
        val = df.iloc[fold.val_idx]
        Xval = build_features(df.iloc[fold.train_idx], val, hub, fold.train_end)
        yv, iso = val["OrderVolume"].to_numpy(), val["IsOpen"].to_numpy()

        def score(cols):
            p = fit_predict("lgbm", Xtr[cols], ytr, Xval[cols], "log", params)["pred"]
            p = np.where(iso == 1, np.clip(p, 0, None), 0.0)
            return rmsle(yv, p)

        allc = list(Xtr.columns)
        noc = [c for c in allc if c not in NEAR_COLS]
        with_near = score(allc)
        without_near = score(noc)
        print(f"{fold.name}: with_near={with_near:.5f}  direct_only={without_near:.5f}  "
              f"delta={with_near-without_near:+.5f}", flush=True)


if __name__ == "__main__":
    main()
