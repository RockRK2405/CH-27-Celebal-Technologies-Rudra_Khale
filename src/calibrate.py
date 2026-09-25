"""Legitimate RMSLE calibration: find the scalar multiplier that minimises RMSLE
on holdout + summer folds (RMSLE punishes under-prediction, so the optimum is
usually >1). A global multiplier is a low-variance correction that transfers well.
Also reports a simple v1+enriched blend for comparison.
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_train, load_hub_metadata
from src.validation import window_fold, rolling_splits, rmsle
from src.features import make_supervised, build_features
from src.models import fit_predict


def best_multiplier(y, pred, iso):
    m = iso == 1
    grid = np.arange(0.90, 1.151, 0.005)
    scores = [(rmsle(y[m], np.clip(mult * pred[m], 0, None)), mult) for mult in grid]
    scores.sort()
    return scores[0][1], scores[0][0], rmsle(y[m], np.clip(pred[m], 0, None))


def main():
    params = json.loads((ROOT / "outputs" / "hyperparameter_results" /
                         "best_params.json").read_text())["params"]
    df = load_train(); hub = load_hub_metadata()
    folds = [window_fold(df, "2014-06-20", "2014-07-31", "summer2014"),
             rolling_splits(df, n_folds=3)[-1]]
    mults = []
    for fold in folds:
        Xtr, ytr = make_supervised(df, hub, fold.train_end)
        val = df.iloc[fold.val_idx]
        Xval = build_features(df.iloc[fold.train_idx], val, hub, fold.train_end)
        yv, iso = val["OrderVolume"].to_numpy(), val["IsOpen"].to_numpy()
        pred = fit_predict("lgbm", Xtr, ytr, Xval, "log", params)["pred"]
        mult, best, base = best_multiplier(yv, pred, iso)
        mults.append(mult)
        print(f"{fold.name}: base RMSLE={base:.5f} -> mult={mult:.3f} gives {best:.5f} "
              f"(gain {base-best:+.5f})", flush=True)
    print(f"\nmean optimal multiplier: {np.mean(mults):.4f}")
    json.dump({"multiplier": float(np.mean(mults))},
              open(ROOT / "outputs" / "calibration.json", "w"), indent=2)


if __name__ == "__main__":
    main()
