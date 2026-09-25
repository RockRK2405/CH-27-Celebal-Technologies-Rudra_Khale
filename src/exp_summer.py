"""
exp_summer.py — Phase 13 EXP-1 + EXP-2.

EXP-1: evaluate on a summer fold (val = 2014-06-20 .. 2014-07-31) that mirrors the
       Kaggle test season, to confirm CV tracks the 0.07181 leaderboard.
EXP-2: measure the effect of the new school-closure features by scoring the tuned
       LightGBM with vs without them, on the summer fold AND the main holdout.

Isolation: one feature build per fold; the two variants differ only by dropping
the SC columns, so the delta is attributable to EXP-2 alone.
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

SC_COLS = ["hub_sc_mean", "hub_sc_ratio", "sc_run_length", "sc_x_weekend"]


def eval_fold(df, hub, fold, params):
    Xtr, ytr = make_supervised(df, hub, fold.train_end)
    val = df.iloc[fold.val_idx]
    Xval = build_features(df.iloc[fold.train_idx], val, hub, fold.train_end)
    yv, iso = val["OrderVolume"].to_numpy(), val["IsOpen"].to_numpy()

    def score(cols):
        r = fit_predict("lgbm", Xtr[cols], ytr, Xval[cols], "log", params)
        p = np.where(iso == 1, np.clip(r["pred"], 0, None), 0.0)
        # school-closure segment RMSLE (open rows)
        m = (iso == 1) & (val["SchoolClosureFlag"].to_numpy() == 1)
        sc = rmsle(yv[m], p[m]) if m.any() else None
        return rmsle(yv, p), sc

    all_cols = list(Xtr.columns)
    no_sc = [c for c in all_cols if c not in SC_COLS]
    with_sc, with_sc_seg = score(all_cols)
    without_sc, without_sc_seg = score(no_sc)
    return {"fold": fold.name, "n_val": len(yv),
            "without_sc": without_sc, "with_sc": with_sc,
            "delta": with_sc - without_sc,
            "sc_seg_without": without_sc_seg, "sc_seg_with": with_sc_seg}


def main():
    params = json.loads((ROOT / "outputs" / "hyperparameter_results" /
                         "best_params.json").read_text())["params"]
    df = load_train(); hub = load_hub_metadata()

    summer = window_fold(df, "2014-06-20", "2014-07-31", "summer2014")
    holdout = rolling_splits(df, n_folds=3)[-1]

    rows = []
    for fold in [summer, holdout]:
        print(f"\n=== {fold.describe()} ===", flush=True)
        r = eval_fold(df, hub, fold, params)
        rows.append(r)
        print(f"  RMSLE  without_sc={r['without_sc']:.5f}  with_sc={r['with_sc']:.5f}  "
              f"delta={r['delta']:+.5f}")
        print(f"  school-closure segment  without={r['sc_seg_without']}  with={r['sc_seg_with']}")

    out = ROOT / "outputs" / "exp_summer_results.json"
    out.write_text(json.dumps(rows, indent=2, default=float))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
