"""
baseline.py — naive forecasting baselines (Phase 4).

These are deliberately simple, leakage-safe reference points. Every later model
must beat the best of these to justify its complexity.

All statistics are fit on the fold's TRAIN slice only (never the validation
future), using OPEN rows only (IsOpen==1), because closed rows are a
deterministic zero that we hard-set at predict time. Predictions for closed
validation rows are forced to 0.

Strategies
----------
* global_median      : one number for every open row.
* hub_median         : median per HubID.
* hub_weekday_median : median per (HubID, Weekday)  <- usually the strongest naive.
* hub_recent_mean    : mean of each hub's last N open days before the fold.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data import load_train
from src.validation import holdout_split, rolling_splits, evaluate, Fold

TARGET = "OrderVolume"


def _fit_predict(train: pd.DataFrame, val: pd.DataFrame, strategy: str,
                 recent_days: int = 28) -> np.ndarray:
    """Return predictions for every val row. Closed rows -> 0."""
    open_tr = train[train.IsOpen == 1]
    global_med = float(open_tr[TARGET].median())

    if strategy == "global_median":
        pred = np.full(len(val), global_med)

    elif strategy == "hub_median":
        hub_med = open_tr.groupby("HubID")[TARGET].median()
        pred = val["HubID"].map(hub_med).fillna(global_med).to_numpy()

    elif strategy == "hub_weekday_median":
        hw = open_tr.groupby(["HubID", "Weekday"])[TARGET].median()
        hub_med = open_tr.groupby("HubID")[TARGET].median()
        keys = list(zip(val["HubID"], val["Weekday"]))
        pred = pd.Series(hw.reindex(keys).to_numpy(), index=val.index)
        # fallbacks: hub median, then global median
        pred = pred.fillna(val["HubID"].map(hub_med)).fillna(global_med).to_numpy()

    elif strategy == "hub_recent_mean":
        cutoff = train.Date.max() - pd.Timedelta(days=recent_days)
        recent = open_tr[open_tr.Date > cutoff]
        hub_recent = recent.groupby("HubID")[TARGET].mean()
        hub_med = open_tr.groupby("HubID")[TARGET].median()
        pred = (val["HubID"].map(hub_recent)
                .fillna(val["HubID"].map(hub_med))
                .fillna(global_med).to_numpy())
    else:
        raise ValueError(strategy)

    # hard-set closed rows to 0 (deterministic)
    pred = np.where(val["IsOpen"].to_numpy() == 1, pred, 0.0)
    return np.clip(pred, 0, None)


def run(df: pd.DataFrame, fold: Fold, strategies: list[str]) -> list[dict]:
    train = df.iloc[fold.train_idx]
    val = df.iloc[fold.val_idx]
    rows = []
    for s in strategies:
        pred = _fit_predict(train, val, s)
        res = evaluate(df, fold, pred)
        res["strategy"] = s
        rows.append(res)
    return rows


def main():
    df = load_train()
    strategies = ["global_median", "hub_median", "hub_weekday_median", "hub_recent_mean"]

    all_rows = []
    print("=" * 70, "\nMAIN HOLDOUT\n", "=" * 70, sep="")
    main_fold = holdout_split(df)
    print(main_fold.describe(), "\n")
    for r in run(df, main_fold, strategies):
        print(f"  {r['strategy']:20s} RMSLE(all)={r['rmsle_all']:.5f} "
              f"| open={r['rmsle_open']:.5f} | closed={r['rmsle_closed']}")
        all_rows.append(r)

    print("\n" + "=" * 70, "\nROLLING FOLDS (stability)\n", "=" * 70, sep="")
    for f in rolling_splits(df, n_folds=3):
        print(f.describe())
        for r in run(df, f, strategies):
            print(f"  {r['strategy']:20s} RMSLE(all)={r['rmsle_all']:.5f}")
            all_rows.append(r)
        print()

    out = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out / "baseline_metrics.csv", index=False)
    print("SAVED outputs/metrics/baseline_metrics.csv")


if __name__ == "__main__":
    main()
