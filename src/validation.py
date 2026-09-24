"""
validation.py — chronological validation framework + safe RMSLE.

Why this exists
---------------
The Kaggle test set is the 42 days *immediately after* the training data
(2015-06-20 .. 2015-07-31). A random train/test split would let the model peek
at the future and give a fantasy score. So every split here is **time-ordered**:
validation always sits AFTER its training slice, and its length (42 days) equals
the real forecast horizon.

The competition metric is RMSLE. We implement it defensively:
    RMSLE = sqrt( mean( (log1p(pred) - log1p(true))^2 ) )
with predictions clipped to >= 0 (rule #15) and targets validated to be
non-negative (they always are in this data, but we assert it).

Two split generators are provided (Phase-3 requirement):
    * `holdout_split`      — one main chronological holdout (the last H days).
    * `rolling_splits`     — several expanding-train / fixed-horizon folds for
                             stability checks.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

DATE = "Date"
HORIZON = 42  # matches the Kaggle test window length


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def rmsle(y_true, y_pred) -> float:
    """Safe Root Mean Squared Logarithmic Error.

    * predictions are clipped to >= 0 before scoring (rule #15);
    * targets are asserted non-negative (no fabricated/invalid labels);
    * NaNs raise, rather than silently propagating.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("NaNs present in y_true or y_pred")
    if (y_true < 0).any():
        raise ValueError("y_true contains negative values — invalid for RMSLE")
    y_pred = np.clip(y_pred, 0, None)  # never let a negative prediction through
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


# ---------------------------------------------------------------------------
# Fold containers
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    name: str
    train_idx: np.ndarray          # positional index into the frame
    val_idx: np.ndarray
    train_end: pd.Timestamp        # last training date (inclusive)
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    def describe(self) -> str:
        return (f"{self.name}: train<= {self.train_end.date()} "
                f"({len(self.train_idx):,} rows) | "
                f"val {self.val_start.date()}..{self.val_end.date()} "
                f"({len(self.val_idx):,} rows)")


def _sorted_unique_dates(df: pd.DataFrame) -> np.ndarray:
    return np.sort(df[DATE].unique())


def holdout_split(df: pd.DataFrame, horizon: int = HORIZON) -> Fold:
    """One main chronological holdout: the last `horizon` distinct days are
    validation; everything strictly before is training.
    """
    dates = _sorted_unique_dates(df)
    if len(dates) <= horizon:
        raise ValueError("not enough distinct dates for the requested horizon")
    val_dates = dates[-horizon:]
    val_start = pd.Timestamp(val_dates[0])
    val_end = pd.Timestamp(val_dates[-1])
    train_mask = df[DATE] < val_start
    val_mask = df[DATE] >= val_start
    train_idx = np.where(train_mask.to_numpy())[0]
    val_idx = np.where(val_mask.to_numpy())[0]
    train_end = pd.Timestamp(dates[dates < np.datetime64(val_start)].max())
    return Fold("holdout_last42", train_idx, val_idx, train_end, val_start, val_end)


def rolling_splits(df: pd.DataFrame, horizon: int = HORIZON,
                   n_folds: int = 3, step: int | None = None) -> list[Fold]:
    """Expanding-train, fixed-`horizon` validation folds.

    The last fold's validation window equals `holdout_split` (the last H days);
    earlier folds step back by `step` days (default = horizon, so windows are
    contiguous and non-overlapping). Training is always everything strictly
    before each fold's validation window (expanding window).
    """
    if step is None:
        step = horizon
    dates = _sorted_unique_dates(df)
    n = len(dates)
    folds: list[Fold] = []
    for k in range(n_folds):
        end_pos = n - k * step                 # exclusive upper bound
        start_pos = end_pos - horizon
        if start_pos <= 0:
            break                               # not enough history for more folds
        val_dates = dates[start_pos:end_pos]
        val_start = pd.Timestamp(val_dates[0])
        val_end = pd.Timestamp(val_dates[-1])
        train_mask = df[DATE] < val_start
        val_mask = (df[DATE] >= val_start) & (df[DATE] <= val_end)
        train_idx = np.where(train_mask.to_numpy())[0]
        val_idx = np.where(val_mask.to_numpy())[0]
        if len(train_idx) == 0:
            break
        train_end = pd.Timestamp(dates[dates < np.datetime64(val_start)].max())
        folds.append(Fold(f"rolling_fold{n_folds - k}", train_idx, val_idx,
                          train_end, val_start, val_end))
    return list(reversed(folds))  # chronological order (earliest first)


def evaluate(df: pd.DataFrame, fold: Fold, pred: np.ndarray,
             target: str = "OrderVolume",
             open_col: str = "IsOpen") -> dict:
    """Score a prediction on one fold.

    Returns overall RMSLE (all val rows), plus open-only and closed-only
    RMSLE for insight. `pred` must align with `fold.val_idx` order.
    """
    y = df[target].to_numpy()[fold.val_idx]
    isopen = df[open_col].to_numpy()[fold.val_idx]
    pred = np.clip(np.asarray(pred, dtype="float64"), 0, None)
    out = {"fold": fold.name, "rmsle_all": rmsle(y, pred),
           "n_val": len(y)}
    if isopen is not None:
        m = isopen == 1
        out["rmsle_open"] = rmsle(y[m], pred[m]) if m.any() else None
        out["rmsle_closed"] = rmsle(y[~m], pred[~m]) if (~m).any() else None
        out["n_open"] = int(m.sum())
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from src.data import load_train

    tr = load_train()
    h = holdout_split(tr)
    print("MAIN HOLDOUT:", h.describe())
    print()
    for f in rolling_splits(tr, n_folds=3):
        print("ROLLING:", f.describe())
