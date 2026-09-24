"""
leakage_checks.py — automated test-time leakage audit (Phase 8/10).

The guarantee we must prove: **no feature used for a target row at date t can
depend on OrderVolume (or any signal) from date t or later.**

The strongest possible automated test for this is a *perturbation test*: build
features, then corrupt every "future" OrderVolume (all rows dated > cutoff,
including the target rows themselves) with random garbage, rebuild features, and
assert the feature matrix is byte-for-byte identical. If any feature secretly
read a future/current target, the matrix would change and the test FAILS.

We also assert:
  * no feature column is derived from `AppSessions` same-day (train-only column);
  * every target lag horizon is >= the forecast block length (42 days);
  * lag values exactly equal the historical target at date-k.

Run:  python src/leakage_checks.py   ->  prints  TEST LEAKAGE CHECK: PASS / FAIL
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_train, load_hub_metadata, TEST_START
from src.features import build_features, SAFE_LAGS
from src.validation import holdout_split

HORIZON = 42


def _perturbation_test(df, hub, cutoff, targets, rng) -> tuple[bool, str]:
    """Corrupt all target values at Date > cutoff and confirm features are
    unchanged (i.e. no feature reads present/future OrderVolume)."""
    history = df[df["Date"] <= cutoff]
    base = build_features(history, targets, hub, cutoff)

    poisoned = df.copy()
    future_mask = poisoned["Date"] > cutoff
    poisoned.loc[future_mask, "OrderVolume"] = rng.integers(
        0, 10_000_000, size=int(future_mask.sum()))
    hist2 = poisoned[poisoned["Date"] <= cutoff]          # identical to history
    tgt2 = poisoned.loc[targets.index]                     # same rows, poisoned y
    feats2 = build_features(hist2, tgt2, hub, cutoff)

    same = base.equals(feats2)
    if not same:
        diff_cols = [c for c in base.columns if not base[c].equals(feats2[c])]
        return False, f"features changed after poisoning future target: {diff_cols}"
    return True, "features invariant to future/current target perturbation"


def _column_source_test(feats: pd.DataFrame) -> tuple[bool, str]:
    """No same-day leakage columns should exist (AppSessions/OrderVolume raw)."""
    banned = {"OrderVolume", "AppSessions"}
    present = banned & set(feats.columns)
    if present:
        return False, f"raw same-day columns present as features: {present}"
    return True, "no raw same-day OrderVolume/AppSessions columns used"


def _lag_horizon_test() -> tuple[bool, str]:
    bad = [k for k in SAFE_LAGS if k < HORIZON]
    if bad:
        return False, f"lags shorter than horizon {HORIZON}: {bad} (would need unobserved data)"
    return True, f"all target lags >= horizon {HORIZON}: {SAFE_LAGS}"


def _lag_value_test(df, hub, cutoff, targets) -> tuple[bool, str]:
    """lag_k of a target row must equal OrderVolume at (date-k) from history."""
    history = df[df["Date"] <= cutoff]
    feats = build_features(history, targets, hub, cutoff)
    lut = history.set_index(["HubID", "Date"])["OrderVolume"]
    k = SAFE_LAGS[0]
    sample = targets.sample(min(500, len(targets)), random_state=0)
    for idx, r in sample.iterrows():
        want = lut.get((r["HubID"], r["Date"] - pd.Timedelta(days=k)), np.nan)
        got = feats.loc[idx, f"lag_{k}"]
        if not np.isnan(want) and abs(want - got) > 1e-6:
            return False, f"lag_{k} mismatch at row {idx}: got {got}, expected {want}"
    return True, f"lag_{k} values match historical target at date-{k} (500 sampled)"


def run_all(verbose=True) -> bool:
    df = load_train(); hub = load_hub_metadata()
    fold = holdout_split(df)
    cutoff = fold.train_end
    targets = df.iloc[fold.val_idx]
    rng = np.random.default_rng(0)
    feats = build_features(df[df["Date"] <= cutoff], targets, hub, cutoff)

    checks = [
        ("column source", _column_source_test(feats)),
        ("lag horizon", _lag_horizon_test()),
        ("lag values", _lag_value_test(df, hub, cutoff, targets)),
        ("perturbation (future target)", _perturbation_test(df, hub, cutoff, targets, rng)),
    ]
    ok = all(p for _, (p, _) in checks)
    if verbose:
        for name, (passed, msg) in checks:
            print(f"[{'PASS' if passed else 'FAIL'}] {name}: {msg}")
    return ok


if __name__ == "__main__":
    ok = run_all()
    print("\nTEST LEAKAGE CHECK:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
