"""
data.py — reproducible loaders for the Celebal <> VIT Vellore dark-store
demand-forecasting competition.

Design goals
------------
* Never mutate the raw CSVs on disk (engineering rule #1). We only read them.
* Provide a single, well-typed entry point (`load_all`) that every later
  phase (validation, features, models) imports, so the whole pipeline sees
  the data the same way.
* Encode the *structural* facts discovered in the Phase-1 audit as named
  constants so the rest of the codebase reads like documentation.

Key structural facts (see outputs/data_audit.md for the full report)
--------------------------------------------------------------------
* Train: 970,379 rows, 2013-01-01 .. 2015-06-19 (900 distinct days), 1115 hubs.
* Test : 46,830 rows,  2015-06-20 .. 2015-07-31 (42 distinct days), 1115 hubs.
  -> Test is a *full grid* of 1115 hubs x 42 days, strictly AFTER train
     (1-day gap). This is genuine future forecasting.
* `AppSessions` exists ONLY in train, NOT in test  -> leakage trap.
* `IsOpen == 0`  =>  `OrderVolume == 0`  (deterministic in train).
* Metadata joins 1:1 on HubID with zero misses.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------------
# Paths (resolved relative to the repo root, so the code is location-agnostic)
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

TRAIN_CSV = DATA_DIR / "orders_train.csv"
TEST_CSV = DATA_DIR / "orders_test.csv"
HUB_CSV = DATA_DIR / "hub_metadata.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"

# ----------------------------------------------------------------------------
# Structural constants from the Phase-1 audit
# ----------------------------------------------------------------------------
TARGET = "OrderVolume"
DATE = "Date"
KEY = ["HubID", "Date"]          # unique row identifier in orders_*
SUBMISSION_ID = "Id"

# Columns present in BOTH train and test -> known at prediction time (safe).
KNOWN_AT_PREDICTION = [
    "HubID", "Weekday", "Date", "IsOpen",
    "PromoActive", "RegionalHoliday", "SchoolClosureFlag",
]
# Present only in train -> unavailable at test time (LEAKAGE TRAP).
TRAIN_ONLY = ["OrderVolume", "AppSessions"]

# Static per-hub metadata columns (safe: same value for all dates).
META_STATIC = ["HubFormat", "AssortmentTier", "CompetitorDistance", "LoyaltyProgram"]
META_TIME_ANCHORS = [  # dates from which we derive time-varying-but-known features
    "CompetitorOpenSinceMonth", "CompetitorOpenSinceYear",
    "LoyaltyProgramSinceWeek", "LoyaltyProgramSinceYear", "LoyaltyProgramInterval",
]

# The competition test window.
TEST_START = pd.Timestamp("2015-06-20")
TEST_END = pd.Timestamp("2015-07-31")
HORIZON_DAYS = 42


def load_train() -> pd.DataFrame:
    """Load orders_train.csv with Date parsed. Raw file is never modified."""
    df = pd.read_csv(TRAIN_CSV, parse_dates=[DATE])
    return df.sort_values(KEY).reset_index(drop=True)


def load_test() -> pd.DataFrame:
    """Load orders_test.csv (has `Id`, no `OrderVolume`/`AppSessions`)."""
    df = pd.read_csv(TEST_CSV, parse_dates=[DATE])
    # Preserve original row order for submission (engineering rule #13) by
    # keeping Id; callers must re-sort by Id before writing a submission.
    return df


def load_hub_metadata() -> pd.DataFrame:
    """Load hub_metadata.csv (1115 rows, one per hub)."""
    return pd.read_csv(HUB_CSV)


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_SUB_CSV)


def load_all() -> dict[str, pd.DataFrame]:
    """Return every raw table in one dict for convenience."""
    return {
        "train": load_train(),
        "test": load_test(),
        "hub": load_hub_metadata(),
        "sample_submission": load_sample_submission(),
    }


def attach_metadata(orders: pd.DataFrame, hub: pd.DataFrame | None = None) -> pd.DataFrame:
    """Left-join static hub metadata onto an orders frame (train or test).

    The audit proved this join is complete (no NaNs introduced), so a `left`
    join is safe and preserves row order/count.
    """
    if hub is None:
        hub = load_hub_metadata()
    merged = orders.merge(hub, on="HubID", how="left")
    assert len(merged) == len(orders), "metadata join changed row count!"
    return merged


if __name__ == "__main__":
    d = load_all()
    for name, df in d.items():
        print(f"{name:18s} {df.shape}")
