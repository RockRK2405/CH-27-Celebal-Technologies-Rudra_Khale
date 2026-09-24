"""
error_analysis.py — Phase 7: feature importance + segmented error analysis.

Reads the main-holdout models saved by train.py, predicts the holdout, and
breaks RMSLE down by business/structural segments so we can propose *targeted*
next experiments (never random feature additions).

Outputs
  outputs/error_analysis.md
  outputs/error_analysis/*.png  (diagnostic plots)
  outputs/error_analysis/segment_rmsle_*.csv
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_train, load_hub_metadata
from src.validation import rolling_splits, rmsle
from src.features import as_model_frame

OUT = ROOT / "outputs" / "error_analysis"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "outputs" / "cache"
METRICS = ROOT / "outputs" / "metrics"


def _seg_rmsle(dfd: pd.DataFrame, by) -> pd.DataFrame:
    g = dfd.groupby(by)
    rows = []
    for key, idx in g.groups.items():
        sub = dfd.loc[idx]
        rows.append({"segment": key, "n": len(sub),
                     "rmsle": rmsle(sub["y"], sub["pred"]),
                     "mean_actual": sub["y"].mean()})
    return pd.DataFrame(rows).sort_values("rmsle", ascending=False)


def pick_best_model():
    bm = pd.read_csv(ROOT / "outputs" / "model_benchmark.csv")
    main = bm[(bm.fold == "rolling_fold3") & (bm.target_mode == "log")]
    best = main.sort_values("rmsle_all").iloc[0]["model"]
    return best, bm


def main():
    df = load_train(); hub = load_hub_metadata()
    fold = rolling_splits(df, n_folds=3)[-1]
    val = df.iloc[fold.val_idx].reset_index(drop=True)
    Xval = pd.read_parquet(CACHE / f"{fold.name}_Xval.parquet")
    meta = np.load(CACHE / f"{fold.name}_valmeta.npy", allow_pickle=True).item()
    y, iso = meta["y"], meta["isopen"]

    best_name, bm = pick_best_model()
    model = joblib.load(ROOT / "models" / f"{best_name}_log_holdout.joblib")
    pred = np.clip(np.expm1(model.predict(as_model_frame(Xval))), 0, None)
    pred = np.where(iso == 1, pred, 0.0)

    # diagnostics frame (open rows carry the signal; closed are exact zeros)
    hub_mean = df[df.IsOpen == 1].groupby("HubID")["OrderVolume"].mean()
    hub_start = df.groupby("HubID")["Date"].min()
    d = pd.DataFrame({
        "HubID": val.HubID, "y": y, "pred": pred, "IsOpen": iso,
        "dow": val.Date.dt.dayofweek, "month": val.Date.dt.month,
        "RegionalHoliday": val.RegionalHoliday, "PromoActive": val.PromoActive,
        "SchoolClosureFlag": val.SchoolClosureFlag,
    })
    d = d.merge(hub[["HubID", "HubFormat", "AssortmentTier"]], on="HubID", how="left")
    d["hub_mean"] = d.HubID.map(hub_mean)
    d["vol_tier"] = pd.qcut(d.hub_mean, 3, labels=["low", "mid", "high"])
    tenure_days = (df.Date.max() - d.HubID.map(hub_start)).dt.days
    d["maturity"] = np.where(tenure_days < 365, "new(<1y)", "mature(>=1y)")
    d_open = d[d.IsOpen == 1].copy()

    overall = rmsle(d.y, d.pred)
    lines = ["# Phase 7 — Error Analysis\n",
             f"**Best model:** `{best_name}` (log1p target) | **holdout RMSLE (all)** = "
             f"`{overall:.5f}` | open-only = `{rmsle(d_open.y, d_open.pred):.5f}`\n"]

    # ---- segmented RMSLE ----
    lines.append("## RMSLE by segment (open rows)\n")
    for by, title in [("dow", "day of week"), ("month", "month"),
                      ("PromoActive", "promotion"), ("RegionalHoliday", "holiday"),
                      ("SchoolClosureFlag", "school closure"),
                      ("HubFormat", "hub format"), ("AssortmentTier", "assortment tier"),
                      ("vol_tier", "hub volume tier"), ("maturity", "hub maturity")]:
        seg = _seg_rmsle(d_open, by)
        seg.to_csv(OUT / f"segment_rmsle_{by}.csv", index=False)
        lines.append(f"### by {title}\n")
        lines.append(seg.to_markdown(index=False, floatfmt=".4f") + "\n")

    # zero/near-zero vs high demand
    nz = d_open[d_open.y <= d_open.y.quantile(0.05)]
    hi = d_open[d_open.y >= d_open.y.quantile(0.95)]
    lines.append("## Demand-level slices (open rows)\n")
    lines.append(f"* near-zero (<=5th pct, y<= {d_open.y.quantile(0.05):.0f}): "
                 f"n={len(nz)}, RMSLE={rmsle(nz.y, nz.pred):.4f}")
    lines.append(f"* high demand (>=95th pct, y>= {d_open.y.quantile(0.95):.0f}): "
                 f"n={len(hi)}, RMSLE={rmsle(hi.y, hi.pred):.4f}\n")

    # per-hub RMSLE distribution -> worst hubs
    per_hub = _seg_rmsle(d_open, "HubID").rename(columns={"segment": "HubID"})
    per_hub.to_csv(OUT / "per_hub_rmsle.csv", index=False)
    lines.append("## Worst 15 hubs by RMSLE\n")
    lines.append(per_hub.head(15).to_markdown(index=False, floatfmt=".4f") + "\n")

    # ---- feature importance comparison ----
    lines.append("## Feature importance (top 20, best model)\n")
    fi = pd.read_csv(METRICS / f"feature_importance_{best_name}.csv",
                     index_col=0).iloc[:, 0]
    lines.append(fi.head(20).to_frame("importance").to_markdown(floatfmt=".1f") + "\n")

    # ---- plots ----
    plt.figure(figsize=(7, 5))
    plt.scatter(d_open.y, d_open.pred, s=2, alpha=0.2)
    lim = [0, d_open.y.max()]; plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel("actual"); plt.ylabel("predicted"); plt.title(f"{best_name}: pred vs actual (open)")
    plt.tight_layout(); plt.savefig(OUT / "pred_vs_actual.png", dpi=110); plt.close()

    resid = np.log1p(d_open.pred) - np.log1p(d_open.y)
    plt.figure(figsize=(7, 4)); plt.hist(resid, bins=80)
    plt.xlabel("log1p(pred) - log1p(actual)"); plt.title("residuals (log space)")
    plt.tight_layout(); plt.savefig(OUT / "residual_hist.png", dpi=110); plt.close()

    plt.figure(figsize=(7, 4)); per_hub.rmsle.hist(bins=60)
    plt.xlabel("per-hub RMSLE"); plt.title("distribution of per-hub RMSLE")
    plt.tight_layout(); plt.savefig(OUT / "per_hub_rmsle_hist.png", dpi=110); plt.close()

    fi.head(20)[::-1].plot(kind="barh", figsize=(7, 7))
    plt.title(f"{best_name} top-20 importance"); plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=110); plt.close()

    (ROOT / "outputs" / "error_analysis.md").write_text("\n".join(lines))
    print(f"best={best_name} overall RMSLE={overall:.5f}; wrote outputs/error_analysis.md")


if __name__ == "__main__":
    main()
