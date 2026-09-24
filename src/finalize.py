"""
finalize.py — Phase 8 ensemble decision using the TUNED LightGBM.

Evaluates, across the two most-recent chronological folds (fold2, fold3):
  * tuned LightGBM (log)
  * default XGBoost (log)         [benchmark's #2, decorrelated algorithm]
  * blends: equal & weighted, in prediction space and log space

Picks the simplest option with a consistent improvement and writes
outputs/ensemble_config.json (including tuned params + weights). CatBoost is
excluded: it was slower and clearly weaker in every fold.
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_train, load_hub_metadata
from src.validation import rolling_splits, rmsle
from src.models import fit_predict
from src.train import get_fold_matrices, N_EST


def main():
    tuned = json.loads((ROOT / "outputs" / "hyperparameter_results" /
                        "best_params.json").read_text())
    lgbm_params = tuned["params"]
    df = load_train(); hub = load_hub_metadata()
    folds = rolling_splits(df, n_folds=3)[-2:]      # fold2, fold3

    per_fold = {}
    for fold in folds:
        Xtr, ytr, Xval, yv, iso = get_fold_matrices(df, hub, fold)
        pl = fit_predict("lgbm", Xtr, ytr, Xval, "log", lgbm_params)["pred"]
        px = fit_predict("xgb", Xtr, ytr, Xval, "log", {"n_estimators": N_EST["xgb"]})["pred"]
        pl = np.where(iso == 1, np.clip(pl, 0, None), 0.0)
        px = np.where(iso == 1, np.clip(px, 0, None), 0.0)
        per_fold[fold.name] = {"y": yv, "lgbm": pl, "xgb": px}
        print(f"{fold.name}: tuned_lgbm={rmsle(yv,pl):.5f} xgb={rmsle(yv,px):.5f}", flush=True)

    def mean_rmsle(fn):
        return float(np.mean([rmsle(d["y"], fn(d)) for d in per_fold.values()]))

    cands = {
        "tuned_lgbm": mean_rmsle(lambda d: d["lgbm"]),
        "xgb": mean_rmsle(lambda d: d["xgb"]),
        "blend_5050_pred": mean_rmsle(lambda d: 0.5 * d["lgbm"] + 0.5 * d["xgb"]),
        "blend_6040_pred": mean_rmsle(lambda d: 0.6 * d["lgbm"] + 0.4 * d["xgb"]),
        "blend_7030_pred": mean_rmsle(lambda d: 0.7 * d["lgbm"] + 0.3 * d["xgb"]),
        "blend_5050_log": mean_rmsle(lambda d: np.expm1(0.5 * np.log1p(d["lgbm"]) + 0.5 * np.log1p(d["xgb"]))),
    }
    print("\n=== mean RMSLE (fold2+fold3) ===")
    for k, v in sorted(cands.items(), key=lambda x: x[1]):
        print(f"  {k:18s} {v:.5f}")

    best_key = min(cands, key=cands.get)
    best_single = min(cands["tuned_lgbm"], cands["xgb"])
    # choose: prefer a blend only if it beats the best single by a clear margin
    if cands[best_key] < best_single - 3e-4 and best_key.startswith("blend"):
        wl = {"blend_5050": 0.5, "blend_6040": 0.6, "blend_7030": 0.7}[best_key.rsplit("_", 1)[0]]
        space = "log" if best_key.endswith("log") else "pred"
        ensemble = [
            {"model": "lgbm", "mode": "log", "weight": wl, "params": lgbm_params},
            {"model": "xgb", "mode": "log", "weight": round(1 - wl, 2), "params": {"n_estimators": 1500}},
        ]
        rationale = f"{best_key} beat best single ({best_single:.5f})"
        val = cands[best_key]
    else:
        ensemble = [{"model": "lgbm", "mode": "log", "weight": 1.0, "params": lgbm_params}]
        space = "pred"; rationale = "tuned single LightGBM (blend gave no clear gain)"
        val = cands["tuned_lgbm"]

    cfg = {"ensemble": ensemble, "blend_space": space, "rationale": rationale,
           "val_rmsle": round(val, 5), "comparison": {k: round(v, 5) for k, v in cands.items()}}
    (ROOT / "outputs" / "ensemble_config.json").write_text(json.dumps(cfg, indent=2))
    print("\nchosen:", rationale, "| val_rmsle=", round(val, 5))


if __name__ == "__main__":
    main()
