"""
train.py — Phase 6 model benchmark (LightGBM / CatBoost / XGBoost, raw vs log1p).

Uses the chronological validation framework and the leakage-safe feature
pipeline. Records, per model/target/fold: validation RMSLE (all + open),
in-sample RMSLE (overfitting), train/predict time, fold-by-fold scores.

Plan (keeps compute reasonable, still answers both questions):
  * raw-vs-log1p comparison: all 3 models on the MAIN holdout fold.
  * fold-by-fold stability : all 3 models in log mode across 3 rolling folds.

Outputs
  outputs/model_benchmark.csv
  models/<model>_log_holdout.joblib   (main-fold models, reused downstream)
  outputs/metrics/feature_importance_<model>.csv
"""
from __future__ import annotations
import sys, pathlib, time, json
import numpy as np
import pandas as pd
import joblib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_train, load_hub_metadata
from src.validation import rolling_splits, rmsle
from src.features import make_supervised, build_features
from src.models import fit_predict, feature_importance

CACHE = ROOT / "outputs" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
METRICS = ROOT / "outputs" / "metrics"; METRICS.mkdir(parents=True, exist_ok=True)

N_EST = {"lgbm": 900, "xgb": 900, "catboost": 700}


def get_fold_matrices(df, hub, fold):
    """Build (or load from cache) train/val matrices for a fold."""
    ftr = CACHE / f"{fold.name}_Xtr.parquet"
    if ftr.exists():
        Xtr = pd.read_parquet(ftr)
        ytr = np.load(CACHE / f"{fold.name}_ytr.npy")
        Xval = pd.read_parquet(CACHE / f"{fold.name}_Xval.parquet")
        meta = np.load(CACHE / f"{fold.name}_valmeta.npy", allow_pickle=True).item()
    else:
        Xtr, ytr = make_supervised(df, hub, fold.train_end)
        val = df.iloc[fold.val_idx]
        Xval = build_features(df.iloc[fold.train_idx], val, hub, fold.train_end)
        meta = {"y": val["OrderVolume"].to_numpy(), "isopen": val["IsOpen"].to_numpy()}
        Xtr.to_parquet(ftr); np.save(CACHE / f"{fold.name}_ytr.npy", ytr)
        Xval.to_parquet(CACHE / f"{fold.name}_Xval.parquet")
        np.save(CACHE / f"{fold.name}_valmeta.npy", meta)
    return Xtr, ytr, Xval, meta["y"], meta["isopen"]


def score(name, Xtr, ytr, Xval, yv, iso, mode):
    r = fit_predict(name, Xtr, ytr, Xval, mode, {"n_estimators": N_EST[name]})
    pred = np.where(iso == 1, r["pred"], 0.0)
    pred = np.clip(pred, 0, None)
    m = iso == 1
    row = {
        "model": name, "target_mode": mode,
        "rmsle_all": rmsle(yv, pred),
        "rmsle_open": rmsle(yv[m], pred[m]),
        "insample_rmsle": rmsle(ytr, r["insample_pred"]),
        "train_time_s": round(r["train_time"], 1),
        "predict_time_s": round(r["predict_time"], 2),
        "n_train": len(ytr), "n_val": len(yv),
    }
    row["overfit_gap"] = round(row["rmsle_open"] - row["insample_rmsle"], 5)
    return row, r["model"]


def main():
    df = load_train(); hub = load_hub_metadata()
    folds = rolling_splits(df, n_folds=3)         # fold1, fold2, fold3(=holdout)
    main_fold = folds[-1]
    rows = []
    log = open(ROOT / "outputs" / "benchmark_run.log", "w")

    def emit(msg):
        print(msg); log.write(msg + "\n"); log.flush()

    for fold in folds:
        emit(f"\n=== {fold.describe()} ===")
        Xtr, ytr, Xval, yv, iso = get_fold_matrices(df, hub, fold)
        emit(f"  matrices: train {Xtr.shape}, val {Xval.shape}")
        is_main = fold.name == main_fold.name
        modes = ["log", "raw"] if is_main else ["log"]
        for name in ["lgbm", "xgb", "catboost"]:
            for mode in modes:
                t0 = time.time()
                row, model = score(name, Xtr, ytr, Xval, yv, iso, mode)
                row["fold"] = fold.name
                rows.append(row)
                emit(f"  {name:9s} {mode:3s} | RMSLE(all)={row['rmsle_all']:.5f} "
                     f"open={row['rmsle_open']:.5f} insample={row['insample_rmsle']:.5f} "
                     f"gap={row['overfit_gap']:+.5f} | {time.time()-t0:.0f}s")
                if is_main and mode == "log":
                    joblib.dump(model, MODELS / f"{name}_log_holdout.joblib")
                    fi = feature_importance(model, list(Xval.columns), name)
                    fi.to_csv(METRICS / f"feature_importance_{name}.csv")
        # write incrementally
        pd.DataFrame(rows).to_csv(ROOT / "outputs" / "model_benchmark.csv", index=False)

    emit("\nDONE. Wrote outputs/model_benchmark.csv")
    log.close()


if __name__ == "__main__":
    main()
