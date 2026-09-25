"""EXP-7: does LightGBM + XGBoost ensemble beat the tuned single LightGBM?
Validated on the summer fold (LB proxy) + main holdout with the current feature set.
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

XGB_PARAMS = {"n_estimators": 2000, "learning_rate": 0.03, "max_depth": 8,
              "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.6,
              "reg_lambda": 1.0}


def main():
    lgbm_params = json.loads((ROOT / "outputs" / "hyperparameter_results" /
                              "best_params.json").read_text())["params"]
    df = load_train(); hub = load_hub_metadata()
    folds = [window_fold(df, "2014-06-20", "2014-07-31", "summer2014"),
             rolling_splits(df, n_folds=3)[-1]]
    store = {}
    for fold in folds:
        Xtr, ytr = make_supervised(df, hub, fold.train_end)
        val = df.iloc[fold.val_idx]
        Xval = build_features(df.iloc[fold.train_idx], val, hub, fold.train_end)
        yv, iso = val["OrderVolume"].to_numpy(), val["IsOpen"].to_numpy()
        pl = fit_predict("lgbm", Xtr, ytr, Xval, "log", lgbm_params)["pred"]
        px = fit_predict("xgb", Xtr, ytr, Xval, "log", XGB_PARAMS)["pred"]
        pl = np.where(iso == 1, np.clip(pl, 0, None), 0.0)
        px = np.where(iso == 1, np.clip(px, 0, None), 0.0)
        store[fold.name] = {"y": yv, "l": pl, "x": px}
        print(f"{fold.name}: lgbm={rmsle(yv,pl):.5f}  xgb={rmsle(yv,px):.5f}", flush=True)

    print("\n=== blends (mean over summer+holdout) ===")
    res = {}
    for wl in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        s = np.mean([rmsle(d["y"], np.clip(wl*d["l"]+(1-wl)*d["x"],0,None)) for d in store.values()])
        res[wl] = s
        print(f"  lgbm {wl:.1f} / xgb {1-wl:.1f}: {s:.5f}")
    best_wl = min(res, key=res.get)
    print(f"\nbest weight: lgbm {best_wl} (mean {res[best_wl]:.5f}); single-lgbm mean {res[1.0]:.5f}")
    json.dump({"best_lgbm_weight": best_wl, "results": {str(k):round(v,5) for k,v in res.items()}},
              open(ROOT/"outputs"/"blend_results.json","w"), indent=2)


if __name__ == "__main__":
    main()
