"""
models.py — thin, uniform wrappers around LightGBM / CatBoost / XGBoost.

Each builder returns an (unfitted) regressor configured with *sensible* defaults
(not a huge search — Phase 6 tunes later). A single `fit_predict` handles the
raw-vs-log1p target comparison and the categorical handling each library wants.

Target modes
------------
* "raw" : model fits OrderVolume directly.
* "log" : model fits log1p(OrderVolume); prediction = expm1(.), clipped to >=0.
  Because the metric is RMSLE = RMSE in log1p space, the log target aligns the
  training loss (L2) with the evaluation metric — usually a win here.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from src.features import CATEGORICAL, as_model_frame

SEED = 42


def build_lgbm(**overrides):
    import lightgbm as lgb
    params = dict(
        objective="regression", n_estimators=1500,
        learning_rate=0.03, num_leaves=63, max_depth=-1,
        min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
        random_state=SEED, n_jobs=-1, verbose=-1)
    params.update(overrides)
    return lgb.LGBMRegressor(**params)


def build_xgb(**overrides):
    import xgboost as xgb
    params = dict(
        objective="reg:squarederror", n_estimators=1500,
        learning_rate=0.03, max_depth=8, min_child_weight=100,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", enable_categorical=True,
        random_state=SEED, n_jobs=-1)
    params.update(overrides)
    return xgb.XGBRegressor(**params)


def build_catboost(**overrides):
    from catboost import CatBoostRegressor
    # map lgbm/xgb-style keys to catboost where they differ
    if "n_estimators" in overrides:
        overrides["iterations"] = overrides.pop("n_estimators")
    params = dict(
        loss_function="RMSE", iterations=1500,
        learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
        random_seed=SEED, thread_count=-1, verbose=False,
        allow_writing_files=False)
    params.update(overrides)
    return CatBoostRegressor(**params)


BUILDERS = {"lgbm": build_lgbm, "xgb": build_xgb, "catboost": build_catboost}


def _cat_idx(X: pd.DataFrame) -> list[int]:
    return [X.columns.get_loc(c) for c in CATEGORICAL if c in X.columns]


def fit_predict(name: str, X_tr: pd.DataFrame, y_tr: np.ndarray,
                X_val: pd.DataFrame, target_mode: str = "log",
                builder_kwargs: dict | None = None):
    """Fit `name` model on (X_tr, y_tr) and predict X_val.

    Returns dict with predictions (clipped >=0), fitted model, and timings.
    Predictions are NOT yet closed-row-zeroed — the caller does that.
    """
    builder_kwargs = builder_kwargs or {}
    model = BUILDERS[name](**builder_kwargs)

    y = np.log1p(y_tr) if target_mode == "log" else y_tr.astype("float64")
    Xtr = as_model_frame(X_tr)
    Xvl = as_model_frame(X_val)

    t0 = time.time()
    if name == "catboost":
        from catboost import Pool
        model.fit(Pool(Xtr, y, cat_features=_cat_idx(Xtr)))
    else:
        model.fit(Xtr, y)
    train_time = time.time() - t0

    t1 = time.time()
    raw_pred = model.predict(Xvl)
    predict_time = time.time() - t1

    pred = np.expm1(raw_pred) if target_mode == "log" else raw_pred
    pred = np.clip(pred, 0, None)

    # in-sample fit for overfitting diagnostics
    ins = model.predict(Xtr)
    ins = np.expm1(ins) if target_mode == "log" else ins
    ins = np.clip(ins, 0, None)

    return {"pred": pred, "insample_pred": ins, "model": model,
            "train_time": train_time, "predict_time": predict_time}


def feature_importance(model, feature_names: list[str], name: str) -> pd.Series:
    if name == "catboost":
        imp = model.get_feature_importance()
    else:
        imp = model.feature_importances_
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)
