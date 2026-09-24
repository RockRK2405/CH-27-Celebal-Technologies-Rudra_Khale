"""
tune.py — Phase 6: controlled hyperparameter search (Optuna) for the strongest
model, optimized against chronological-validation RMSLE.

To avoid overfitting parameters to a single window, the objective is the MEAN
holdout-fold RMSLE over the two most-recent folds (fold2 + fold3). We keep the
search modest (rule #17): a bounded TPE search over the standard booster knobs.

Outputs
  outputs/hyperparameter_results/best_params.json
  outputs/hyperparameter_results/trials.csv
  outputs/hyperparameter_results/study_summary.md
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
import pandas as pd
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_train, load_hub_metadata
from src.validation import rolling_splits, rmsle
from src.features import as_model_frame, CATEGORICAL
from src.train import get_fold_matrices

OUT = ROOT / "outputs" / "hyperparameter_results"; OUT.mkdir(parents=True, exist_ok=True)
N_TRIALS = 12
SEED = 42


def _save_best(study, model_name):
    """Persist best params after every trial so an early stop loses nothing."""
    (OUT / "best_params.json").write_text(json.dumps(
        {"model": model_name, "best_value_meanfold_rmsle": study.best_value,
         "params": study.best_params}, indent=2))


def _load_folds():
    df = load_train(); hub = load_hub_metadata()
    folds = rolling_splits(df, n_folds=3)[-2:]      # fold2, fold3
    data = []
    for f in folds:
        Xtr, ytr, Xval, yv, iso = get_fold_matrices(df, hub, f)
        data.append((as_model_frame(Xtr), np.log1p(ytr), as_model_frame(Xval), yv, iso))
    return data


def objective_lgbm(trial, data):
    import lightgbm as lgb
    params = dict(
        objective="regression", n_jobs=-1, verbose=-1, random_state=SEED,
        n_estimators=trial.suggest_int("n_estimators", 500, 2000, step=250),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 255),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 300),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    )
    scores = []
    for Xtr, ytr, Xval, yv, iso in data:
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, ytr)
        p = np.where(iso == 1, np.clip(np.expm1(m.predict(Xval)), 0, None), 0.0)
        scores.append(rmsle(yv, p))
    return float(np.mean(scores))


def main(model_name=None):
    if model_name is None:
        bm = pd.read_csv(ROOT / "outputs" / "model_benchmark.csv")
        main_f = bm[(bm.fold == "rolling_fold3") & (bm.target_mode == "log")]
        model_name = main_f.sort_values("rmsle_all").iloc[0]["model"]
    print("tuning:", model_name)
    data = _load_folds()

    obj = {"lgbm": objective_lgbm}.get(model_name, objective_lgbm)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: obj(t, data), n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[lambda st, tr: _save_best(st, model_name)])

    best = study.best_params
    (OUT / "best_params.json").write_text(json.dumps(
        {"model": model_name, "best_value_meanfold_rmsle": study.best_value,
         "params": best}, indent=2))
    study.trials_dataframe().to_csv(OUT / "trials.csv", index=False)
    summary = [f"# Hyperparameter tuning — {model_name}\n",
               f"* trials: {len(study.trials)}",
               f"* best mean-fold RMSLE: **{study.best_value:.5f}**",
               f"* best params:\n```json\n{json.dumps(best, indent=2)}\n```"]
    (OUT / "study_summary.md").write_text("\n".join(summary))
    print(f"best mean-fold RMSLE={study.best_value:.5f}")
    print("params:", best)


if __name__ == "__main__":
    main()
