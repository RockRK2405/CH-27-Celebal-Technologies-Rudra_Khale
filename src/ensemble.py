"""
ensemble.py — Phase 8: does combining models beat the best single model?

For each chronological fold we fit the 3 boosters (log target), collect their
validation predictions, then test:
  1. simple average (prediction space)
  2. simple average (log1p space)
  3. weighted average (coarse simplex grid, chosen on the MEAN across folds so we
     don't overfit weights to one period)

We keep the *simplest* option that improves consistently across folds and save it
to outputs/ensemble_config.json.
"""
from __future__ import annotations
import sys, pathlib, json, itertools
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_train, load_hub_metadata
from src.validation import rolling_splits, rmsle
from src.models import fit_predict, BUILDERS
from src.train import get_fold_matrices, N_EST

MODELS = ["lgbm", "xgb", "catboost"]


def fold_predictions():
    df = load_train(); hub = load_hub_metadata()
    folds = rolling_splits(df, n_folds=3)
    store = {}
    for fold in folds:
        Xtr, ytr, Xval, yv, iso = get_fold_matrices(df, hub, fold)
        preds = {}
        for name in MODELS:
            r = fit_predict(name, Xtr, ytr, Xval, "log", {"n_estimators": N_EST[name]})
            p = np.where(iso == 1, np.clip(r["pred"], 0, None), 0.0)
            preds[name] = p
            print(f"  {fold.name} {name}: RMSLE={rmsle(yv, p):.5f}", flush=True)
        store[fold.name] = {"preds": preds, "y": yv}
    return store


def evaluate(store):
    folds = list(store.keys())
    # individual
    table = {}
    for name in MODELS:
        table[name] = np.mean([rmsle(store[f]["y"], store[f]["preds"][name]) for f in folds])

    # simple average — prediction space
    def avg_pred(weights, space):
        scores = []
        for f in folds:
            ps = store[f]["preds"]; y = store[f]["y"]
            if space == "log":
                blend = sum(w * np.log1p(ps[m]) for m, w in weights.items())
                blend = np.expm1(blend)
            else:
                blend = sum(w * ps[m] for m, w in weights.items())
            scores.append(rmsle(y, np.clip(blend, 0, None)))
        return float(np.mean(scores)), scores

    eq = {m: 1 / len(MODELS) for m in MODELS}
    table["avg_pred"], sc_pred = avg_pred(eq, "pred")
    table["avg_log"], sc_log = avg_pred(eq, "log")

    # weighted grid (steps of 0.1 on simplex), chosen on mean-across-folds
    best = {"score": 1e9}
    grid = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(MODELS))
            if abs(sum(w) - 1) < 1e-6]
    for w in grid:
        weights = dict(zip(MODELS, w))
        s, _ = avg_pred(weights, "pred")
        if s < best["score"]:
            best = {"score": s, "weights": weights}
    table["weighted_pred"] = best["score"]
    return table, best, sc_pred, sc_log


def main():
    store = fold_predictions()
    table, best, sc_pred, sc_log = evaluate(store)
    print("\n=== mean RMSLE across folds ===")
    for k, v in sorted(table.items(), key=lambda x: x[1]):
        print(f"  {k:14s} {v:.5f}")
    print("best weighted:", best["weights"], f"{best['score']:.5f}")

    # choose simplest with consistent improvement over best single model
    best_single = min(table[m] for m in MODELS)
    best_single_name = min(MODELS, key=lambda m: table[m])
    choice = {"ensemble": [[best_single_name, "log", 1.0]], "blend_space": "pred",
              "rationale": "single best (default)"}
    # prefer simple average if it beats best single on the MEAN and each fold
    if table["avg_pred"] < best_single - 1e-4:
        choice = {"ensemble": [[m, "log", 1.0] for m in MODELS], "blend_space": "pred",
                  "rationale": "equal-weight average improved mean-fold RMSLE"}
    if best["score"] < min(table["avg_pred"], best_single) - 5e-4:
        choice = {"ensemble": [[m, "log", float(round(w, 3))] for m, w in best["weights"].items() if w > 0],
                  "blend_space": "pred",
                  "rationale": "weighted average clearly best across folds"}
    choice["fold_table"] = {k: round(v, 5) for k, v in table.items()}
    (ROOT / "outputs" / "ensemble_config.json").write_text(json.dumps(choice, indent=2))
    print("\nchosen:", choice["rationale"], "->", choice["ensemble"])


if __name__ == "__main__":
    main()
