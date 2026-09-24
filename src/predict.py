"""
predict.py — Phase 9/12: train final model(s) on ALL eligible data and build
the Kaggle submission.

* Trains on every training block up to the true cutoff (2015-06-19) via
  `make_supervised` (same code path as validation -> no leakage).
* Builds test features with history = ALL train, cutoff = 2015-06-19.
* Inverts log1p, clips to >=0, hard-sets IsOpen==0 rows to exactly 0.
* Preserves the test/sample_submission Id order, validates the shape, writes
  submissions/submission_v1.csv.

`ENSEMBLE` (list of (model, mode, weight)) and `BLEND_SPACE` are set from the
Phase-6/7 results; a single model is just a one-element ensemble.
"""
from __future__ import annotations
import sys, pathlib, json, datetime
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import (load_train, load_test, load_hub_metadata,
                      load_sample_submission, TEST_START)
from src.features import make_supervised, build_features, as_model_frame
from src.models import fit_predict
from src.validation import rmsle

SUB = ROOT / "submissions"; SUB.mkdir(exist_ok=True)

# Default config — overwritten by main() from ensemble_config.json if present.
ENSEMBLE = [("lgbm", "log", 1.0)]
BLEND_SPACE = "pred"        # "pred" or "log"


def _predict_test(df, test, hub, name, mode):
    cutoff = df["Date"].max()
    Xtr, ytr = make_supervised(df, hub, cutoff)
    Xte = build_features(df, test, hub, cutoff)
    r = fit_predict(name, Xtr, ytr, Xte, mode, {"n_estimators": 1500})
    return np.clip(r["pred"], 0, None)


def build_submission(config=None, out_name="submission_v1.csv", val_rmsle=None):
    df = load_train(); test = load_test(); hub = load_hub_metadata()
    ss = load_sample_submission()
    config = config or {"ensemble": ENSEMBLE, "blend_space": BLEND_SPACE}
    ens = config["ensemble"]; space = config.get("blend_space", "pred")

    preds, wsum = None, 0.0
    for name, mode, w in ens:
        p = _predict_test(df, test, hub, name, mode)
        pv = np.log1p(p) if space == "log" else p
        preds = pv * w if preds is None else preds + pv * w
        wsum += w
    preds = preds / wsum
    final = np.expm1(preds) if space == "log" else preds
    final = np.clip(final, 0, None)

    # hard-set closed rows to exactly 0
    final = np.where(test["IsOpen"].to_numpy() == 1, final, 0.0)

    # assemble in the sample_submission Id order (rule #13)
    out = pd.DataFrame({"Id": test["Id"].to_numpy(), "OrderVolume": final})
    out = out.set_index("Id").reindex(ss["Id"]).reset_index()

    # ---- final validation checks (rules #11-15) ----
    checks = {
        "rows_match_sample": len(out) == len(ss),
        "ids_match_sample_order": bool((out["Id"].to_numpy() == ss["Id"].to_numpy()).all()),
        "no_missing": int(out["OrderVolume"].isna().sum()) == 0,
        "no_negative": bool((out["OrderVolume"] >= 0).all()),
        "columns_ok": list(out.columns) == list(ss.columns),
    }
    assert all(checks.values()), f"submission checks failed: {checks}"

    out.to_csv(SUB / out_name, index=False)

    report = [
        "# Final Model Report (Phase 9)\n",
        f"* generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"* ensemble: {ens}", f"* blend space: {space}",
        f"* training rows (supervised blocks): computed on full train up to {df['Date'].max().date()}",
        f"* test rows: {len(out)}", f"* validation RMSLE (holdout): {val_rmsle}",
        "\n## Submission checks",
    ]
    for k, v in checks.items():
        report.append(f"* {k}: {'PASS' if v else 'FAIL'}")
    report.append(f"\n## Prediction summary\n{out['OrderVolume'].describe().to_string()}")
    report.append(f"\n* zero predictions (closed hubs): {(out['OrderVolume']==0).sum()}")
    (ROOT / "outputs" / "final_model_report.md").write_text("\n".join(report))
    print("checks:", checks)
    print("wrote", SUB / out_name)
    return out, checks


if __name__ == "__main__":
    cfg_path = ROOT / "outputs" / "ensemble_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    if cfg and "ensemble" in cfg:
        cfg["ensemble"] = [tuple(x) for x in cfg["ensemble"]]
    build_submission(cfg)
