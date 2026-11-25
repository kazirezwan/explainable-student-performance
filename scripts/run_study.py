"""Train, evaluate and explain student-performance models.

Synthetic data (default):
    python scripts/run_study.py
Your own anonymised CSV (numeric features + a 0/1 target column):
    python scripts/run_study.py --csv my_class.csv --target passed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from esp.data import FEATURES, TRUE_WEIGHTS, simulate  # noqa: E402
from esp.model import MODELS, agreement_with_truth, cross_validate, global_importance, shap_values  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--target", default="passed")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        features = [c for c in df.columns if c != args.target]
        truth = None
    else:
        df = simulate(args.n, seed=0)
        features, truth = FEATURES, TRUE_WEIGHTS
    X, y = df[features], df[args.target]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    cv = cross_validate(X, y)
    cv.to_csv(out / "cross_validation.csv", index=False)
    print(cv.round(3).to_string(index=False))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    report, importances = {"pass_rate": float(y.mean())}, {}
    for name, make in MODELS.items():
        model = make().fit(X_tr, y_tr)
        sv = shap_values(model, X_tr, X_te)
        imp = global_importance(sv, features)
        importances[name] = imp
        entry = {"top_features": imp.head(5).round(3).to_dict()}
        if truth:
            entry["spearman_vs_truth"] = round(agreement_with_truth(imp, truth), 3)
            entry["irrelevant_features_rank"] = {f: int(list(imp.index).index(f)) + 1
                                                 for f in ("seat_row", "birth_month")}
        report[name] = entry
        if name == "gradient_boosting":
            proba = model.predict_proba(X_te)[:, 1]
            i = int(np.argmin(proba))   # the student the model is most worried about
            local = pd.Series(sv[i], index=features).sort_values()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(local.index, local.values, color=["#c0392b" if v < 0 else "#2e86c1" for v in local.values])
            ax.set_title(f"Why is this student flagged? (P(pass) = {proba[i]:.2f})")
            ax.set_xlabel("SHAP value (log-odds of passing)")
            ax.axvline(0, color="k", lw=0.8)
            fig.tight_layout()
            fig.savefig(out / "local_explanation.png", dpi=150)
            report["flagged_student"] = {"p_pass": round(float(proba[i]), 3),
                                         "features": X_te.iloc[i].round(2).to_dict(),
                                         "shap": local.round(3).to_dict()}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    imp_df = pd.DataFrame(importances).loc[importances["gradient_boosting"].index[::-1]]
    imp_df = imp_df / imp_df.max()
    if truth:
        imp_df["ground truth |w|"] = pd.Series({k: abs(v) for k, v in truth.items()}) / max(abs(v) for v in truth.values())
    imp_df.plot.barh(ax=ax, width=0.8)
    ax.set_xlabel("relative importance (scaled to max = 1)")
    ax.set_title("Global importance: SHAP vs ground truth")
    fig.tight_layout()
    fig.savefig(out / "global_importance.png", dpi=150)

    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
