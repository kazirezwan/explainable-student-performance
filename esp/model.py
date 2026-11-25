"""Models, evaluation and SHAP explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=0),
}


def cross_validate(X: pd.DataFrame, y: pd.Series, folds: int = 5) -> pd.DataFrame:
    cv = StratifiedKFold(folds, shuffle=True, random_state=0)
    rows = []
    for name, make in MODELS.items():
        for metric in ("roc_auc", "accuracy", "f1"):
            s = cross_val_score(make(), X, y, cv=cv, scoring=metric)
            rows.append(dict(model=name, metric=metric, mean=s.mean(), std=s.std()))
    return pd.DataFrame(rows)


def shap_values(model, X_background: pd.DataFrame, X_explain: pd.DataFrame) -> np.ndarray:
    """SHAP values (log-odds space) for the positive class."""
    if isinstance(model, GradientBoostingClassifier):
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X_explain)
    else:  # linear pipeline: explain on the scaled inputs
        scaler, lr = model[0], model[-1]
        explainer = shap.LinearExplainer(lr, scaler.transform(X_background))
        vals = explainer.shap_values(scaler.transform(X_explain))
    vals = np.asarray(vals)
    return vals[..., 1] if vals.ndim == 3 else vals


def global_importance(sv: np.ndarray, features: list[str]) -> pd.Series:
    return pd.Series(np.abs(sv).mean(axis=0), index=features).sort_values(ascending=False)


def agreement_with_truth(importance: pd.Series, true_weights: dict) -> float:
    """Spearman correlation between |true weight| and mean |SHAP|."""
    truth = pd.Series({k: abs(v) for k, v in true_weights.items()})
    return float(spearmanr(truth[importance.index], importance).statistic)
