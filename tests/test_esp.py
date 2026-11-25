import numpy as np

from esp.data import FEATURES, TRUE_WEIGHTS, simulate
from esp.model import MODELS, agreement_with_truth, global_importance, shap_values


def test_simulation_shape_and_ranges():
    df = simulate(300, seed=1)
    assert len(df) == 300 and set(FEATURES) <= set(df.columns)
    assert df["attendance_rate"].between(0, 1).all()
    assert df["quiz_avg"].between(0, 100).all()
    assert set(df["passed"].unique()) <= {0, 1}


def test_simulation_is_reproducible():
    assert simulate(100, seed=3).equals(simulate(100, seed=3))


def test_shap_recovers_signal_and_ignores_noise():
    df = simulate(1500, seed=0)
    X, y = df[FEATURES], df["passed"]
    model = MODELS["logistic_regression"]().fit(X, y)
    sv = shap_values(model, X, X.iloc[:300])
    assert sv.shape == (300, len(FEATURES))
    imp = global_importance(sv, FEATURES)
    assert imp.index[0] == "quiz_avg"
    assert set(imp.index[-3:]) >= {"seat_row", "birth_month"} or set(imp.index[-2:]) == {"seat_row", "birth_month"}
    assert agreement_with_truth(imp, TRUE_WEIGHTS) > 0.7


def test_shap_additivity_for_trees():
    df = simulate(500, seed=2)
    X, y = df[FEATURES], df["passed"]
    model = MODELS["gradient_boosting"]().fit(X, y)
    sv = shap_values(model, X, X.iloc[:20])
    import shap
    base = shap.TreeExplainer(model).expected_value
    base = np.asarray(base).ravel()[-1]
    margin = model.decision_function(X.iloc[:20])
    assert np.allclose(sv.sum(axis=1) + base, margin, atol=1e-4)
