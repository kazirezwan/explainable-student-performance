"""Synthetic learning-analytics data with a known ground truth.

Real student records are private, so this module simulates a large
introductory ICT course. Because we choose the data-generating process,
we know which features truly drive the outcome and can check whether a
model *and its explanations* recover them.

Ground truth (logit of passing):
    +1.2 * quiz_avg          (standardised)
    +0.9 * assignments_rate
    +0.7 * attendance_rate
    +0.5 * practice_hours
    +0.4 * prior_digital_literacy
    +0.3 * lms_logins_per_week
    -0.6 * late_submissions
    +0.4 * has_personal_computer
     0.0 * seat_row, birth_month   (irrelevant by construction)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_WEIGHTS = {
    "quiz_avg": 1.2,
    "assignments_rate": 0.9,
    "attendance_rate": 0.7,
    "practice_hours": 0.5,
    "prior_digital_literacy": 0.4,
    "lms_logins_per_week": 0.3,
    "late_submissions": -0.6,
    "has_personal_computer": 0.4,
    "seat_row": 0.0,
    "birth_month": 0.0,
}
FEATURES = list(TRUE_WEIGHTS)


def simulate(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    literacy = rng.integers(1, 6, n)                       # 1..5 self-reported
    computer = rng.binomial(1, 0.35 + 0.1 * (literacy - 1))
    engagement = rng.normal(0, 1, n)                       # latent motivation
    df = pd.DataFrame({
        "attendance_rate": np.clip(0.75 + 0.12 * engagement + rng.normal(0, 0.1, n), 0, 1),
        "assignments_rate": np.clip(0.7 + 0.15 * engagement + rng.normal(0, 0.12, n), 0, 1),
        "quiz_avg": np.clip(55 + 8 * engagement + 4 * (literacy - 3) + rng.normal(0, 10, n), 0, 100),
        "practice_hours": np.clip(2 + 1.0 * engagement + 0.8 * computer + rng.normal(0, 1, n), 0, None),
        "prior_digital_literacy": literacy,
        "lms_logins_per_week": rng.poisson(np.exp(1.0 + 0.3 * engagement)),
        "late_submissions": rng.poisson(np.exp(0.5 - 0.4 * engagement)),
        "has_personal_computer": computer,
        "seat_row": rng.integers(1, 21, n),
        "birth_month": rng.integers(1, 13, n),
    })
    z = (df - df.mean()) / df.std()
    logit = sum(w * z[f] for f, w in TRUE_WEIGHTS.items()) - 0.3
    df["passed"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df
