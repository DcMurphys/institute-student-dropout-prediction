"""Rebuild the engineered features and log transform used in the notebook.

The formulas mirror notebook Section 5.1 (feature engineering) and Section 5.4
(log transform) exactly, so the pipeline receives the same inputs it was
trained on. Change them only together with the notebook.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

SEMESTER_PREFIXES = {
    "Sem1": "Curricular_units_1st_sem",
    "Sem2": "Curricular_units_2nd_sem",
}
PREV_QUAL_GRADE_SCALE = 10  # previous grade is on 0-200, curricular grades on 0-20

ENGINEERED_FEATURES = (
    "Sem1_approval_rate",
    "Sem2_approval_rate",
    "Overall_approval_rate",
    "Sem1_without_evaluation",
    "Sem2_without_evaluation",
    "Change_units_without_evaluations",
    "Overall_average_grade",
    "Grade_vs_prev_qual",
    "Sem1_approved_to_credited_ratio",
    "Sem2_approved_to_credited_ratio",
    "Overall_approved_to_credited_ratio",
)


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide element-wise, returning 0 where the denominator is 0.

    Example:
        >>> safe_ratio(pd.Series([3, 1]), pd.Series([6, 0])).tolist()
        [0.5, 0.0]
    """
    return (numerator / denominator.replace(0, np.nan)).fillna(0)


def add_engineered_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``raw`` with every engineered feature added.

    Args:
        raw: Student records with the original curricular-unit columns.

    Returns:
        A new DataFrame; ``raw`` is not modified.
    """
    data = raw.copy()
    for short, prefix in SEMESTER_PREFIXES.items():
        data[f"{short}_approval_rate"] = safe_ratio(
            data[f"{prefix}_approved"], data[f"{prefix}_enrolled"]
        )
    data["Overall_approval_rate"] = data[
        ["Sem1_approval_rate", "Sem2_approval_rate"]
    ].mean(axis=1)

    for short, prefix in SEMESTER_PREFIXES.items():
        data[f"{short}_without_evaluation"] = safe_ratio(
            data[f"{prefix}_without_evaluations"], data[f"{prefix}_enrolled"]
        )
    data["Change_units_without_evaluations"] = (
        data["Sem2_without_evaluation"] - data["Sem1_without_evaluation"]
    )

    weighted_grade_sum = sum(
        data[f"{prefix}_grade"] * data[f"{prefix}_approved"]
        for prefix in SEMESTER_PREFIXES.values()
    )
    total_approved = sum(
        data[f"{prefix}_approved"] for prefix in SEMESTER_PREFIXES.values()
    )
    data["Overall_average_grade"] = safe_ratio(weighted_grade_sum, total_approved)
    data["Grade_vs_prev_qual"] = (
        data["Overall_average_grade"]
        - data["Previous_qualification_grade"] / PREV_QUAL_GRADE_SCALE
    )

    for short, prefix in SEMESTER_PREFIXES.items():
        data[f"{short}_approved_to_credited_ratio"] = safe_ratio(
            data[f"{prefix}_approved"], data[f"{prefix}_credited"]
        )
    total_credited = sum(
        data[f"{prefix}_credited"] for prefix in SEMESTER_PREFIXES.values()
    )
    data["Overall_approved_to_credited_ratio"] = safe_ratio(
        total_approved, total_credited
    )
    return data


def apply_log_transform(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with ``log1p`` applied to the listed columns that are present."""
    transformed = data.copy()
    present = [column for column in columns if column in transformed.columns]
    transformed[present] = np.log1p(transformed[present])
    return transformed
