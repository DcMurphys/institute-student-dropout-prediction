"""Input schema and validation for single-student and batch predictions.

Rules are checked on raw values (before feature engineering). Errors block a
row from scoring; warnings are shown but the row is still scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.models.feature_engineering import ENGINEERED_FEATURES, SEMESTER_PREFIXES

CATEGORICAL_INPUTS = (
    "Marital_status",
    "Application_order",
    "Course",
    "Daytime_evening_attendance",
    "Previous_qualification",
    "Displaced",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Gender",
    "Scholarship_holder",
)

UNIT_LIMIT = 60  # generous upper bound for curricular units per semester
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "Previous_qualification_grade": (0, 200),
    "Admission_grade": (0, 200),
    "Age_at_enrollment": (15, 80),
    **{
        f"{prefix}_{suffix}": (0, UNIT_LIMIT)
        for prefix in SEMESTER_PREFIXES.values()
        for suffix in (
            "credited",
            "enrolled",
            "evaluations",
            "approved",
            "without_evaluations",
        )
    },
    **{f"{prefix}_grade": (0, 20) for prefix in SEMESTER_PREFIXES.values()},
}

ID_COLUMN_CANDIDATES = ("Student_ID", "student_id", "ID", "id", "NIM")


def required_input_columns(feature_columns: list[str]) -> list[str]:
    """Raw columns a user must provide: model features minus engineered ones."""
    return [column for column in feature_columns if column not in ENGINEERED_FEATURES]


def find_id_column(columns: pd.Index) -> str | None:
    """Return the first column that looks like a student identifier."""
    return next((c for c in ID_COLUMN_CANDIDATES if c in columns), None)


@dataclass
class ValidationReport:
    """Outcome of validating a batch of student records."""

    missing_columns: list[str] = field(default_factory=list)
    errors: pd.Series | None = None  # one list of messages per row
    warnings: pd.Series | None = None

    @property
    def is_usable(self) -> bool:
        """True when every required column is present."""
        return not self.missing_columns

    @property
    def valid_mask(self) -> pd.Series:
        """Rows without blocking errors."""
        return self.errors.map(len).eq(0)


def _append(messages: pd.Series, mask: pd.Series, text: str) -> None:
    for index in mask[mask].index:
        messages.loc[index].append(text)


def validate_records(
    records: pd.DataFrame,
    required_columns: list[str],
    category_options: dict[str, list[int]],
) -> ValidationReport:
    """Check columns, value types, ranges and cross-field consistency.

    Args:
        records: Uploaded or form-entered student records (raw values).
        required_columns: Columns the model needs (``required_input_columns``).
        category_options: Codes seen in training for each categorical input.

    Returns:
        A ``ValidationReport``; rows are not modified.
    """
    missing = [column for column in required_columns if column not in records.columns]
    if missing:
        return ValidationReport(missing_columns=missing)

    errors = pd.Series([[] for _ in range(len(records))], index=records.index)
    warnings = pd.Series([[] for _ in range(len(records))], index=records.index)
    numeric = records[required_columns].apply(pd.to_numeric, errors="coerce")

    for column in required_columns:
        _append(errors, numeric[column].isna(), f"{column}: missing or not a number")

    for column, (low, high) in NUMERIC_BOUNDS.items():
        if column in numeric:
            outside = numeric[column].notna() & ~numeric[column].between(low, high)
            _append(errors, outside, f"{column}: must be between {low:g} and {high:g}")

    for column, options in category_options.items():
        if column in numeric:
            unknown = numeric[column].notna() & ~numeric[column].isin(options)
            _append(warnings, unknown, f"{column}: code not seen in training data")

    for prefix in SEMESTER_PREFIXES.values():
        semester = prefix.replace("Curricular_units_", "").replace("_", " ")
        approved = numeric[f"{prefix}_approved"]
        _append(
            errors,
            approved > numeric[f"{prefix}_enrolled"],
            f"{semester}: approved units exceed enrolled units",
        )
        _append(
            errors,
            approved > numeric[f"{prefix}_evaluations"],
            f"{semester}: approved units exceed evaluations",
        )
        grade = numeric[f"{prefix}_grade"]
        _append(
            errors,
            (approved == 0) & (grade > 0),
            f"{semester}: grade must be 0 when no unit is approved",
        )
        _append(
            errors,
            (approved > 0) & (grade == 0),
            f"{semester}: grade is missing for approved units",
        )
        _append(
            warnings,
            numeric[f"{prefix}_without_evaluations"] > numeric[f"{prefix}_enrolled"],
            f"{semester}: units without evaluation exceed enrolled units",
        )
    return ValidationReport(errors=errors, warnings=warnings)
