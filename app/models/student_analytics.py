"""Dashboard analytics: filters, status summary, dropout rates and highlights.

Rates follow notebook Section 4.4.10: dropout rate per group, lift against
the overall rate, and groups below ``MIN_GROUP_SIZE`` marked as too small.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.config import MIN_GROUP_SIZE, POSITIVE_CLASS, STATUS_ORDER, TARGET

AGE_BAND = "Age_band"
AGE_BINS = [0, 20, 24, 34, 200]
AGE_LABELS = ["20 or younger", "21-24", "25-34", "35 or older"]

# Dimensions offered in "Dropout rate by group" (notebook Section 4.4.10)
RATE_DIMENSIONS = (
    "Tuition_fees_up_to_date",
    "Debtor",
    "Scholarship_holder",
    AGE_BAND,
    "Daytime_evening_attendance",
    "Course",
    "Application_mode",
    "Previous_qualification",
    "Marital_status",
    "Gender",
    "Displaced",
    "International",
)
# Dimensions scanned for the automatic "what stands out" highlights
HIGHLIGHT_DIMENSIONS = (
    "Tuition_fees_up_to_date",
    "Debtor",
    "Scholarship_holder",
    AGE_BAND,
    "Daytime_evening_attendance",
    "Course",
    "Application_mode",
)


@dataclass(frozen=True)
class StatusSummary:
    """Headline numbers for a set of students."""

    students: int
    counts: dict[str, int]

    def share(self, status: str) -> float | None:
        return self.counts.get(status, 0) / self.students if self.students else None


@dataclass(frozen=True)
class GroupHighlight:
    """One group whose dropout rate stands out from the average."""

    dimension: str
    group: object
    rate: float
    lift: float
    students: int


def add_age_band(data: pd.DataFrame) -> pd.DataFrame:
    """Add an ordered age band column used by rate tables and filters."""
    banded = data.copy()
    banded[AGE_BAND] = pd.cut(
        banded["Age_at_enrollment"], bins=AGE_BINS, labels=AGE_LABELS
    ).astype(str)
    return banded


def apply_filters(data: pd.DataFrame, filters: dict[str, list]) -> pd.DataFrame:
    """Keep rows whose value is in every non-empty filter selection."""
    mask = pd.Series(True, index=data.index)
    for column, selected in filters.items():
        if selected:
            mask &= data[column].isin(selected)
    return data[mask]


def status_summary(data: pd.DataFrame) -> StatusSummary:
    """Student count and count per status."""
    counts = data[TARGET].value_counts()
    return StatusSummary(
        students=len(data),
        counts={status: int(counts.get(status, 0)) for status in STATUS_ORDER},
    )


def status_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Students and share per status, in the canonical status order."""
    summary = status_summary(data)
    return pd.DataFrame(
        {
            "Status": list(STATUS_ORDER),
            "Students": [summary.counts[s] for s in STATUS_ORDER],
            "Share": [summary.share(s) or 0.0 for s in STATUS_ORDER],
        }
    )


def dropout_rate(data: pd.DataFrame) -> float | None:
    """Share of students who dropped out (None for an empty selection)."""
    return data[TARGET].eq(POSITIVE_CLASS).mean() if len(data) else None


def dropout_rate_table(
    data: pd.DataFrame, column: str, min_group_size: int = MIN_GROUP_SIZE
) -> pd.DataFrame:
    """Dropout rate and lift per group of ``column``, highest rate first.

    Args:
        data: Students with ``Status`` and ``column``.
        column: Grouping column (codes or labels).
        min_group_size: Groups below this size are flagged as too small.

    Returns:
        Columns ``group``, ``students``, ``dropouts``, ``rate``, ``lift``,
        ``too_small``.
    """
    overall = dropout_rate(data)
    table = (
        data.assign(_dropout=data[TARGET].eq(POSITIVE_CLASS))
        .groupby(column, observed=True)["_dropout"]
        .agg(students="size", dropouts="sum")
        .reset_index()
        .rename(columns={column: "group"})
    )
    table["rate"] = table["dropouts"] / table["students"]
    table["lift"] = table["rate"] / overall if overall else float("nan")
    table["too_small"] = table["students"] < min_group_size
    return table.sort_values(
        ["too_small", "rate"], ascending=[True, False]
    ).reset_index(drop=True)


def top_highlights(
    data: pd.DataFrame,
    dimensions: tuple[str, ...] = HIGHLIGHT_DIMENSIONS,
    n: int = 3,
    min_group_size: int = MIN_GROUP_SIZE,
) -> list[GroupHighlight]:
    """Groups with the highest lift, at most one per dimension.

    Only groups large enough to give a stable rate are considered.
    """
    candidates = []
    for dimension in dimensions:
        table = dropout_rate_table(data, dimension, min_group_size)
        table = table[~table["too_small"]]
        if table.empty or table["lift"].isna().all():
            continue
        best = table.loc[table["lift"].idxmax()]
        candidates.append(
            GroupHighlight(
                dimension=dimension,
                group=best["group"],
                rate=float(best["rate"]),
                lift=float(best["lift"]),
                students=int(best["students"]),
            )
        )
    return sorted(candidates, key=lambda h: h.lift, reverse=True)[:n]


def progress_by_status(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Values of one academic-progress column with each student's status."""
    return data[[TARGET, column]].dropna()


def median_by_status(data: pd.DataFrame, column: str) -> dict[str, float]:
    """Median of ``column`` per status (for chart captions)."""
    medians = data.groupby(TARGET)[column].median()
    return {s: float(medians[s]) for s in STATUS_ORDER if s in medians}


def typical_profile(data: pd.DataFrame, columns: list[str], categorical: tuple) -> dict:
    """Most common code (categorical) or median (numeric) per input column."""
    profile = {}
    for column in columns:
        series = data[column].dropna()
        if column in categorical:
            profile[column] = int(series.mode().iloc[0])
        else:
            profile[column] = float(series.median())
    return profile


def example_profiles(data: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    """Real records used as form presets: one dropout and one graduate example.

    The dropout example has fees in arrears and a low first-year approval rate;
    the graduate example passed every enrolled unit. Both are the first match
    in file order, so presets are stable between runs.
    """
    approval = (
        data["Overall_approval_rate"] if "Overall_approval_rate" in data else None
    )
    at_risk = data[
        data[TARGET].eq(POSITIVE_CLASS)
        & data["Tuition_fees_up_to_date"].eq(0)
        & data["Curricular_units_1st_sem_enrolled"].gt(0)
        & (approval.lt(0.5) if approval is not None else True)
    ]
    on_track = data[
        data[TARGET].eq("Graduate")
        & data["Tuition_fees_up_to_date"].eq(1)
        & (approval.eq(1) if approval is not None else True)
    ]
    examples = {}
    if not at_risk.empty:
        examples["Example: student at risk"] = at_risk.iloc[0][columns].to_dict()
    if not on_track.empty:
        examples["Example: student on track"] = on_track.iloc[0][columns].to_dict()
    return examples
