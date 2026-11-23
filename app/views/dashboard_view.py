"""Layout of the Dashboard page (notebook Section 8.2, "Dashboard")."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from app.config import MIN_GROUP_SIZE
from app.models.student_analytics import GroupHighlight, StatusSummary
from app.views.components import charts
from app.views.components.filters import filter_bar
from app.views.components.kpi_cards import Kpi, kpi_row
from app.views.components.layout import (
    about_ratio,
    callout,
    card,
    card_title,
    chart_card,
    empty_state,
    grid,
    page_header,
    pct,
    pp_delta,
    section_heading,
)
from app.views.display_names import PROGRESS_MEASURES, RATE_MEASURES, field_label

Formatter = Callable[[str, object], str]
KPI_COLUMNS = 4  # shared by the KPI row and the highlight row


def header() -> None:
    page_header(
        "Insights · Student outcomes",
        "Dashboard",
        "Who drops out, where is the risk concentrated, and how does first-year "
        "progress differ between students who leave and students who graduate?",
    )


def filters(options: dict[str, list], format_option: Formatter) -> dict[str, list]:
    return filter_bar(options, format_option)


def no_students() -> None:
    empty_state(
        "No students match these filters",
        "Remove one of the filters above to see results again.",
    )


def kpis(selection: StatusSummary, everyone: StatusSummary, filtered: bool) -> None:
    """Headline numbers; deltas compare the selection with all students."""

    def delta(status: str) -> str | None:
        return (
            pp_delta(selection.share(status), everyone.share(status))
            if filtered
            else None
        )

    scope = "in the current selection" if filtered else "in the institutional records"
    kpi_row(
        columns=KPI_COLUMNS,
        kpis=[
            Kpi(
                "Students",
                f"{selection.students:,}",
                f"{selection.students / everyone.students:.0%} of all students "
                f"{'(filtered)' if filtered else ''}".strip(),
            ),
            Kpi(
                "Dropout rate",
                pct(selection.share("Dropout")),
                f"{selection.counts['Dropout']:,} students left without a degree.",
                delta=delta("Dropout"),
                delta_color="inverse",
                help="Share of students whose final status is Dropout.",
            ),
            Kpi(
                "Graduation rate",
                pct(selection.share("Graduate")),
                f"{selection.counts['Graduate']:,} students graduated {scope}.",
                delta=delta("Graduate"),
            ),
            Kpi(
                "Still enrolled",
                pct(selection.share("Enrolled")),
                "Not finished at the end of the normal course duration.",
                delta=delta("Enrolled"),
                delta_color="off",
            ),
        ],
    )


def highlights(
    items: list[GroupHighlight], overall: float, format_option: Formatter
) -> None:
    """Key message plus the groups whose dropout rate stands out most."""
    if not items:
        return
    top = items[0]
    callout(
        f"**{field_label(top.dimension)}: {format_option(top.dimension, top.group)}** "
        f"has the highest dropout rate in this view: {top.rate:.0%}, "
        f"{top.lift:.1f} times the average of {overall:.0%}."
    )
    # Same column count as the KPI row, so both rows share one card width.
    for column, item in zip(grid(KPI_COLUMNS), items, strict=False):
        with column, card():
            st.caption(field_label(item.dimension))
            card_title(format_option(item.dimension, item.group))
            st.metric(
                "Dropout rate",
                f"{item.rate:.0%}",
                delta=f"{item.lift:.1f}× the average",
                delta_color="inverse",
                label_visibility="collapsed",
            )
            st.caption(f"{item.students:,} students")


def rate_controls(dimensions: tuple[str, ...]) -> tuple[str, bool]:
    """Dimension selector and the small-group toggle."""
    left, right = grid(2, vertical_alignment="bottom")
    dimension = left.selectbox(
        "Compare dropout rates by",
        dimensions,
        format_func=field_label,
        key="rate_dimension",
    )
    hide_small = right.toggle(
        f"Hide groups under {MIN_GROUP_SIZE} students",
        value=True,
        key="rate_hide_small",
    )
    return dimension, hide_small


def rate_section(table: pd.DataFrame, overall: float, dimension: str) -> None:
    """Ranked dropout rate per group with the overall average."""
    title = field_label(dimension)
    if table.empty:
        empty_state(
            "No group is large enough", "Turn off the toggle to see small groups."
        )
        return
    top = table.iloc[0]
    chart_card(
        f"Dropout rate by {title.lower()}",
        lambda: st.altair_chart(
            charts.dropout_rate_chart(table, overall, title), width="stretch"
        ),
        f"{top['group']} has the highest rate ({top['rate']:.0%}, "
        f"{top['lift']:.1f}× the average). The dashed line is the average "
        f"for the current selection ({overall:.0%}). "
        "Rates describe patterns, not causes.",
        table.rename(
            columns={
                "group": title,
                "students": "Students",
                "dropouts": "Dropouts",
                "rate": "Dropout rate",
                "lift": "Times the average",
                "too_small": "Under 50 students",
            }
        ),
        key="rates",
    )


def status_section(distribution: pd.DataFrame) -> None:
    dropout = distribution.set_index("Status").loc["Dropout"]
    chart_card(
        "Final status of students",
        lambda: st.altair_chart(
            charts.status_distribution_chart(distribution), width="stretch"
        ),
        f"About {about_ratio(dropout['Share'])} students in this view dropped out "
        f"({int(dropout['Students']):,} students).",
        distribution.assign(Share=distribution["Share"].map(lambda s: f"{s:.1%}")),
        key="status",
    )


def progress_controls() -> str:
    """Measure selector, shared by the two cards below it."""
    return grid(2)[0].selectbox(
        "Progress measure",
        list(PROGRESS_MEASURES),
        format_func=field_label,
        key="progress_measure",
    )


def progress_section(
    data: pd.DataFrame, column: str, medians: dict[str, float]
) -> None:
    """Box plot of a first-year progress measure per final status."""
    as_percent = column in RATE_MEASURES
    fmt = (lambda v: f"{v:.0%}") if as_percent else (lambda v: f"{v:.1f}")
    summary = ", ".join(f"{status} {fmt(value)}" for status, value in medians.items())
    chart_card(
        f"{field_label(column)} by final status",
        lambda: st.altair_chart(
            charts.progress_boxplot(data, column, field_label(column), as_percent),
            width="stretch",
        ),
        f"{PROGRESS_MEASURES[column]}. Medians: {summary}. Each box holds the "
        "middle half of the students.",
        pd.DataFrame(
            {"Status": list(medians), "Median": [fmt(v) for v in medians.values()]}
        ),
        key="progress",
    )


def section(title: str, eyebrow: str) -> None:
    section_heading(title, eyebrow)


def two_columns():
    """Two equal columns for side-by-side cards."""
    return grid(2)
