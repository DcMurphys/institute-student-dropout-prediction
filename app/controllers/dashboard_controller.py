"""Dashboard page: filters → analytics → charts."""

from __future__ import annotations

from app.controllers import resources
from app.models import student_analytics as analytics
from app.models.student_repository import decode_labels
from app.views import dashboard_view as view

# Filters requested in notebook Section 8.2
FILTER_COLUMNS = (
    "Course",
    "Application_mode",
    "Daytime_evening_attendance",
    "Scholarship_holder",
)


def show_dashboard_page() -> None:
    everyone = resources.students()
    view.header()

    options = {
        column: sorted(everyone[column].unique().tolist()) for column in FILTER_COLUMNS
    }
    selection = view.filters(options, resources.format_option)
    data = analytics.apply_filters(everyone, selection)
    if data.empty:
        view.no_students()
        return
    filtered = any(selection.values())
    overall = analytics.dropout_rate(data)

    view.kpis(
        analytics.status_summary(data), analytics.status_summary(everyone), filtered
    )
    readable = decode_labels(data, resources.label_mappings())
    view.highlights(
        analytics.top_highlights(readable, n=view.KPI_COLUMNS),
        overall,
        resources.format_option,
    )

    view.section("Where dropout concentrates", "Risk by group")
    dimension, hide_small = view.rate_controls(analytics.RATE_DIMENSIONS)
    table = analytics.dropout_rate_table(readable, dimension)
    if hide_small:
        table = table[~table["too_small"]]
    view.rate_section(table, overall, dimension)

    view.section("First-year progress by final status", "Academic progress")
    measure = view.progress_controls()
    left, right = view.two_columns()
    with left:
        view.status_section(analytics.status_distribution(data))
    with right:
        view.progress_section(
            analytics.progress_by_status(data, measure),
            measure,
            analytics.median_by_status(data, measure),
        )
