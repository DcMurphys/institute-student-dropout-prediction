"""Single-student input form for the Student Check page."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import streamlit as st

from app.models.feature_engineering import SEMESTER_PREFIXES
from app.views.components.layout import grid
from app.views.display_names import FIELD_HELP, SEMESTER_FIELD_LABELS, field_label

KEY_PREFIX = "form_"

PROFILE_FIELDS = ("Gender", "Age_at_enrollment", "Marital_status", "Displaced")
ADMISSION_FIELDS = (
    "Course",
    "Daytime_evening_attendance",
    "Application_order",
    "Previous_qualification",
    "Previous_qualification_grade",
    "Admission_grade",
)
FINANCE_FIELDS = ("Tuition_fees_up_to_date", "Debtor", "Scholarship_holder")
FORM_COLUMNS = 3  # default fields per row
# (title, fields, fields per row): each count divides its section evenly, so no
# row has an orphan field and every field in a row has the same width.
FORM_SECTIONS = (
    ("Student profile", PROFILE_FIELDS, 4),
    ("Admission", ADMISSION_FIELDS, FORM_COLUMNS),
    ("Finances", FINANCE_FIELDS, FORM_COLUMNS),
)
SEMESTER_SUFFIXES = tuple(SEMESTER_FIELD_LABELS)
SENSITIVE_FIELDS = ("Gender", "Marital_status")

FLOAT_FIELDS = {"Previous_qualification_grade", "Admission_grade"}
INT_LIMITS = {"Age_at_enrollment": (15, 80)}
FLOAT_LIMITS = {
    "Previous_qualification_grade": (0.0, 200.0),
    "Admission_grade": (0.0, 200.0),
}


def form_key(column: str) -> str:
    return f"{KEY_PREFIX}{column}"


def is_float_field(column: str) -> bool:
    """Grades are decimals; every other input is a whole number or a code."""
    return column in FLOAT_FIELDS or column.endswith("_grade")


def load_values(values: dict) -> None:
    """Put a profile into the form widgets (used by the preset selector)."""
    for column, value in values.items():
        st.session_state[form_key(column)] = (
            round(float(value), 1) if is_float_field(column) else int(value)
        )


def _category_input(
    column: str, container, options: list[int], format_option: Callable[[str, int], str]
) -> None:
    help_text = FIELD_HELP.get(column)
    if column in SENSITIVE_FIELDS:
        help_text = (
            "Sensitive attribute: it is used by the model, but its use in individual "
            "decisions still needs a policy decision by Academic Affairs."
        )
    if len(options) == 2:
        container.segmented_control(
            field_label(column),
            options,
            format_func=lambda code: format_option(column, code),
            key=form_key(column),
            help=help_text,
            selection_mode="single",
            required=True,
            width="stretch",
        )
    else:
        container.selectbox(
            field_label(column),
            options,
            format_func=lambda code: format_option(column, code),
            key=form_key(column),
            help=help_text,
        )


def _number_input(column: str, container) -> None:
    if column in FLOAT_FIELDS:
        low, high = FLOAT_LIMITS[column]
        container.number_input(
            field_label(column),
            min_value=low,
            max_value=high,
            step=0.1,
            format="%.1f",
            key=form_key(column),
        )
    else:
        low, high = INT_LIMITS[column]
        container.number_input(
            field_label(column),
            min_value=low,
            max_value=high,
            step=1,
            key=form_key(column),
        )


def _rows(fields: Sequence[str], columns: int):
    """Yield ``(slot, field)`` pairs, starting a new grid row every ``columns``.

    A new row per group keeps widgets of one row on the same baseline even when
    their labels differ in height.
    """
    for start in range(0, len(fields), columns):
        chunk = fields[start : start + columns]
        yield from zip(grid(columns), chunk, strict=False)


def _field_input(
    column: str,
    slot,
    category_options: dict[str, list[int]],
    format_option: Callable[[str, int], str],
) -> None:
    if column in category_options:
        _category_input(column, slot, category_options[column], format_option)
    else:
        _number_input(column, slot)


def _semester_inputs(prefix: str, title: str) -> None:
    st.markdown(f"**{title}**")
    for slot, suffix in _rows(SEMESTER_SUFFIXES, FORM_COLUMNS):
        column = f"{prefix}_{suffix}"
        if suffix == "grade":
            slot.number_input(
                SEMESTER_FIELD_LABELS[suffix],
                min_value=0.0,
                max_value=20.0,
                step=0.1,
                format="%.1f",
                key=form_key(column),
            )
        else:
            slot.number_input(
                SEMESTER_FIELD_LABELS[suffix],
                min_value=0,
                max_value=60,
                step=1,
                key=form_key(column),
                help=FIELD_HELP.get(suffix),
            )


def student_form(
    category_options: dict[str, list[int]],
    format_option: Callable[[str, int], str],
    input_columns: list[str],
) -> dict | None:
    """Render the form; return the entered values when the form is submitted.

    Widget values live in ``st.session_state`` (keys ``form_<column>``), so a
    preset loaded before rendering fills every field.
    """
    with st.form("student_form", border=True):
        tab_profile, tab_semesters = st.tabs(
            [
                ":material/badge: Profile, admission & finances",
                ":material/menu_book: First-year results",
            ]
        )
        with tab_profile:
            for title, fields, per_row in FORM_SECTIONS:
                st.markdown(f"**{title}**")
                for slot, column in _rows(fields, per_row):
                    _field_input(column, slot, category_options, format_option)
        with tab_semesters:
            st.caption(
                "Predictions are made at the end of the first academic year, so both "
                "semesters are needed."
            )
            for prefix, title in zip(
                SEMESTER_PREFIXES.values(), ("Semester 1", "Semester 2"), strict=True
            ):
                _semester_inputs(prefix, title)
        submitted = st.form_submit_button(
            "Check this student", type="primary", icon=":material/person_search:"
        )
    if not submitted:
        return None
    return {column: st.session_state.get(form_key(column)) for column in input_columns}
