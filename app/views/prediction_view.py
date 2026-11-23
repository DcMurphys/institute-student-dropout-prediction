"""Layout of the Student Check and Batch Screening pages (notebook Section 8.2)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from app.config import MAX_UPLOAD_ROWS, STATUS_ORDER
from app.models.support_signals import SupportSignal
from app.views.components import charts
from app.views.components.kpi_cards import Kpi, kpi_row
from app.views.components.layout import (
    card,
    chart_card,
    empty_state,
    grid,
    page_header,
    reliability_panel,
    section_heading,
    status_badge_html,
    status_text,
)
from app.views.components.student_form import student_form
from app.views.display_names import STATUS_DISPLAY

SIGNAL_COLUMNS = 2  # signal cards per row


# ---------------------------------------------------------------- Student Check
def student_check_header() -> None:
    page_header(
        "Early warning · One student",
        "Student Check",
        "How likely is this student to drop out, and what could an advisor talk "
        "about first? Enter the student's records at the end of the first year.",
    )


def preset_picker(presets: list[str]) -> tuple[str, bool]:
    """Preset selector; returns the preset and whether "Fill form" was pressed."""
    left, right = grid([3, 1], vertical_alignment="bottom")
    choice = left.selectbox(
        "Start from",
        presets,
        key="preset_choice",
        help="Fill the form with a typical student or a real example, then adjust.",
    )
    pressed = right.button("Fill form", icon=":material/edit_note:", width="stretch")
    return choice, pressed


def form(
    category_options: dict[str, list[int]],
    format_option: Callable[[str, int], str],
    input_columns: list[str],
) -> dict | None:
    return student_form(category_options, format_option, input_columns)


def form_errors(messages: list[str]) -> None:
    st.error(
        "Please correct the form before checking:\n\n"
        + "\n".join(f"- {message}" for message in messages),
        icon=":material/error:",
    )


def student_result(
    prediction: pd.Series,
    threshold: float,
    signals: list[SupportSignal],
    performance: dict,
) -> None:
    """Verdict, probabilities, signals to discuss and reliability."""
    status = prediction["Predicted status"]
    dropout_probability = prediction["Dropout probability"]
    section_heading("Result", "Prediction")
    left, right = grid(2)
    with left, card():
        st.html(status_badge_html(status))
        flagged = dropout_probability >= threshold
        st.html(
            f'<div class="mi-verdict">{dropout_probability:.0%} chance of '
            "dropping out</div>"
        )
        st.caption(
            f"Students are flagged from {threshold:.0%}. This student is "
            + ("**above** the flag line." if flagged else "**below** the flag line.")
        )
        probabilities = pd.DataFrame(
            {
                "Status": list(STATUS_ORDER),
                "Probability": [prediction[f"P({s})"] for s in STATUS_ORDER],
            }
        )
        st.altair_chart(charts.class_probability_chart(probabilities), width="stretch")
        st.caption(
            "**What this shows:** the model's estimate for each final status. "
            f'"{STATUS_DISPLAY["Enrolled"]["label"]}" is the hardest outcome to '
            "predict, so treat it with extra care."
        )
    with right:
        reliability_panel(performance)

    section_heading("Signals to discuss", "Next step")
    if not signals:
        st.success(
            "No early-warning signal in this record. Keep the regular check-ins.",
            icon=":material/check_circle:",
        )
        return
    for start in range(0, len(signals), SIGNAL_COLUMNS):
        row = signals[start : start + SIGNAL_COLUMNS]
        for column, signal in zip(grid(SIGNAL_COLUMNS), row, strict=False):
            with column, card():
                st.markdown(f"**{signal.icon} {signal.title}**")
                st.write(signal.detail)
                st.caption(f":material/support_agent: Who can help: {signal.owner}")
    st.caption(
        "Signals come from patterns in past student records. They are conversation "
        "starters, not the model's reasons for this prediction."
    )


def awaiting_result() -> None:
    empty_state(
        "No student checked yet",
        "Fill in the form (or start from an example) and press **Check this student**.",
    )


# ------------------------------------------------------------- Batch Screening
def batch_header() -> None:
    page_header(
        "Early warning · Many students",
        "Batch Screening",
        "Which students should advisors contact first? Upload the first-year records "
        "of a cohort and get a list ranked by dropout risk.",
    )


def upload_panel(template_csv: bytes, sample_csv: bytes) -> tuple[object, bool]:
    """Uploader with the template and a sample file; returns (file, use_sample)."""
    with st.container(border=True):
        st.markdown("**:material/upload_file: 1 · Upload student records**")
        st.caption(
            f"CSV file, one row per student, up to {MAX_UPLOAD_ROWS:,} rows. Use the "
            "template for the expected columns. A `Student_ID` column is kept in the "
            "results if present."
        )
        uploaded = st.file_uploader(
            "Student records (CSV)",
            type=["csv"],
            key="batch_file",
            label_visibility="collapsed",
        )
        a, b, c = grid(3)
        a.download_button(
            "Download template",
            template_csv,
            "student_records_template.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
        b.download_button(
            "Download sample file",
            sample_csv,
            "sample_student_records.csv",
            mime="text/csv",
            icon=":material/description:",
            width="stretch",
        )
        use_sample = c.button(
            "Try with sample file",
            icon=":material/science:",
            width="stretch",
            type="primary",
        )
    return uploaded, use_sample


def read_error(message: str) -> None:
    st.error(message, icon=":material/error:")


def missing_columns(columns: list[str]) -> None:
    st.error(
        "The file is missing columns the model needs: "
        + ", ".join(f"`{c}`" for c in columns)
        + ". Download the template to see the expected format.",
        icon=":material/error:",
    )


def validation_summary(total: int, valid: int, issues: pd.DataFrame) -> None:
    """Rows excluded or flagged during the checks."""
    st.markdown("**:material/fact_check: 2 · Check the records**")
    if issues.empty:
        st.success(
            f"All {total:,} rows passed the checks.", icon=":material/check_circle:"
        )
        return
    excluded = total - valid
    message = f"{valid:,} of {total:,} rows will be scored."
    if excluded:
        message += f" {excluded:,} rows have errors and are left out."
    st.warning(message, icon=":material/warning:")
    with st.expander(
        f"See the {len(issues):,} rows with issues", icon=":material/list:"
    ):
        st.dataframe(issues, hide_index=True, key="batch_issues")


def batch_results(
    results: pd.DataFrame, threshold: float, csv_bytes: bytes, id_column: str
) -> None:
    """KPIs, risk distribution and the ranked student list."""
    st.markdown("**:material/format_list_numbered: 3 · Review the ranked list**")
    flagged = results["Dropout probability"].ge(threshold)
    kpi_row(
        [
            Kpi(
                "Students screened", f"{len(results):,}", "Rows that passed the checks."
            ),
            Kpi(
                "Flagged for follow-up",
                f"{flagged.sum():,}",
                f"{flagged.mean():.0%} of screened students reach the "
                f"{threshold:.0%} flag line.",
            ),
            Kpi(
                "Highest dropout probability",
                f"{results['Dropout probability'].max():.0%}",
                f"Student {results.iloc[0][id_column]} is first on the list.",
            ),
        ]
    )
    chart_card(
        "Dropout probability across the cohort",
        lambda: st.altair_chart(
            charts.risk_distribution_chart(results, threshold), width="stretch"
        ),
        f"Students to the right of the dashed line ({threshold:.0%}) are flagged. "
        "Start with the highest probabilities.",
        results[[id_column, "Dropout probability"]],
        key="risk_hist",
    )
    only_flagged = st.toggle(
        "Show flagged students only", value=True, key="only_flagged"
    )
    table = results[flagged] if only_flagged else results
    display = table.assign(
        **{"Predicted status": table["Predicted status"].map(status_text)}
    )
    st.dataframe(
        display,
        hide_index=True,
        key="batch_table",
        column_config={
            "Dropout probability": st.column_config.ProgressColumn(
                "Dropout probability", format="percent", min_value=0.0, max_value=1.0
            ),
            **{
                f"P({s})": st.column_config.NumberColumn(
                    f"{s} probability", format="percent"
                )
                for s in STATUS_ORDER
            },
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
        },
    )
    st.download_button(
        "Download ranked list (CSV)",
        csv_bytes,
        "dropout_risk_ranking.csv",
        mime="text/csv",
        icon=":material/download:",
        type="primary",
    )


def awaiting_upload() -> None:
    empty_state(
        "No file uploaded yet",
        "Upload a CSV file or press **Try with sample file** to see how screening "
        "works.",
    )
