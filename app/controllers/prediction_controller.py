"""Student Check and Batch Screening pages: input → validation → prediction."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from app.config import MAX_UPLOAD_ROWS
from app.controllers import resources
from app.models import student_analytics as analytics
from app.models.dropout_predictor import DROPOUT_PROBABILITY, PREDICTED_STATUS
from app.models.feature_engineering import add_engineered_features
from app.models.schemas import CATEGORICAL_INPUTS, find_id_column, validate_records
from app.models.support_signals import detect_signals
from app.views import prediction_view as view
from app.views.components.student_form import load_values

TYPICAL_PRESET = "Typical student (most common values)"
DEFAULT_ID_COLUMN = "Student_ID"
SAMPLE_ROWS = 60
SAMPLE_SEED = 7


# ---------------------------------------------------------------- Student Check
def _presets() -> dict[str, dict]:
    columns = resources.predictor().input_columns
    data = resources.students()
    return {
        TYPICAL_PRESET: analytics.typical_profile(data, columns, CATEGORICAL_INPUTS),
        **analytics.example_profiles(data, columns),
    }


def show_student_check_page() -> None:
    predictor = resources.predictor()
    view.student_check_header()

    presets = _presets()
    if "form_initialised" not in st.session_state:
        load_values(presets[TYPICAL_PRESET])
        st.session_state.form_initialised = True
    choice, pressed = view.preset_picker(list(presets))
    if pressed:
        load_values(presets[choice])
        st.session_state.pop("student_result", None)

    values = view.form(
        predictor.category_options, resources.format_option, predictor.input_columns
    )
    if values is not None:
        record = pd.DataFrame([values])
        report = validate_records(
            record, predictor.input_columns, predictor.category_options
        )
        errors = report.errors.iloc[0]
        if errors:
            view.form_errors(errors)
            st.session_state.pop("student_result", None)
        else:
            prediction = predictor.predict(record).iloc[0]
            engineered = add_engineered_features(record.apply(pd.to_numeric)).iloc[0]
            st.session_state.student_result = (
                prediction,
                detect_signals(engineered, resources.signal_rates()),
            )

    result = st.session_state.get("student_result")
    if result is None:
        view.awaiting_result()
        return
    prediction, signals = result
    view.student_result(
        prediction, predictor.threshold, signals, resources.model_performance()
    )


# ------------------------------------------------------------- Batch Screening
@st.cache_data(show_spinner=False)
def _template_csv(columns: tuple[str, ...]) -> bytes:
    return (
        pd.DataFrame(columns=[DEFAULT_ID_COLUMN, *columns]).to_csv(index=False).encode()
    )


@st.cache_data(show_spinner=False)
def _sample_csv(columns: tuple[str, ...]) -> bytes:
    """Random past students without their outcome, with made-up IDs."""
    sample = resources.students().sample(SAMPLE_ROWS, random_state=SAMPLE_SEED)
    sample = sample[list(columns)].reset_index(drop=True)
    sample.insert(
        0, DEFAULT_ID_COLUMN, [f"S{n:04d}" for n in range(1, len(sample) + 1)]
    )
    return sample.to_csv(index=False).encode()


def _read_csv(content: bytes) -> pd.DataFrame:
    """Read comma- or semicolon-separated files."""
    return pd.read_csv(io.BytesIO(content), sep=None, engine="python")


def _issues_table(records: pd.DataFrame, report, id_column: str) -> pd.DataFrame:
    rows = []
    for index in records.index:
        for kind, messages in (
            ("Error", report.errors[index]),
            ("Warning", report.warnings[index]),
        ):
            rows.extend(
                {
                    "Row": index + 1,
                    id_column: records.loc[index, id_column],
                    "Type": kind,
                    "Issue": m,
                }
                for m in messages
            )
    return pd.DataFrame(rows, columns=["Row", id_column, "Type", "Issue"])


def show_batch_screening_page() -> None:
    predictor = resources.predictor()
    columns = tuple(predictor.input_columns)
    view.batch_header()

    sample_csv = _sample_csv(columns)
    uploaded, use_sample = view.upload_panel(_template_csv(columns), sample_csv)
    if use_sample:
        st.session_state.batch_source = ("sample_student_records.csv", sample_csv)
    if uploaded is not None:
        st.session_state.batch_source = (uploaded.name, uploaded.getvalue())
    source = st.session_state.get("batch_source")
    if source is None:
        view.awaiting_upload()
        return

    try:
        records = _read_csv(source[1]).reset_index(drop=True)
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError):
        view.read_error(
            "The file could not be read as CSV. Save it as CSV (UTF-8) and try again."
        )
        return
    if len(records) > MAX_UPLOAD_ROWS:
        view.read_error(
            f"The file has {len(records):,} rows; the limit is {MAX_UPLOAD_ROWS:,}."
        )
        return

    id_column = find_id_column(records.columns)
    if id_column is None:
        id_column = DEFAULT_ID_COLUMN
        records.insert(0, id_column, [f"Row {n}" for n in range(1, len(records) + 1)])

    report = validate_records(records, list(columns), predictor.category_options)
    if not report.is_usable:
        view.missing_columns(report.missing_columns)
        return
    valid = records[report.valid_mask]
    view.validation_summary(
        len(records), len(valid), _issues_table(records, report, id_column)
    )
    if valid.empty:
        return

    scores = predictor.predict(valid)
    dropout_column = f"P({predictor.metadata.decision_rule.dropout_class})"
    probability_columns = [
        c for c in scores.columns if c.startswith("P(") and c != dropout_column
    ]
    results = (
        pd.concat(
            [
                valid[[id_column]],
                scores[[DROPOUT_PROBABILITY, PREDICTED_STATUS, *probability_columns]],
            ],
            axis=1,
        )
        .sort_values("Dropout probability", ascending=False)
        .reset_index(drop=True)
    )
    results.insert(0, "Rank", range(1, len(results) + 1))
    view.batch_results(
        results, predictor.threshold, results.to_csv(index=False).encode(), id_column
    )
