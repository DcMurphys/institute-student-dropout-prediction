"""Cached, read-only access to data and model artifacts (shared by all visitors)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import config
from app.models import student_analytics, student_repository, support_signals
from app.models.dropout_predictor import DropoutPredictor


@st.cache_resource(show_spinner="Loading the early-warning model…")
def predictor() -> DropoutPredictor:
    """Pipeline, target classes and decision rule."""
    return DropoutPredictor.load(
        config.PIPELINE_PATH, config.TARGET_ENCODER_PATH, config.METADATA_PATH
    )


@st.cache_data(show_spinner="Loading student records…")
def students() -> pd.DataFrame:
    """Dashboard dataset (codes, original units) with an age band column."""
    data = student_repository.load_students(config.STUDENT_DATA_PATH)
    return student_analytics.add_age_band(data)


@st.cache_data(show_spinner=False)
def signal_rates() -> support_signals.SignalRates:
    """Dropout rates quoted by the Student Check signals."""
    return support_signals.signal_rates(students())


@st.cache_data(show_spinner=False)
def label_mappings() -> dict[str, dict[int, str]]:
    return student_repository.load_label_mappings(config.LABEL_MAPPINGS_PATH)


@st.cache_data(show_spinner=False)
def model_performance() -> dict:
    return student_repository.load_json(config.MODEL_PERFORMANCE_PATH)


def format_option(column: str, value: object) -> str:
    """Readable label for a code; other values (labels, bands) pass through."""
    if isinstance(value, (int, np.integer)):
        return student_repository.code_label(int(value), column, label_mappings())
    return str(value)
