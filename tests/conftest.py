"""Shared fixtures (import paths come from ``pythonpath`` in pyproject.toml)."""

import pandas as pd
import pytest

from app import config
from app.models.dropout_predictor import ArtifactsMissingError, DropoutPredictor
from app.models.feature_engineering import ENGINEERED_FEATURES


@pytest.fixture(scope="session")
def students() -> pd.DataFrame:
    """Dashboard dataset exported by the notebook (with engineered features)."""
    if not config.STUDENT_DATA_PATH.exists():
        pytest.skip("data/processed/student.csv has not been exported yet")
    return pd.read_csv(config.STUDENT_DATA_PATH)


@pytest.fixture(scope="session")
def raw_students(students) -> pd.DataFrame:
    """Same students without the engineered columns (as a user would upload)."""
    return students.drop(columns=list(ENGINEERED_FEATURES))


@pytest.fixture(scope="session")
def predictor() -> DropoutPredictor:
    try:
        return DropoutPredictor.load(
            config.PIPELINE_PATH, config.TARGET_ENCODER_PATH, config.METADATA_PATH
        )
    except ArtifactsMissingError as error:
        pytest.skip(str(error))
