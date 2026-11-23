"""The app must rebuild exactly the features the notebook engineered."""

import numpy as np
import pandas as pd
import pytest

from app.models.feature_engineering import (
    ENGINEERED_FEATURES,
    add_engineered_features,
    apply_log_transform,
    safe_ratio,
)


def test_safe_ratio_returns_zero_for_zero_denominator():
    assert safe_ratio(pd.Series([3, 1]), pd.Series([6, 0])).tolist() == [0.5, 0.0]


@pytest.mark.parametrize("column", ENGINEERED_FEATURES)
def test_matches_notebook_export(students, raw_students, column):
    rebuilt = add_engineered_features(raw_students)
    np.testing.assert_allclose(rebuilt[column], students[column], atol=1e-9)


def test_does_not_modify_input(raw_students):
    before = raw_students.copy()
    add_engineered_features(raw_students)
    pd.testing.assert_frame_equal(raw_students, before)


def test_log_transform_skips_missing_columns():
    data = pd.DataFrame({"a": [0.0, np.e - 1]})
    result = apply_log_transform(data, ["a", "not_there"])
    np.testing.assert_allclose(result["a"], [0.0, 1.0])
