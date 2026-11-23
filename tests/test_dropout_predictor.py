"""Decision rule, prediction output and input validation."""

import numpy as np
import pandas as pd

from app.models.dropout_predictor import (
    DROPOUT_PROBABILITY,
    PREDICTED_STATUS,
    DecisionRule,
    apply_decision_rule,
)
from app.models.schemas import validate_records

LABELS = ["Dropout", "Enrolled", "Graduate"]
RULE = DecisionRule(dropout_class="Dropout", threshold=0.37)


def test_threshold_flags_dropout_below_argmax():
    probabilities = np.array(
        [[0.40, 0.10, 0.50], [0.30, 0.20, 0.50], [0.30, 0.45, 0.25]]
    )
    assert apply_decision_rule(probabilities, LABELS, RULE).tolist() == [
        "Dropout",
        "Graduate",
        "Enrolled",
    ]


def test_threshold_is_inclusive():
    probabilities = np.array([[0.37, 0.30, 0.33]])
    assert apply_decision_rule(probabilities, LABELS, RULE).tolist() == ["Dropout"]


def test_metadata_threshold_is_the_tuned_value(predictor):
    assert predictor.metadata.decision_rule.threshold == 0.37
    assert predictor.metadata.class_labels == LABELS


def test_predict_output(predictor, raw_students):
    sample = raw_students.head(50)
    result = predictor.predict(sample)
    assert list(result.index) == list(sample.index)
    probability_columns = [f"P({label})" for label in LABELS]
    np.testing.assert_allclose(result[probability_columns].sum(axis=1), 1.0)
    assert result[PREDICTED_STATUS].isin(LABELS).all()
    flagged = result[DROPOUT_PROBABILITY] >= 0.37
    assert (result.loc[flagged, PREDICTED_STATUS] == "Dropout").all()


def test_threshold_beats_argmax_on_dropout_recall(predictor, raw_students, students):
    """The tuned rule exists to catch more dropouts than the default argmax."""
    result = predictor.predict(raw_students)
    actual = students["Status"].eq("Dropout")
    argmax = result[[f"P({label})" for label in LABELS]].to_numpy().argmax(axis=1)
    recall_rule = (result[PREDICTED_STATUS].eq("Dropout") & actual).sum() / actual.sum()
    recall_argmax = ((argmax == 0) & actual).sum() / actual.sum()
    assert recall_rule >= recall_argmax


def test_real_records_pass_validation(predictor, raw_students):
    report = validate_records(
        raw_students, predictor.input_columns, predictor.category_options
    )
    assert report.is_usable
    assert report.valid_mask.all()


def test_validation_reports_missing_columns(predictor, raw_students):
    report = validate_records(
        raw_students.drop(columns=["Course"]),
        predictor.input_columns,
        predictor.category_options,
    )
    assert report.missing_columns == ["Course"]


def test_validation_catches_inconsistent_units(predictor, raw_students):
    record = raw_students.head(1).copy()
    record["Curricular_units_1st_sem_approved"] = (
        record["Curricular_units_1st_sem_enrolled"] + 1
    )
    record["Age_at_enrollment"] = 5
    report = validate_records(
        record, predictor.input_columns, predictor.category_options
    )
    messages = " ".join(report.errors.iloc[0])
    assert "approved units exceed enrolled units" in messages
    assert "Age_at_enrollment" in messages


def test_unknown_category_is_a_warning_not_an_error(predictor, raw_students):
    record = raw_students.head(1).copy()
    record["Course"] = 1234
    report = validate_records(
        record, predictor.input_columns, predictor.category_options
    )
    assert report.valid_mask.all()
    assert any("Course" in m for m in report.warnings.iloc[0])
    assert pd.notna(predictor.predict(record)[DROPOUT_PROBABILITY].iloc[0])
