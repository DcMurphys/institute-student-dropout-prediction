"""Dashboard analytics reproduce the notebook's rate tables."""

import pytest

from app.models import student_analytics as analytics


def test_overall_dropout_rate(students):
    assert analytics.dropout_rate(students) == pytest.approx(0.3212, abs=1e-4)


def test_tuition_arrears_rate_and_lift(students):
    table = analytics.dropout_rate_table(students, "Tuition_fees_up_to_date")
    arrears = table.set_index("group").loc[0]
    assert arrears["rate"] == pytest.approx(0.866, abs=1e-3)
    assert arrears["lift"] == pytest.approx(2.69, abs=1e-2)


def test_small_groups_are_flagged(students):
    table = analytics.dropout_rate_table(students, "Course")
    assert (table["too_small"] == (table["students"] < 50)).all()
    assert not table.iloc[0]["too_small"]  # large groups are listed first


def test_filters_keep_matching_rows(students):
    filtered = analytics.apply_filters(
        students, {"Scholarship_holder": [1], "Course": []}
    )
    assert len(filtered) == students["Scholarship_holder"].eq(1).sum()


def test_highlights_are_sorted_by_lift(students):
    items = analytics.top_highlights(analytics.add_age_band(students))
    assert [h.lift for h in items] == sorted((h.lift for h in items), reverse=True)
    assert items[0].dimension == "Tuition_fees_up_to_date"
