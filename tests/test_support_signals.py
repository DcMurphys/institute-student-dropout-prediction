"""Support signals quote rates computed from the records and flag the right students."""

import pytest

from app.models import support_signals as signals


@pytest.fixture(scope="module")
def rates(students):
    return signals.signal_rates(students)


def test_rates_reproduce_notebook_eda(rates):
    groups = rates.groups
    assert rates.overall == pytest.approx(0.321, abs=1e-3)
    assert groups[signals.ARREARS].flagged == pytest.approx(0.866, abs=1e-3)
    assert groups[signals.ARREARS].others == pytest.approx(0.247, abs=1e-3)
    assert rates.lift(signals.ARREARS) == pytest.approx(2.69, abs=1e-2)
    assert groups[signals.DEBTOR].flagged == pytest.approx(0.620, abs=1e-3)
    assert groups[signals.DEBTOR].others == pytest.approx(0.283, abs=1e-3)
    assert groups[signals.EVENING].flagged == pytest.approx(0.429, abs=1e-3)
    assert groups[signals.EVENING].others == pytest.approx(0.308, abs=1e-3)
    assert groups[signals.MATURE].flagged > 0.5


def test_every_rule_has_a_rate(rates):
    assert set(rates.groups) == set(signals.SIGNAL_RULES)


def test_flagged_student_gets_matching_signals(students, rates):
    student = students[
        students["Tuition_fees_up_to_date"].eq(0)
        & students["Debtor"].eq(0)
        & students["Daytime_evening_attendance"].eq(1)
    ].iloc[0]
    titles = [s.title for s in signals.detect_signals(student, rates)]
    assert "Tuition fees in arrears" in titles
    assert "Outstanding debt recorded" not in titles
    assert "Evening attendance" not in titles


def test_signal_text_uses_computed_rates(students, rates):
    student = students[students["Debtor"].eq(1)].iloc[0]
    debt = next(
        s
        for s in signals.detect_signals(student, rates)
        if s.title == "Outstanding debt recorded"
    )
    assert f"{rates.groups[signals.DEBTOR].flagged:.1%}" in debt.detail


def test_low_approval_detail_names_semesters(students, rates):
    student = students[
        students["Curricular_units_1st_sem_enrolled"].gt(0)
        & students["Sem1_approval_rate"].lt(0.5)
        & students["Curricular_units_2nd_sem_enrolled"].gt(0)
        & students["Sem2_approval_rate"].ge(0.5)
    ].iloc[0]
    low = next(
        s
        for s in signals.detect_signals(student, rates)
        if s.title == "Low share of units passed"
    )
    assert "in semester 1" in low.detail
    assert "in semester 2" not in low.detail
