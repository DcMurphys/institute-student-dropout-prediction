"""Early-warning signals to discuss with a student, with the responsible office.

These are rule-based flags taken from the notebook's insights and
recommendations (Insights 2, 5, 9 and Section 9.5). They give advisors a
starting point for the conversation; they are not an explanation of the
model's internal reasoning.

The dropout rates quoted in each signal are computed from the student records
(``signal_rates``) with the same rule that flags the student, so the figures
stay in line with the data the dashboard shows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from app.config import POSITIVE_CLASS, TARGET
from app.models.feature_engineering import SEMESTER_PREFIXES

LOW_APPROVAL_RATE = 0.5  # notebook Section 9.5, recommendation 1
MATURE_ENTRY_AGE = 25

ARREARS = "arrears"
DEBTOR = "debtor"
LOW_APPROVAL = "low_approval"
MATURE = "mature"
EVENING = "evening"


def _low_approval_semesters(data: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: semester with enrolled units and an approval rate below 50%."""
    return pd.DataFrame(
        {
            short: data[f"{prefix}_enrolled"].gt(0)
            & data[f"{short}_approval_rate"].lt(LOW_APPROVAL_RATE)
            for short, prefix in SEMESTER_PREFIXES.items()
        },
        index=data.index,
    )


# Rule per signal: rows (students) the signal applies to. Expects the raw inputs
# plus the engineered approval rates (feature_engineering.add_engineered_features).
SIGNAL_RULES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    ARREARS: lambda d: d["Tuition_fees_up_to_date"].eq(0),
    DEBTOR: lambda d: d["Debtor"].eq(1),
    LOW_APPROVAL: lambda d: _low_approval_semesters(d).any(axis=1),
    MATURE: lambda d: d["Age_at_enrollment"].ge(MATURE_ENTRY_AGE),
    EVENING: lambda d: d["Daytime_evening_attendance"].eq(0),
}


@dataclass(frozen=True)
class GroupRate:
    """Dropout rate of the students a rule flags, and of everyone else."""

    flagged: float
    others: float


@dataclass(frozen=True)
class SignalRates:
    """Overall dropout rate and the rate per signal group."""

    overall: float
    groups: dict[str, GroupRate]

    def lift(self, key: str) -> float:
        """How many times the group's rate exceeds the overall rate."""
        return self.groups[key].flagged / self.overall


def signal_rates(data: pd.DataFrame) -> SignalRates:
    """Compute the dropout rates quoted by the signals.

    Args:
        data: Student records with ``Status`` and the engineered features.

    Returns:
        Overall rate and, per rule in ``SIGNAL_RULES``, the rate of flagged
        and non-flagged students.
    """
    dropped = data[TARGET].eq(POSITIVE_CLASS)
    groups = {}
    for key, rule in SIGNAL_RULES.items():
        mask = rule(data)
        groups[key] = GroupRate(
            flagged=float(dropped[mask].mean()), others=float(dropped[~mask].mean())
        )
    return SignalRates(overall=float(dropped.mean()), groups=groups)


@dataclass(frozen=True)
class SupportSignal:
    """One observable signal and the office that can act on it."""

    title: str
    detail: str
    owner: str
    icon: str


def _low_approval_detail(student: pd.DataFrame, rate: GroupRate) -> str:
    flags = _low_approval_semesters(student).iloc[0]
    passed = " and ".join(
        f"{student[f'{short}_approval_rate'].iloc[0]:.0%} in semester {number}"
        for number, short in enumerate(SEMESTER_PREFIXES, start=1)
        if flags[short]
    )
    return (
        f"Passed {passed} of enrolled units. Students who passed fewer than half "
        f"of their units in a semester dropped out at {rate.flagged:.1%} vs. "
        f"{rate.others:.1%} for other students; this is the strongest academic "
        "warning sign. Offer tutoring and a study-plan review."
    )


def detect_signals(student: pd.Series, rates: SignalRates) -> list[SupportSignal]:
    """Return the signals present for one student (engineered features included).

    Args:
        student: One row with raw inputs and engineered features.
        rates: Dropout rates from ``signal_rates`` quoted in each signal.
    """
    record = student.to_frame().T.infer_objects()
    present = {key for key, rule in SIGNAL_RULES.items() if rule(record).iloc[0]}
    group = rates.groups
    signals = []
    if ARREARS in present:
        signals.append(
            SupportSignal(
                "Tuition fees in arrears",
                f"Students with fees in arrears dropped out at "
                f"{group[ARREARS].flagged:.1%}, about {rates.lift(ARREARS):.1f} "
                "times the average. Discuss a payment plan or emergency grant.",
                "Student Financial Services",
                ":material/payments:",
            )
        )
    if DEBTOR in present:
        signals.append(
            SupportSignal(
                "Outstanding debt recorded",
                f"Debtors dropped out at {group[DEBTOR].flagged:.1%} vs. "
                f"{group[DEBTOR].others:.1%} for other students.",
                "Student Financial Services",
                ":material/account_balance_wallet:",
            )
        )
    if LOW_APPROVAL in present:
        signals.append(
            SupportSignal(
                "Low share of units passed",
                _low_approval_detail(record, group[LOW_APPROVAL]),
                "Academic advisors / tutors",
                ":material/menu_book:",
            )
        )
    if MATURE in present:
        signals.append(
            SupportSignal(
                "Mature entrant",
                f"Students who enrolled at {MATURE_ENTRY_AGE} or older dropped out "
                f"at {group[MATURE].flagged:.1%} vs. {group[MATURE].others:.1%} for "
                "younger students. Flexible scheduling and mentoring help balance "
                "study with work.",
                "Admissions office & programme directors",
                ":material/schedule:",
            )
        )
    if EVENING in present:
        signals.append(
            SupportSignal(
                "Evening attendance",
                f"Evening students dropped out at {group[EVENING].flagged:.1%} vs. "
                f"{group[EVENING].others:.1%} for daytime students.",
                "Admissions office & programme directors",
                ":material/dark_mode:",
            )
        )
    return signals
