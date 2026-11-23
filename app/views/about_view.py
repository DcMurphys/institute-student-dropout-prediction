"""Layout of the About the Model page: reliability, method, limits, glossary."""

from __future__ import annotations

import streamlit as st

from app.views.components.layout import (
    card,
    grid,
    page_header,
    reliability_panel,
    section_heading,
)

HOW_IT_WORKS = (
    (
        ":material/edit_note:",
        "Records in",
        "Admission details, finances and first-year "
        "results per semester (units enrolled, passed and grades).",
    ),
    (
        ":material/calculate:",
        "Progress measures",
        "The app derives approval rates and "
        "the first-year average grade, exactly as in the analysis.",
    ),
    (
        ":material/forest:",
        "Random Forest",
        "Hundreds of decision trees, trained on past "
        "students, each vote for Dropout, Enrolled or Graduate.",
    ),
    (
        ":material/flag:",
        "Flag line",
        "A student is flagged when the dropout probability "
        "reaches the flag line, chosen to catch at least 3 in 4 dropouts.",
    ),
)

LIMITATIONS = (
    "**Proxy data.** The records come from a Portuguese polytechnic (2008/09-2018/19) "
    "and stand in for institutional records. Re-check the model on the polytechnic's "
    "own students.",
    "**Timing.** Predictions need first-year results, so they are made at the end of "
    "the first year, not at enrollment.",
    "**Fee status.** Tuition status is a strong signal. If it is recorded after a "
    "student has already disengaged, the model looks better than it is.",
    "**No dropout reason.** The data does not say whether a student withdrew or was "
    "dismissed.",
    "**Patterns, not causes.** Rates and signals describe what happened to past "
    "students; they do not prove why.",
    "**Sensitive attributes.** Gender and marital status are used by the model; their "
    "use in individual decisions needs a policy decision by Academic Affairs.",
)

GLOSSARY = {
    "Dropout rate": "Share of students whose final status is Dropout.",
    "Times the average": "A group's dropout rate divided by the average rate. 2.0× "
    "means twice as likely to drop out as the average student.",
    "Approval rate": "Units passed divided by units enrolled in a semester.",
    "Dropout probability": "The model's estimate (0-100%) that a student will "
    "drop out.",
    "Flag line": "The probability from which a student is listed for follow-up.",
    "Still enrolled": "Not finished at the end of the normal course duration "
    "(three years; four for Nursing).",
}


def render(performance: dict, threshold: float) -> None:
    page_header(
        "Help · Method",
        "About the Model",
        "What the early-warning check does, how well it works and where to be careful.",
    )
    section_heading("How a check works", "Method")
    left, right = grid(2)
    with left, card():
        st.markdown("**:material/route: From records to a flag**")
        for icon, title, text in HOW_IT_WORKS:
            st.markdown(f"**{icon} {title}** — {text}")
        st.caption(f"Current flag line: {threshold:.0%} dropout probability.")
    with right:
        reliability_panel(performance)

    section_heading("Use with care", "Limitations")
    for item in LIMITATIONS:
        st.markdown(f"- {item}")

    section_heading("Glossary", "Terms")
    for term, meaning in GLOSSARY.items():
        st.markdown(f"**{term}** — {meaning}")
