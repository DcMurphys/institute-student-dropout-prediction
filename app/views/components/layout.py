"""Page scaffolding: grid, cards, global CSS, headers, badges and footer.

Symmetry rules used on every page:
- every row of cards or controls is built with ``grid`` (equal columns, one gap);
- every card in a row uses ``card`` (bordered, as tall as the tallest card);
- a row keeps its column count even when it has fewer items, so card widths
  stay identical between rows of the same kind.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
import streamlit as st

from app.config import PALETTE
from app.views.display_names import STATUS_DISPLAY

GRID_GAP = "medium"  # one gap for every row, so card edges line up across rows
CONTROL_HEIGHT = "2.5rem"  # inputs, selects, buttons and button groups

# MIT-inspired typography: one grotesque family with tight headline leading
# (85-100% of the size) and comfortable body leading; Roboto Mono for small
# uppercase labels. Colours come from the Murphys design system.
GLOBAL_CSS = f"""
<style>
  h1, h2, h3 {{ letter-spacing: -0.02em; }}
  h1 {{ line-height: 1.0 !important; }}
  h2, h3 {{ line-height: 1.1 !important; }}
  [data-testid="stMetricValue"], [data-testid="stDataFrame"], .tabular {{
    font-variant-numeric: tabular-nums;
  }}
  *:focus-visible {{
    outline: 2px solid {PALETTE["brand-dark"]} !important;
    outline-offset: 2px;
  }}
  .mi-eyebrow {{
    font-family: "Roboto Mono", monospace; font-size: 0.75rem; font-weight: 500;
    letter-spacing: 0.12em; text-transform: uppercase; color: {PALETTE["brand-dark"]};
    margin-bottom: -0.5rem;
  }}
  .mi-subtitle {{ color: {PALETTE["slate"]}; font-size: 1.05rem; line-height: 1.35;
    max-width: 52rem; margin-top: -0.25rem; }}
  .mi-badge {{
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600; line-height: 1.4; white-space: nowrap;
  }}
  .mi-badge.dropout {{ background: {PALETTE["status-dropout"]}; color: #fff; }}
  .mi-badge.enrolled {{
    background: {PALETTE["status-enrolled"]}; color: {PALETTE["ink"]};
  }}
  .mi-badge.graduate {{ background: {PALETTE["status-graduate"]}; color: #fff; }}
  .mi-badge.neutral {{ background: {PALETTE["border"]}; color: {PALETTE["ink"]}; }}
  .mi-verdict {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em;
    line-height: 1.05; margin: 0.4rem 0 0.25rem; }}
  [data-testid="stButtonGroup"] button,
  [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"],
  [data-testid="stBaseButton-primaryFormSubmit"] {{
    min-height: {CONTROL_HEIGHT};
  }}
  /* Button groups: equal segments, and a label as tall as other widget labels
     (the help icon otherwise makes it taller and pushes the buttons down). */
  [data-testid="stButtonGroup"] button {{ flex: 1 1 0; }}
  [data-testid="stButtonGroup"] [data-testid="stWidgetLabel"] {{
    height: 1.5rem; align-items: center;
  }}
  [data-testid="stButtonGroup"] [data-testid="stWidgetLabel"] > span {{
    max-height: 1.5rem; align-items: center;
  }}
  .mi-card-title {{
    font-weight: 600; line-height: 1.3; min-height: 2.6em; margin: 0.15rem 0 0;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
  }}
  .mi-footer {{ color: {PALETTE["slate"]}; font-size: 0.75rem; margin-top: 2.5rem;
    border-top: 1px solid {PALETTE["border"]}; padding-top: 0.75rem; }}
</style>
"""


def inject_css() -> None:
    """Apply the global style additions."""
    st.html(GLOBAL_CSS)


def grid(spec: int | Sequence[float], vertical_alignment: str = "top") -> list:
    """Columns with the shared gap; pass a count for equal-width columns."""
    return st.columns(spec, gap=GRID_GAP, vertical_alignment=vertical_alignment)


def card():
    """Bordered card that stretches to the height of the tallest card in its row."""
    return st.container(border=True, height="stretch")


def card_title(text: str) -> None:
    """Card title reserving two lines, so content below aligns across cards."""
    st.html(f'<div class="mi-card-title">{text}</div>')


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    """Eyebrow label, page title and a one-line question the page answers."""
    st.html(f'<div class="mi-eyebrow">{eyebrow}</div>')
    st.title(title)
    st.html(f'<p class="mi-subtitle">{subtitle}</p>')


def section_heading(title: str, eyebrow: str | None = None) -> None:
    """Section title with an optional eyebrow label."""
    if eyebrow:
        st.html(f'<div class="mi-eyebrow">{eyebrow}</div>')
    st.subheader(title)


def status_badge_html(status: str) -> str:
    """Badge for a student status: icon + label + colour."""
    style = STATUS_DISPLAY.get(status)
    if style is None:
        return f'<span class="mi-badge neutral">{status}</span>'
    return (
        f'<span class="mi-badge {style["css"]}">{style["icon"]} {style["label"]}</span>'
    )


def status_text(status: str) -> str:
    """Plain-text status for tables (icon + label)."""
    style = STATUS_DISPLAY.get(status)
    return f"{style['icon']} {style['label']}" if style else status


def callout(message: str, icon: str = ":material/lightbulb:") -> None:
    """Key message of a page, in the info style."""
    st.info(message, icon=icon)


def chart_card(
    title: str,
    render_chart: Callable[[], None],
    what_this_shows: str,
    data: pd.DataFrame,
    key: str,
) -> None:
    """Bordered card: title, chart, "What this shows" caption and a data table.

    Every chart has a plain-language caption and its underlying numbers, so
    readers who cannot see colours or charts get the same information.
    """
    with st.container(border=True, height="stretch", vertical_alignment="distribute"):
        with st.container():
            st.markdown(f"**{title}**")
            render_chart()
            st.caption(f"**What this shows:** {what_this_shows}")
        with st.expander("Show data", icon=":material/table:"):
            st.dataframe(data, hide_index=True, key=f"{key}_data")


def empty_state(title: str, message: str) -> None:
    """Designed empty state for selections without data."""
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.caption(message)


def reliability_panel(performance: dict) -> None:
    """Plain-language summary of how the model performed on unseen students."""
    recall = performance["dropout_recall"]
    precision = performance["dropout_precision"]
    rule = performance["business_rule"]
    with card():
        st.markdown("**:material/verified: How reliable is this check?**")
        st.markdown(
            f"- Catches **{recall:.0%}** of students who later drop out "
            f"(about {about_ratio(recall)}).\n"
            f"- **{precision:.0%}** of flagged students actually drop out "
            f"(about {about_ratio(precision)}).\n"
            f"- A simple rule (no second-semester unit passed, or fees in arrears) "
            f"catches {rule['dropout_recall']:.0%}."
        )
        st.caption(
            f"Tested on {performance['evaluated_students']:,} past students the model "
            "never saw. Use it to decide who to talk to first, not as a verdict."
        )


def footer(text: str) -> None:
    """Global footer shown under every page."""
    st.html(f'<div class="mi-footer">{text}</div>')


SIMPLE_DENOMINATORS = (2, 3, 4, 5)
SIMPLE_RATIO_TOLERANCE = 0.02


def about_ratio(share: float) -> str:
    """Express a share as a plain "N in M" ratio for non-technical readers.

    Uses a small denominator (halves, thirds, quarters, fifths) when one is
    within two percentage points, otherwise "N in 10".

    Example:
        >>> about_ratio(0.761), about_ratio(0.321), about_ratio(0.837)
        ('3 in 4', '1 in 3', '8 in 10')
    """
    for denominator in SIMPLE_DENOMINATORS:
        numerator = round(share * denominator)
        if (
            0 < numerator < denominator
            and abs(share - numerator / denominator) <= SIMPLE_RATIO_TOLERANCE
        ):
            return f"{numerator} in {denominator}"
    return f"{round(share * 10)} in 10"


def pct(value: float | None, digits: int = 1) -> str:
    """Format a share as a percentage ("—" when unavailable)."""
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def pp_delta(value: float | None, reference: float | None) -> str | None:
    """Difference in percentage points, for ``st.metric`` deltas."""
    if value is None or reference is None:
        return None
    return f"{100 * (value - reference):+.1f} pp vs. all students"
