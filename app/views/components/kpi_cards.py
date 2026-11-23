"""KPI cards: a metric, an optional delta and a one-line caption."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.views.components.layout import card, grid


@dataclass(frozen=True)
class Kpi:
    """Content of one KPI card."""

    label: str
    value: str
    caption: str
    delta: str | None = None
    delta_color: str = "normal"  # "inverse" when a rise is bad (e.g. dropout rate)
    help: str | None = None


def kpi_card(kpi: Kpi) -> None:
    """Bordered card; all cards in a row share the tallest card's height."""
    with card():
        st.metric(
            kpi.label,
            kpi.value,
            delta=kpi.delta,
            delta_color=kpi.delta_color,
            help=kpi.help,
        )
        st.caption(kpi.caption)


def kpi_row(kpis: list[Kpi], columns: int | None = None) -> None:
    """Render KPI cards side by side in equal columns.

    Args:
        kpis: Cards to show.
        columns: Fixed column count; keeps card widths equal to other rows
            even when this row has fewer cards. Defaults to ``len(kpis)``.
    """
    for column, kpi in zip(grid(columns or len(kpis)), kpis, strict=False):
        with column:
            kpi_card(kpi)
