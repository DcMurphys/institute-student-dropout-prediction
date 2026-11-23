"""Filter bar for the dashboard: one multiselect per dimension."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from app.views.components.layout import grid
from app.views.display_names import field_label

KEY_PREFIX = "filter_"


def _clear(columns: list[str]) -> None:
    for column in columns:
        st.session_state[f"{KEY_PREFIX}{column}"] = []


def filter_bar(
    options: dict[str, list],
    format_option: Callable[[str, object], str],
) -> dict[str, list]:
    """Render the filters and return the selection per column.

    An empty selection means "all students" for that dimension.

    Args:
        options: Available values per column, in display order.
        format_option: ``(column, value) -> label`` for readable option names.
    """
    columns = list(options)
    with st.container(border=True):
        head, action = grid([5, 1], vertical_alignment="center")
        head.markdown("**:material/filter_alt: Filter students**")
        action.button(
            "Clear all",
            icon=":material/filter_alt_off:",
            on_click=_clear,
            args=(columns,),
            width="stretch",
            key="filters_clear",
        )
        selection = {}
        for slot, column in zip(grid(len(columns)), columns, strict=True):
            selection[column] = slot.multiselect(
                field_label(column),
                options[column],
                format_func=lambda value, c=column: format_option(c, value),
                placeholder="All",
                key=f"{KEY_PREFIX}{column}",
            )
    return selection
