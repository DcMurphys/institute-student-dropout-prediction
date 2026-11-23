"""App shell: page config, branding, navigation, error boundary and footer."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

from app.config import APP_NAME, FOOTER_TEXT, INSTITUTION_NAME, LOGO_PATH, SYMBOL_PATH
from app.controllers import resources
from app.controllers.about_controller import show_about_page
from app.controllers.dashboard_controller import show_dashboard_page
from app.controllers.prediction_controller import (
    show_batch_screening_page,
    show_student_check_page,
)
from app.models.dropout_predictor import ArtifactsMissingError
from app.views.components import layout

logger = logging.getLogger("student_early_warning")


@dataclass(frozen=True)
class PageSpec:
    """One page in the sidebar navigation."""

    section: str
    url: str
    title: str
    icon: str
    render: Callable[[], None]


PAGE_SPECS = (
    PageSpec(
        "Insights",
        "dashboard",
        "Dashboard",
        ":material/dashboard:",
        show_dashboard_page,
    ),
    PageSpec(
        "Early warning",
        "student-check",
        "Student Check",
        ":material/person_search:",
        show_student_check_page,
    ),
    PageSpec(
        "Early warning",
        "batch-screening",
        "Batch Screening",
        ":material/groups:",
        show_batch_screening_page,
    ),
    PageSpec(
        "Help", "about-model", "About the Model", ":material/school:", show_about_page
    ),
)
DEFAULT_PAGE = "dashboard"


def _guarded(spec: PageSpec) -> Callable[[], None]:
    """Wrap a page so unexpected errors show a friendly message and a reference code."""

    def run() -> None:
        try:
            spec.render()
        except Exception:  # noqa: BLE001 - last-resort boundary for the UI
            code = uuid.uuid4().hex[:8].upper()
            logger.exception("Unhandled error on page %s (ref %s)", spec.url, code)
            st.error(
                "Something went wrong while showing this page. Please try again. "
                f"Reference code: **{code}**",
                icon=":material/error:",
            )

    run.__name__ = f"page_{spec.url.replace('-', '_')}"
    return run


def _check_resources() -> bool:
    """Stop early with a clear message when data or model files are missing."""
    try:
        resources.predictor()
        resources.students()
    except (ArtifactsMissingError, FileNotFoundError) as error:
        logger.error("%s", error)
        st.error(
            "The early-warning data or model is not available yet, so the app cannot "
            "start. Ask the data team to run the notebook export.",
            icon=":material/error:",
        )
        st.caption(str(error))
        return False
    return True


def run() -> None:
    """Entry point called by ``app/main.py``."""
    st.set_page_config(
        page_title=f"{APP_NAME} · {INSTITUTION_NAME}",
        page_icon=str(SYMBOL_PATH),
        layout="wide",
        initial_sidebar_state="auto",
    )
    layout.inject_css()
    st.logo(str(LOGO_PATH), icon_image=str(SYMBOL_PATH), size="large")

    if not _check_resources():
        layout.footer(FOOTER_TEXT)
        st.stop()

    sections: dict[str, list] = {}
    for spec in PAGE_SPECS:
        page = st.Page(
            _guarded(spec),
            title=spec.title,
            icon=spec.icon,
            url_path=spec.url,
            default=spec.url == DEFAULT_PAGE,
        )
        sections.setdefault(spec.section, []).append(page)
    current = st.navigation(sections, position="sidebar", expanded=True)
    with st.sidebar:
        st.caption(
            f":material/info: {APP_NAME}. Prototype for advisors and academic "
            "leadership; predictions support, not replace, staff judgement."
        )
    current.run()
    layout.footer(FOOTER_TEXT)
