"""Smoke tests: every page renders, and the main user flows work."""

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ("dashboard", "student_check", "batch_screening", "about")


def _harness(page: str) -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))
    from app.controllers import (
        about_controller,
        dashboard_controller,
        prediction_controller,
    )

    {
        "dashboard": dashboard_controller.show_dashboard_page,
        "student_check": prediction_controller.show_student_check_page,
        "batch_screening": prediction_controller.show_batch_screening_page,
        "about": about_controller.show_about_page,
    }[page]()


def _run(page: str) -> AppTest:
    at = AppTest.from_function(_harness, args=(page,), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.usefixtures("predictor", "students")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = _run(page)
    assert not at.error, [e.value for e in at.error]
    assert at.title, "page has no title"


@pytest.mark.usefixtures("predictor", "students")
def test_student_check_flow():
    at = _run("student_check")
    at.button[0].click()  # "Fill form" with the typical student
    at.run()
    at.get("form_submit_button")[0].click()
    at.run()
    assert not at.exception, at.exception
    assert "Result" in [s.value for s in at.subheader]


@pytest.mark.usefixtures("predictor", "students")
def test_batch_screening_with_sample_file():
    at = _run("batch_screening")
    next(b for b in at.button if b.label == "Try with sample file").click()
    at.run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "Students screened" in labels


@pytest.mark.usefixtures("predictor", "students")
@pytest.mark.parametrize("page", PAGES)
def test_pages_avoid_technical_jargon(page):
    """Staff never see data-science jargon on the pages."""
    at = _run(page)
    texts = [
        e.value
        for kind in ("markdown", "caption", "info", "warning", "title", "subheader")
        for e in getattr(at, kind)
    ]
    shown = " ".join(str(t) for t in texts).lower()
    for term in (
        "macro f1",
        "recall",
        "precision",
        "log1p",
        "pipeline",
        "notebook",
        "hyperparameter",
        "optuna",
        "p-value",
        "chi-square",
    ):
        assert term not in shown, term
