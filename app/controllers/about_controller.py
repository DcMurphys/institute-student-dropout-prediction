"""About the Model page."""

from __future__ import annotations

from app.controllers import resources
from app.views import about_view as view


def show_about_page() -> None:
    view.render(resources.model_performance(), resources.predictor().threshold)
