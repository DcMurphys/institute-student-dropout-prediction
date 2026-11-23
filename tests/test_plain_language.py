"""Plain-language number helpers used on the pages."""

import pytest

from app.views.components.layout import about_ratio


@pytest.mark.parametrize(
    ("share", "expected"),
    [
        (0.761, "3 in 4"),  # dropouts caught on the test students
        (0.321, "1 in 3"),  # overall dropout rate
        (0.837, "8 in 10"),  # no simple fraction within 2 points
        (0.5, "1 in 2"),
        (0.0, "0 in 10"),
        (1.0, "10 in 10"),
    ],
)
def test_about_ratio(share, expected):
    assert about_ratio(share) == expected
