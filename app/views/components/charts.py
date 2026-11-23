"""Altair charts following the palette rules.

Context series are neutral grey and the series to look at is brand orange.
Status colours (Dropout red, Enrolled amber, Graduate green) are used only
when a chart is split by status, and always come with a text label.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from app.config import PALETTE, STATUS_ORDER

FONT = "Arimo, Helvetica Neue, Arial, sans-serif"
STATUS_COLORS = [
    PALETTE["status-dropout"],
    PALETTE["status-enrolled"],
    PALETTE["status-graduate"],
]
STATUS_SCALE = alt.Scale(domain=list(STATUS_ORDER), range=STATUS_COLORS)
PANEL_HEIGHT = 280
LABEL_HEADROOM = 1.3  # x-axis room beyond the longest bar for its text label


def _style(chart: alt.Chart | alt.LayerChart, height: int | alt.Step = PANEL_HEIGHT):
    return (
        chart.properties(height=height)
        .configure_axis(
            labelColor=PALETTE["slate"],
            titleColor=PALETTE["slate"],
            labelFont=FONT,
            titleFont=FONT,
            gridColor=PALETTE["border"],
            domainColor=PALETTE["light-grey"],
        )
        .configure_legend(
            labelFont=FONT, titleFont=FONT, labelColor=PALETTE["slate"], labelLimit=0
        )
        .configure_text(font=FONT)
        .configure_view(strokeWidth=0)
    )


def status_distribution_chart(distribution: pd.DataFrame) -> alt.Chart:
    """Horizontal bars of students per status with count and share labels.

    Args:
        distribution: Columns ``Status``, ``Students``, ``Share``.
    """
    data = distribution.assign(
        label=lambda d: [
            f"{n:,} ({s:.1%})" for n, s in zip(d["Students"], d["Share"], strict=True)
        ]
    )
    # Headroom to the right of the longest bar so its label is not cut off.
    x_max = max(int(data["Students"].max() * LABEL_HEADROOM), 1)
    base = alt.Chart(data).encode(
        y=alt.Y("Status:N", sort=list(STATUS_ORDER), title=None),
        x=alt.X("Students:Q", title="Students", scale=alt.Scale(domain=[0, x_max])),
        tooltip=[
            "Status",
            alt.Tooltip("Students:Q", format=","),
            alt.Tooltip("Share:Q", format=".1%"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color("Status:N", scale=STATUS_SCALE, legend=None)
    )
    labels = base.mark_text(align="left", dx=6, color=PALETTE["ink"]).encode(
        text="label:N"
    )
    return _style(bars + labels, height=alt.Step(72))


def dropout_rate_chart(table: pd.DataFrame, overall: float, title: str) -> alt.Chart:
    """Ranked dropout rate per group with the overall average as a dashed line.

    Args:
        table: Output of ``dropout_rate_table`` with a readable ``group`` column.
        overall: Overall dropout rate of the selection.
        title: Axis title for the grouping.
    """
    data = table.assign(
        Size=lambda d: d["too_small"].map(
            {False: "Large enough to compare", True: "Fewer than 50 students"}
        ),
        label=lambda d: [
            f"{r:.0%} · n = {n:,}"
            for r, n in zip(d["rate"], d["students"], strict=True)
        ],
    )
    order = data["group"].astype(str).tolist()
    data = data.assign(group=data["group"].astype(str))
    y = alt.Y("group:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=y,
            x=alt.X(
                "rate:Q",
                title="Dropout rate",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "Size:N",
                scale=alt.Scale(
                    domain=["Large enough to compare", "Fewer than 50 students"],
                    range=[PALETTE["brand"], PALETTE["light-grey"]],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("group:N", title=title),
                alt.Tooltip("students:Q", title="Students", format=","),
                alt.Tooltip("rate:Q", title="Dropout rate", format=".1%"),
                alt.Tooltip("lift:Q", title="Times the average", format=".2f"),
            ],
        )
    )
    text = bars.mark_text(
        align="left", dx=4, color=PALETTE["slate"], fontSize=11
    ).encode(text="label:N", color=alt.value(PALETTE["slate"]))
    rule = (
        alt.Chart(pd.DataFrame({"overall": [overall]}))
        .mark_rule(strokeDash=[6, 4], color=PALETTE["ink"])
        .encode(
            x="overall:Q",
            tooltip=[alt.Tooltip("overall:Q", title="All students", format=".1%")],
        )
    )
    # A fixed step per group keeps bars readable for 2 or 20 groups.
    return _style(rule + bars + text, height=alt.Step(34))


def progress_boxplot(
    data: pd.DataFrame, column: str, title: str, as_percent: bool
) -> alt.Chart:
    """Distribution of one progress measure per status (box plot).

    The box covers the middle half of the students; the line is the median.
    """
    axis = alt.Axis(format="%") if as_percent else alt.Axis()
    chart = (
        alt.Chart(data)
        .mark_boxplot(
            extent=1.5,
            size=38,
            median={"color": PALETTE["ink"]},
            outliers={"size": 8, "opacity": 0.35},
        )
        .encode(
            x=alt.X(
                "Status:N",
                sort=list(STATUS_ORDER),
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(f"{column}:Q", title=title, axis=axis),
            color=alt.Color("Status:N", scale=STATUS_SCALE, legend=None),
        )
    )
    return _style(chart)


def class_probability_chart(probabilities: pd.DataFrame) -> alt.Chart:
    """Bars of the model's probability for each status.

    Args:
        probabilities: Columns ``Status`` and ``Probability``.
    """
    base = alt.Chart(probabilities).encode(
        y=alt.Y("Status:N", sort=list(STATUS_ORDER), title=None),
        x=alt.X(
            "Probability:Q",
            title="Probability",
            axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=[0, 1]),
        ),
        tooltip=["Status", alt.Tooltip("Probability:Q", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color("Status:N", scale=STATUS_SCALE, legend=None)
    )
    labels = base.mark_text(align="left", dx=6, color=PALETTE["ink"]).encode(
        text=alt.Text("Probability:Q", format=".0%")
    )
    return _style(bars + labels, height=alt.Step(30))


def risk_distribution_chart(scores: pd.DataFrame, threshold: float) -> alt.Chart:
    """Histogram of dropout probabilities with the decision threshold.

    Args:
        scores: Column ``Dropout probability``.
        threshold: Probability at or above which a student is flagged.
    """
    data = scores.assign(
        Flag=lambda d: (
            d["Dropout probability"]
            .ge(threshold)
            .map({True: "Flagged", False: "Not flagged"})
        )
    )
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "Dropout probability:Q",
                bin=alt.Bin(step=0.05, extent=[0, 1]),
                title="Dropout probability",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("count():Q", title="Students"),
            color=alt.Color(
                "Flag:N",
                scale=alt.Scale(
                    domain=["Flagged", "Not flagged"],
                    range=[PALETTE["status-dropout"], PALETTE["context-grey"]],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[alt.Tooltip("count():Q", title="Students"), "Flag:N"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(strokeDash=[6, 4], color=PALETTE["ink"])
        .encode(x="threshold:Q")
    )
    label = (
        alt.Chart(
            pd.DataFrame(
                {"threshold": [threshold], "text": [f"Flag at {threshold:.0%}"]}
            )
        )
        .mark_text(align="left", dx=4, dy=-6, color=PALETTE["ink"], fontSize=11)
        .encode(x="threshold:Q", y=alt.value(8), text="text:N")
    )
    return _style(bars + rule + label, height=240)
