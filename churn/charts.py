"""Altair chart builders. Pure functions that return charts so they can be unit-tested."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd  # type: ignore
from typing import Any

RED, GREEN, BLUE, GREY = "#dc2626", "#16a34a", "#2563eb", "#9ca3af"
CHURN_COLORS = alt.Scale(domain=["No", "Yes"], range=[BLUE, RED])


def drivers_chart(drivers: pd.DataFrame, top: int = 8) -> Any:
    data = drivers.head(top).copy()
    data["Percentage points"] = data["Effect on churn probability"] * 100
    data["Direction"] = data["Percentage points"].map(lambda v: "Raises risk" if v > 0 else "Lowers risk")
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("Percentage points:Q", title="Effect on churn probability (percentage points)"),
            y=alt.Y("Feature:N", sort=None, title=None),
            color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RED, GREEN]), legend=None),
            tooltip=["Feature", "Customer value", "Typical value", alt.Tooltip("Percentage points:Q", format="+.1f")],
        )
        .properties(height=28 * len(data) + 20)
    )


def rate_chart(df: pd.DataFrame, column: str) -> Any:
    """Churn rate per category of ``column`` (df must contain a 0/1 ``churned`` column)."""
    data = df.groupby(column, observed=True)["churned"].agg(["mean", "size"]).reset_index()
    data.columns = [column, "Churn rate", "Customers"]
    overall = float(df["churned"].mean())
    bars = alt.Chart(data).mark_bar(color=BLUE).encode(
        x=alt.X(f"{column}:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Churn rate:Q", axis=alt.Axis(format="%"), title="Churn rate"),
        tooltip=[column, alt.Tooltip("Churn rate:Q", format=".1%"), "Customers"],
    )
    rule = alt.Chart(pd.DataFrame({"y": [overall]})).mark_rule(color=RED, strokeDash=[5, 4]).encode(y="y:Q")
    return (bars + rule).properties(height=300)


def _bin_counts(values, edges) -> pd.DataFrame:
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    return pd.DataFrame({"start": edges[:-1], "end": edges[1:], "Customers": counts})


def hist_chart(df: pd.DataFrame, column: str, bins: int = 30) -> Any:
    """Overlaid histograms of ``column`` for churned and retained customers (pre-aggregated)."""
    edges = np.linspace(df[column].min(), df[column].max(), bins + 1)
    parts = []
    for label in ("No", "Yes"):
        part = _bin_counts(df.loc[df["Churn"] == label, column], edges)
        part["Churn"] = label
        parts.append(part)
    return (
        alt.Chart(pd.concat(parts, ignore_index=True))
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("start:Q", title=column),
            x2="end:Q",
            y=alt.Y("Customers:Q", stack=None),
            color=alt.Color("Churn:N", scale=CHURN_COLORS, title="Churned"),
            tooltip=[alt.Tooltip("start:Q", title=f"{column} from", format=".1f"), alt.Tooltip("end:Q", title="to", format=".1f"), "Churn", "Customers"],
        )
        .properties(height=300)
    )


def prob_hist(scored: pd.DataFrame, threshold: float) -> Any:
    data = _bin_counts(scored["churn_probability"], np.linspace(0, 1, 21))
    bars = alt.Chart(data).mark_bar(color=BLUE).encode(
        x=alt.X("start:Q", title="Predicted churn probability", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        x2="end:Q",
        y=alt.Y("Customers:Q"),
        tooltip=[alt.Tooltip("start:Q", title="from", format=".0%"), alt.Tooltip("end:Q", title="to", format=".0%"), "Customers"],
    )
    rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color=RED, strokeDash=[5, 4]).encode(x="x:Q")
    return (bars + rule).properties(height=240)


def roc_chart(points: pd.DataFrame, auc: float) -> Any:
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=GREY, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    curve = alt.Chart(points).mark_line(color=BLUE).encode(
        x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("True positive rate:Q", scale=alt.Scale(domain=[0, 1])),
    )
    return (diagonal + curve).properties(height=300, title=f"ROC curve (AUC = {auc:.3f})")


def threshold_chart(table: pd.DataFrame, current: float) -> Any:
    long = table.melt("threshold", ["precision", "recall", "f1"], var_name="Metric", value_name="Score")
    lines = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("threshold:Q", title="Decision threshold"),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Metric:N", scale=alt.Scale(range=[BLUE, RED, GREEN])),
    )
    rule = alt.Chart(pd.DataFrame({"x": [current]})).mark_rule(color=GREY, strokeDash=[5, 4]).encode(x="x:Q")
    return (lines + rule).properties(height=300, title="Precision / recall trade-off")


def importance_chart(table: pd.DataFrame, top: int = 10) -> Any:
    return (
        alt.Chart(table.head(top))
        .mark_bar(color=BLUE)
        .encode(
            x=alt.X("Importance:Q", title="Drop in ROC-AUC when the column is shuffled"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f"), alt.Tooltip("Std:Q", format=".4f")],
        )
        .properties(height=28 * min(top, len(table)) + 20)
    )


def confusion_chart(m: dict) -> Any:
    data = pd.DataFrame(
        [
            {"Actual": "Stayed", "Predicted": "Stayed", "Customers": m["tn"]},
            {"Actual": "Stayed", "Predicted": "Churned", "Customers": m["fp"]},
            {"Actual": "Churned", "Predicted": "Stayed", "Customers": m["fn"]},
            {"Actual": "Churned", "Predicted": "Churned", "Customers": m["tp"]},
        ]
    )
    base = alt.Chart(data).encode(x=alt.X("Predicted:N", sort=["Stayed", "Churned"]), y=alt.Y("Actual:N", sort=["Stayed", "Churned"]))
    heat = base.mark_rect().encode(color=alt.Color("Customers:Q", scale=alt.Scale(scheme="blues"), legend=None))
    text = base.mark_text(fontSize=22).encode(text="Customers:Q", color=alt.condition(alt.datum.Customers > data["Customers"].max() / 2, alt.value("white"), alt.value("black")))
    return (heat + text).properties(height=240, width=300, title="Confusion matrix")
