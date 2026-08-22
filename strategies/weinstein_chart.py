"""
weinstein_chart.py
------------------
The Weinstein Stage-1 chart: the generic panel stack from src.plotter, plus the
base overlay that only this strategy's columns can support -- support and
resistance, the shaded range, the base rectangle, the swing touches that define
the levels, and the diagnostic metric box.

Why it lives here and not in src/plotter.py
-------------------------------------------
This overlay used to run for EVERY strategy, gated only on the presence of
columns named "Resistance"/"Support". That is column-name coupling masquerading
as generality: a second strategy publishing a column called "Support" would get
Weinstein annotations it never asked for, and Strategy.plot() -- the documented
extension point -- was dead code. The strategy owns its chart; src/plotter.py
owns the primitives.

Panels
------
Price, volume, MACD and relative strength all come from src/plotter.py, because
none of them needs a column this strategy declares. This module adds only the
price-panel overlay. make_figure() returns a dict of axes, so a panel this file
does not know about is simply drawn by the shared helpers and ignored here.

Columns consumed (all published by WeinsteinSetup.generate_signals):
    Resistance, Support, Base_Weeks, Range_Width_Pct, Prior_Decline_Pct,
    Distance_to_Breakout, Res_Touches, Sup_Touches, Is_Res_Touch, Is_Sup_Touch,
    Stage, Readiness_Score
Every one is optional at render time: a frame missing any of them still
produces a chart, minus that element. Diagnosing a non-detection is exactly the
case where columns are absent, so the chart must not fail then.
"""

from __future__ import annotations

import logging

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd

from src.instrument import Instrument
from src.plotter import (benchmark_label, draw_candles, draw_ma,
                         draw_ma_distance, draw_macd, draw_rs,
                         draw_signal_marker, draw_volume, finalize,
                         make_figure, prepare_frame)

log = logging.getLogger(__name__)

RES_COLOR = "#ef4444"
SUP_COLOR = "#22c55e"
BASE_COLOR = "#3b82f6"


def _value(row, key):
    """Scalar from a Series row, or None when absent/NaN."""
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return None
    return None if pd.isna(value) else value


def _subtitle(last) -> str:
    """Stage and score under the title, so a gallery of charts is readable
    without cross-referencing the results table."""
    bits = []
    stage = _value(last, "Stage")
    if stage not in (None, "", "-"):
        bits.append(f"Stage {stage}")
    score = _value(last, "Readiness_Score")
    if score is not None:
        bits.append(f"Readiness {float(score):.1f}")
    return "   \u00b7   ".join(bits)


def _base_overlay(ax, df: pd.DataFrame, last) -> None:
    """Support/resistance levels, the shaded range and the base rectangle --
    i.e. what the algorithm decided the base WAS on the final bar. This is the
    key view for diagnosing why a setup did or did not fire."""
    res = _value(last, "Resistance")
    sup = _value(last, "Support")
    if res is None or sup is None:
        return

    ax.axhspan(sup, res, color=BASE_COLOR, alpha=0.08, zorder=1)
    ax.axhline(res, color=RES_COLOR, linewidth=1.4, alpha=0.85, zorder=5,
               label=f"Resistance {res:.2f}")
    ax.axhline(sup, color=SUP_COLOR, linewidth=1.4, alpha=0.85, zorder=5,
               label=f"Support {sup:.2f}")

    weeks = _value(last, "Base_Weeks")
    if weeks is None or weeks < 1:
        return
    weeks = int(weeks)
    start = max(0, len(df) - weeks)
    x0 = mdates.date2num(df.index[start].to_pydatetime())
    x1 = mdates.date2num(df.index[-1].to_pydatetime())
    ax.add_patch(mpatches.Rectangle(
        (x0, sup), x1 - x0, res - sup, fill=False, edgecolor=BASE_COLOR,
        linewidth=1.6, linestyle="--", zorder=6))
    ax.text(x0, res, f" base: {weeks}w", color=BASE_COLOR, fontsize=9,
            va="bottom", ha="left", zorder=7, fontweight="bold")


def _touch_markers(ax, df: pd.DataFrame) -> None:
    """Mark the swing bars that were counted as level tests.

    _mark_touch_bars() has been computing Is_Res_Touch / Is_Sup_Touch all along
    and nothing was drawing them, so "Res_Touches: 3" had to be taken on trust.
    Showing WHICH bars were counted is what makes the touch count auditable.
    """
    for col, price_col, marker, color, label in (
        ("Is_Res_Touch", "High", "v", RES_COLOR, "Resistance touch"),
        ("Is_Sup_Touch", "Low", "^", SUP_COLOR, "Support touch"),
    ):
        if col not in df.columns:
            continue
        mask = df[col].fillna(False).astype(bool)
        if not mask.any():
            continue
        ax.scatter(df.index[mask], df.loc[mask, price_col], marker=marker,
                   s=30, color=color, zorder=7, label=label)


def _metric_box(ax, last) -> None:
    """Corner box with the numbers that decided the signal."""
    bits = []

    width = _value(last, "Range_Width_Pct")
    decline = _value(last, "Prior_Decline_Pct")

    base_weeks = _value(last, "Base_Weeks")
    if base_weeks is not None:
        bits.append(f"Base: {base_weeks:.0f}w")
    if width is not None and decline:
        # Weinstein's proportionality check: a base must be small relative to
        # the decline that preceded it.
        bits.append(f"Width/Decline: {width / decline:.2f}")
    for label, key, fmt in (
        ("Dist2BO", "Distance_to_Breakout", "{:.1f}%"),
        ("PriorDecline", "Prior_Decline_Pct", "{:.0f}%"),
        ("ResTouch", "Res_Touches", "{:.0f}"),
        ("SupTouch", "Sup_Touches", "{:.0f}"),
    ):
        value = _value(last, key)
        if value is not None:
            bits.append(f"{label}: {fmt.format(value)}")

    if not bits:
        return
    ax.text(0.985, 0.97, "\n".join(bits), transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9), zorder=8)


def plot_weinstein(instrument: "Instrument | str", df: pd.DataFrame,
                   out_path: str, signal_column: str = "Signal") -> bool:
    """Render the Stage-1 base chart. Returns False when the frame cannot be
    plotted at all, so the caller can fall back to the generic chart."""
    inst = Instrument.of(instrument)
    frame = prepare_frame(df)
    if frame is None:
        log.warning("plot_weinstein: insufficient columns for %s", inst.ticker)
        return False

    last = frame.iloc[-1]
    fig, axes = make_figure(inst, frame, subtitle=_subtitle(last))
    price = axes["price"]

    draw_candles(price, frame)
    ma = draw_ma(price, frame, weeks=30, column="SMA_30W", label="MA30 (30W)")
    _base_overlay(price, frame, last)
    _touch_markers(price, frame)
    # Upper left: distance from the MA30. Upper right: the base metrics. The
    # two corners answer "how extended is it right now" and "what did the
    # algorithm decide the base was" -- keep them apart.
    draw_ma_distance(price, frame, ma, ma_label="MA30")
    _metric_box(price, last)

    draw_volume(axes["volume"], frame)
    draw_macd(axes["macd"], frame)
    draw_rs(axes["rs"], frame, label=benchmark_label(frame))
    draw_signal_marker(price, frame, signal_column)

    finalize(fig, axes, out_path)
    return True
