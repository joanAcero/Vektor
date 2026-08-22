"""
stage_chart.py
--------------
Everything matplotlib that has to do with Weinstein stages. Two entry points:

    shade_stages(ax, index, stage_nums)   paint stage bands behind any chart
    plot_stage_chart(...)                 a standalone weekly stage chart

Both take their labels and colours from src/stages.py -- this module never
decides what a stage is, it only draws one.

WHITE BACKGROUND, on purpose. These charts used to be dark to match the web
UI, and it made the bands unreadable: a translucent wash over near-black
desaturates whatever it is given, so four different hues all composited into
much the same dark grey. On white the same colours keep their contrast. It
also matches src/plotter.py, so the sector charts and the Weinstein charts
look like one product.

Each band is drawn twice: a pale TINT across the full height for context, and
a full-opacity RIBBON along the bottom with the stage digit printed in it. The
digit is there because colour is not a reliable channel for everyone -- it
costs four lines and cannot be misread.

LAYERING NOTE
-------------
src/plotter.py imports shade_stages from here, so src/ depends on
notifications/. The clean fix is to move plotter.py into notifications/, where
the project's own rule already puts renderers -- it is the file in the wrong
place, not this one.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # headless: cron / systemd / Flask have no display
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.stages import (FALLBACK_WEEKS, MA_FAST_WEEKS, SMA_WEEKS, STAGE_COLOR,
                        Stage, classify, stage_spans)

log = logging.getLogger(__name__)

BAND_ALPHA = 0.20               # tint opacity; the ribbon carries the colour
RIBBON_FRAC = 0.06              # ribbon height, as a fraction of the axes
RIBBON_MIN_LABEL_FRAC = 0.03    # print the digit only if the run is this wide

_BG = "#ffffff"
_FG = "#222222"
_GRID = "#d8dde3"
# Price is the only pure black on the chart. Both averages are achromatic --
# every colour here means stage, and a coloured MA competes with the bands --
# so they are separated from the price and from each other on THREE channels
# at once: lightness (black / mid grey / light grey), weight (1.7 / 2.0 / 1.3)
# and dash (solid / solid / dashed). MA30 was previously near-black and read
# as a second price line; a mid grey keeps it prominent as the decision line
# without competing with the price itself.
_PRICE = "#000000"
_MA30 = "#6b7785"
_MA10 = "#aab4c0"

def shade_stages(ax, index, stage_nums, *, alpha: float = BAND_ALPHA,
                 ribbon: bool = True, labels: bool = True) -> None:
    """Paint contiguous stage runs as coloured bands on `ax`.

    Args:
        ax: any matplotlib Axes whose x-axis is dates.
        index: the DatetimeIndex the stages belong to.
        stage_nums: per-bar integer Stage values, same length as index.
        ribbon: draw the full-opacity strip along the bottom, with the stage
            digit in it. Turn off only if something else occupies the bottom
            of the axes.

    Silently does nothing on empty or mismatched input: a chart without bands
    is a smaller failure than no chart.
    """
    idx = pd.DatetimeIndex(index)
    arr = np.asarray(stage_nums, dtype=float)
    if len(idx) == 0 or len(idx) != len(arr):
        log.warning("shade_stages: index/stage length mismatch (%d vs %d); "
                    "skipping bands.", len(idx), len(arr))
        return

    # NaN (a column that never got computed) becomes UNKNOWN rather than
    # crashing the int cast.
    arr = np.where(np.isfinite(arr), arr, int(Stage.UNKNOWN)).astype(int)
    x = mdates.date2num(idx.to_pydatetime())
    # Half a bar of padding so adjacent bands meet instead of leaving hairlines.
    step = float(np.median(np.diff(x))) if len(x) > 1 else 7.0
    total = (x[-1] - x[0]) + step or 1.0

    for start, end, stage in stage_spans(arr):
        if stage is Stage.UNKNOWN:
            continue  # leave unclassified history as plain background
        x0, x1 = x[start] - step / 2, x[end] + step / 2
        color = STAGE_COLOR[stage]
        ax.axvspan(x0, x1, facecolor=color, alpha=alpha, edgecolor="none",
                   zorder=0)
        if not ribbon:
            continue
        # ymin/ymax on axvspan are AXES fractions, so the ribbon keeps its
        # height whatever the price range does.
        ax.axvspan(x0, x1, ymin=0.0, ymax=RIBBON_FRAC, facecolor=color,
                   edgecolor="none", zorder=2)
        if labels and (x1 - x0) / total >= RIBBON_MIN_LABEL_FRAC:
            ax.text((x0 + x1) / 2, RIBBON_FRAC / 2, stage.short,
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="center", color="#ffffff", fontsize=9,
                    fontweight="bold", zorder=3, clip_on=True)


def stage_legend_handles(stages=(Stage.TWO, Stage.ONE, Stage.THREE, Stage.FOUR)):
    """Patch handles for a stage legend, in hunting order. Full opacity, so the
    key shows the reference colour rather than a washed-out sample of it."""
    from matplotlib.patches import Patch
    return [Patch(facecolor=STAGE_COLOR[s], edgecolor="none", label=s.label)
            for s in stages]


def plot_stage_chart(symbol: str, weekly_close: pd.Series, out_path: str, *,
                     subtitle: str = "", years: float = 5.0,
                     prior_decline_pct: pd.Series | None = None) -> str | None:
    """
    Weekly close + 30W MA on stage-coloured bands, written to `out_path`.

    This is the verification instrument: it shows what src/stages.py decided
    for every week, so a wrong rule is visible rather than inferred. Returns
    the path, or None when there is nothing to draw.

    `years` trims the VIEW only. Classification always runs on the full series,
    because the MA needs SMA_WEEKS of warm-up and the 1-vs-3 fallback needs
    FALLBACK_WEEKS more -- classifying a trimmed series would relabel the first
    ~1.6 years as UNKNOWN and the chart would lie.
    """
    if weekly_close is None or len(weekly_close) < SMA_WEEKS + 2:
        log.warning("plot_stage_chart: not enough history for %s.", symbol)
        return None

    close = pd.Series(weekly_close).astype(float).dropna()
    st = classify(close, prior_decline_pct=prior_decline_pct)

    cutoff = close.index.max() - pd.Timedelta(weeks=int(years * 52))
    view = st.loc[st.index >= cutoff]
    if view.empty:
        view = st
    vclose = close.reindex(view.index)

    fig, ax = plt.subplots(figsize=(12, 4.0))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    shade_stages(ax, view.index, view["stage"].values)

    ax.plot(view.index, vclose, color=_PRICE, lw=1.7, zorder=6,
            label="weekly close")
    ax.plot(view.index, view["ma"], color=_MA30, lw=2.0, zorder=5,
            label=f"{SMA_WEEKS}W MA")
    ax.plot(view.index, view["ma_fast"], color=_MA10, lw=1.3, ls=(0, (4, 2)),
            zorder=4, label=f"{MA_FAST_WEEKS}W MA ({MA_FAST_WEEKS * 5}d)")

    current = Stage(int(view["stage"].iloc[-1]))
    margin = float(view["transition_margin_pct"].iloc[-1])
    used_decline = bool(view["decline_used"].iloc[-1])

    head = f"{symbol}  —  stage {current.label}"
    if subtitle:
        head += f"   ·   {subtitle}"
    ax.set_title(head, color=STAGE_COLOR[current], fontsize=13,
                 fontweight="bold", pad=10, loc="left")

    # The honesty line: name the transition that produced the current label
    # and when it fired, so a wrong band can be traced to a rule rather than
    # guessed at.
    spans = [sp for sp in stage_spans(st["stage"].values)
             if sp[2] is not Stage.UNKNOWN]
    if len(spans) >= 2:
        prev = spans[-2][2]
        since = st.index[spans[-1][0]].date()
        rule = {
            (Stage.ONE, Stage.TWO):   f"close above the {SMA_WEEKS}W MA with the MA rising",
            (Stage.ONE, Stage.FOUR):  f"close below the {SMA_WEEKS}W MA with the MA falling",
            (Stage.TWO, Stage.THREE): f"close below the {MA_FAST_WEEKS}W MA",
            (Stage.TWO, Stage.FOUR):  f"close below the {SMA_WEEKS}W MA",
            (Stage.THREE, Stage.FOUR): f"close below the {SMA_WEEKS}W MA",
            (Stage.THREE, Stage.TWO): f"close back above the {MA_FAST_WEEKS}W MA with the MA30 rising",
            (Stage.FOUR, Stage.ONE):  f"close back above the {MA_FAST_WEEKS}W MA, MA30 no longer falling",
        }.get((prev, current), "seeded, no transition on record")
        note = f"stage {prev.short} -> {current.short} on {since}: {rule}"
    else:
        note = ("seeded from the measured prior decline" if used_decline
                else f"seeded from MA context: MA is {margin:+.1f}% vs "
                     f"{FALLBACK_WEEKS}w ago" if np.isfinite(margin)
                else "no context available")
    ax.text(0.005, -0.17, note, transform=ax.transAxes, color="#666666",
            fontsize=8.5, family="monospace", va="top")

    ax.legend(handles=stage_legend_handles()
              + list(ax.get_legend_handles_labels()[0]),
              loc="upper left", fontsize=8.5, ncols=7, framealpha=0.9,
              facecolor=_BG, edgecolor=_GRID, labelcolor=_FG)

    lo, hi = float(np.nanmin(vclose)), float(np.nanmax(vclose))
    pad = (hi - lo) * 0.06 or hi * 0.06
    ax.set_ylim(lo - pad - (hi - lo) * RIBBON_FRAC * 2.0, hi + pad * 3.0)

    ax.set_ylabel("Price", color=_FG, fontsize=10)
    ax.tick_params(colors=_FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(True, color=_GRID, alpha=0.7, lw=0.6, zorder=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=110, facecolor=_BG, bbox_inches="tight")
    except OSError as e:
        log.error("Could not write stage chart to %s: %s", out_path, e)
        plt.close(fig)
        return None
    plt.close(fig)
    return out_path
