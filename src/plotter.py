"""
plotter.py
----------
Chart PRIMITIVES, plus one genuinely strategy-agnostic fallback chart.

Scope rule (previously stated, previously violated)
---------------------------------------------------
Nothing in this module may reference a column outside the OHLCV contract, the
generic StrategyMeta declarations, or the display-indicator columns produced by
src/indicators.py. An earlier version claimed to have "no hard-coded references
to Weinstein columns" and then read Resistance, Support, Base_Weeks and five
more -- it was the Weinstein chart under a generic name. That overlay now lives
with its owner in strategies/weinstein_chart.py, which composes the helpers
below.

PANEL STACK
-----------
    price   candles, MA, S/R overlays, signal marker
    volume  bars + 4-week average
    MACD    12/26/9, histogram + both lines
    RS      Mansfield relative strength vs the run's benchmark

Panels below price appear only when their columns are present, so a frame
without Volume, or a strategy run with no benchmark, simply gets a shorter
chart instead of an empty panel. MACD and RS columns come from
src/indicators.py::add_display_indicators(), which run.py calls on the FULL
frame before the display trim -- see that module for why the ordering matters.

BREAKING CHANGE, DOCUMENTED
---------------------------
make_figure() now returns (fig, axes) where `axes` is a dict keyed
"price"/"volume"/"macd"/"rs", values None when that panel is absent. It used to
return a fixed 3-tuple. A dict rather than a longer tuple because the next
panel added would break every call site again; with a dict, a caller that does
not know about a new panel is simply unaffected.

Titles come from Instrument.label() -- "Name - TICKER - Sector", degrading to
whichever parts are known. The first argument accepts a bare ticker string too
(coerced by Instrument.of), so an un-migrated caller keeps working.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # headless: required for CI / cron with no display
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.indicators import BENCHMARK_COLUMN
from src.instrument import Instrument

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared palette. Strategy charts import these so a custom overlay never
# reinvents the candle or MA colours.
# ---------------------------------------------------------------------------
UP, DN = "#26a641", "#e03131"
VOL_UP, VOL_DN = "#86efac", "#fca5a5"
MA_COLOR = "#e07b00"
# Dark slate: the volume average has to read over both the green and the red
# bars, so it cannot borrow either of their hues.
VOL_MA_COLOR = "#334155"
MACD_COLOR = "#1d4ed8"
MACD_SIGNAL_COLOR = "#b91c1c"
MACD_HIST_UP, MACD_HIST_DN = "#93c5fd", "#fecaca"
RS_COLOR = "#6d28d9"
RS_FILL_UP, RS_FILL_DN = "#c4b5fd", "#fca5a5"
SIGNAL_COLOR = "#16a34a"
ZERO_LINE = "#94a3b8"

OHLC = ("Open", "High", "Low", "Close")
BAR_WIDTH_DAYS = 5  # weekly bars drawn on a date axis

VOL_MA_WEEKS = 4

# A strategy MAY publish its averages under one of these names so the plotter
# reuses them instead of recomputing. A naming CONVENTION open to every
# strategy, not knowledge of any particular one.
MA_COLUMN_CANDIDATES = ("SMA_30W", "SMA", "MA30")
VOL_MA_COLUMN_CANDIDATES = ("Vol_SMA_4", "Vol_MA_4", "Volume_SMA_4")

# Panel layout: height ratios and the base figure height they add.
_PRICE_RATIO = 3
_SUB_RATIO = 1
_BASE_HEIGHT = 6.4
_SUB_HEIGHT = 1.9


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def prepare_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    """Validate and normalise a frame for plotting; None if unusable."""
    if df is None or df.empty or not set(OHLC) <= set(df.columns):
        return None
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _published_column(df: pd.DataFrame, candidates) -> pd.Series | None:
    for name in candidates:
        if name in df.columns:
            return df[name]
    return None


def _has_values(df: pd.DataFrame, column: str) -> bool:
    """True when the column exists AND carries at least one finite value.

    Presence alone is not enough: a strategy that runs without a benchmark
    still emits an all-NaN Mansfield_RS column, and drawing an empty panel for
    it wastes a third of the chart on nothing.
    """
    if column not in df.columns:
        return False
    return bool(pd.to_numeric(df[column], errors="coerce").notna().any())


def benchmark_label(df: pd.DataFrame) -> str:
    """The benchmark symbol carried on the frame, or "" if unknown."""
    if BENCHMARK_COLUMN not in df.columns or df.empty:
        return ""
    value = df[BENCHMARK_COLUMN].dropna()
    return "" if value.empty else str(value.iloc[-1])


def make_figure(instrument: "Instrument | str", df: pd.DataFrame, *,
                subtitle: str = ""):
    """Figure and the panel stack. Returns (fig, axes-dict).

    See the module docstring: `axes` is keyed "price"/"volume"/"macd"/"rs" and
    a key is None when that panel's data is absent.
    """
    inst = Instrument.of(instrument)

    wanted = [("price", True),
              ("volume", "Volume" in df.columns),
              ("macd", _has_values(df, "MACD")),
              ("rs", _has_values(df, "Mansfield_RS"))]
    keys = [name for name, present in wanted if present]
    ratios = [_PRICE_RATIO if k == "price" else _SUB_RATIO for k in keys]
    height = _BASE_HEIGHT + _SUB_HEIGHT * (len(keys) - 1)

    fig, raw = plt.subplots(len(keys), 1, figsize=(14, height), sharex=True,
                            gridspec_kw={"height_ratios": ratios})
    raw = np.atleast_1d(raw)
    axes = {name: None for name, _ in wanted}
    for key, ax in zip(keys, raw):
        axes[key] = ax

    fig.suptitle(inst.label(), fontsize=14, fontweight="bold")
    if subtitle:
        axes["price"].set_title(subtitle, fontsize=9.5, color="#555555", loc="left")
    return fig, axes


def draw_candles(ax, df: pd.DataFrame, width: float = BAR_WIDTH_DAYS) -> None:
    o, h, l, c = (df[k].values for k in OHLC)
    x = mdates.date2num(df.index.to_pydatetime())
    for i in range(len(df)):
        col = UP if c[i] >= o[i] else DN
        ax.plot([x[i], x[i]], [l[i], h[i]], color=col, linewidth=0.9, zorder=3)
        ax.add_patch(mpatches.Rectangle(
            (x[i] - width / 2, min(o[i], c[i])), width, abs(c[i] - o[i]) or 1e-9,
            facecolor=col, edgecolor=col, linewidth=0.4, zorder=4))


def draw_ma(ax, df: pd.DataFrame, weeks: int = 30, column: str | None = None,
            label: str | None = None) -> pd.Series:
    """Draw the trend MA, reusing a published column when there is one.

    Frames reaching here are weekly (strategies resample to W-FRI), so the
    default is the 30-week MA. Returns the series so the caller can pass it to
    draw_ma_distance() without recomputing or re-guessing which column won.
    """
    series = df[column] if (column and column in df.columns) else None
    if series is None:
        series = _published_column(df, MA_COLUMN_CANDIDATES)
    if series is None:
        series = df["Close"].rolling(window=weeks).mean()

    ax.plot(df.index, series, color=MA_COLOR, linewidth=1.6, linestyle="--",
            alpha=0.9, label=label or f"MA{weeks} ({weeks}W)", zorder=6)
    return series


def draw_ma_distance(ax, df: pd.DataFrame, ma_series: pd.Series | None, *,
                     ma_label: str = "MA30") -> None:
    """Upper-left box: how far the last bar sits from the trend MA.

    BOTH the close and the weekly high are printed, because they answer
    different questions and the difference between them is itself information.
    The close is what the stage rules act on. The high is how extended the bar
    actually got -- on a breakout week the high can be several percent above a
    close that still looks tame, and that gap is exactly the risk in chasing it.
    """
    if ma_series is None or df is None or df.empty:
        return
    try:
        ma = float(ma_series.iloc[-1])
        close = float(df["Close"].iloc[-1])
        high = float(df["High"].iloc[-1])
    except (IndexError, TypeError, ValueError):
        return
    if not np.isfinite(ma) or ma <= 0:
        return

    text = (f"Close vs {ma_label}: {(close / ma - 1) * 100:+.1f}%\n"
            f"High  vs {ma_label}: {(high / ma - 1) * 100:+.1f}%")
    ax.text(0.012, 0.97, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9), zorder=8)


def draw_volume(ax, df: pd.DataFrame, ma_weeks: int = VOL_MA_WEEKS) -> None:
    """Volume bars plus the `ma_weeks`-week average as a line.

    The average is the reference the bars are read AGAINST: a breakout is
    supposed to come on volume well above it, and a Stage 1 base is supposed to
    dry up below it. Without the line you are eyeballing that comparison
    against a memory of the left-hand side of the chart.
    """
    if ax is None or "Volume" not in df.columns:
        return
    opens = df["Open"] if "Open" in df.columns else df["Close"].shift(1)
    colors = np.where(df["Close"] >= opens, VOL_UP, VOL_DN)
    ax.bar(df.index, df["Volume"], color=colors, width=BAR_WIDTH_DAYS)

    series = _published_column(df, VOL_MA_COLUMN_CANDIDATES)
    if series is None:
        series = df["Volume"].rolling(window=ma_weeks).mean()
    ax.plot(df.index, series, color=VOL_MA_COLOR, linewidth=1.3, alpha=0.95,
            label=f"Vol MA({ma_weeks})", zorder=5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    ax.set_ylabel("Volume")
    ax.grid(True, linestyle=":", alpha=0.5)


def draw_macd(ax, df: pd.DataFrame) -> None:
    """MACD panel: histogram behind, both lines in front, zero line.

    Columns come from src/indicators.py. Nothing is computed here -- the same
    single-renderer discipline the RRG and stage charts follow, for the same
    reason: two implementations of one indicator eventually disagree.
    """
    if ax is None or "MACD" not in df.columns:
        return
    hist = df.get("MACD_Hist")
    if hist is not None:
        colors = np.where(hist.fillna(0) >= 0, MACD_HIST_UP, MACD_HIST_DN)
        ax.bar(df.index, hist, color=colors, width=BAR_WIDTH_DAYS, zorder=2)
    ax.plot(df.index, df["MACD"], color=MACD_COLOR, linewidth=1.3,
            label="MACD(12,26)", zorder=4)
    if "MACD_Signal" in df.columns:
        ax.plot(df.index, df["MACD_Signal"], color=MACD_SIGNAL_COLOR,
                linewidth=1.1, label="Signal(9)", zorder=4)
    ax.axhline(0, color=ZERO_LINE, linewidth=0.9, zorder=1)
    ax.set_ylabel("MACD")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85, ncol=2)


def draw_rs(ax, df: pd.DataFrame, *, label: str = "") -> None:
    """Mansfield relative strength, filled either side of zero.

    Zero is the whole point of the panel and is drawn heavier than the grid:
    above it the stock is beating the benchmark relative to its own 52-week
    norm, below it it is lagging. The fill makes the crossing visible at a
    glance, which is the event Weinstein cares about at the purchase point.

    The benchmark is named in the y-label because it is a RUN-level choice --
    the same stock charted from a US-index scan and from a sector scan can show
    different RS, and an unlabelled panel would make those two charts look
    contradictory.
    """
    if ax is None or "Mansfield_RS" not in df.columns:
        return
    rs = pd.to_numeric(df["Mansfield_RS"], errors="coerce")
    ax.plot(df.index, rs, color=RS_COLOR, linewidth=1.4, zorder=4)
    ax.fill_between(df.index, 0, rs, where=(rs >= 0), color=RS_FILL_UP,
                    alpha=0.75, interpolate=True, zorder=2)
    ax.fill_between(df.index, 0, rs, where=(rs < 0), color=RS_FILL_DN,
                    alpha=0.75, interpolate=True, zorder=2)
    ax.axhline(0, color=ZERO_LINE, linewidth=1.1, zorder=3)
    ax.set_ylabel(f"RS vs {label}" if label else "Mansfield RS")
    ax.grid(True, linestyle=":", alpha=0.5)


def draw_signal_marker(ax, df: pd.DataFrame,
                       signal_column: str = "Signal") -> bool:
    """Arrow on the final bar when the signal fires. Column name comes from
    StrategyMeta, so this stays generic."""
    if signal_column not in df.columns or df.empty:
        return False
    last = df[signal_column].iloc[-1]
    if pd.isna(last) or int(last) == 0:
        return False
    x = mdates.date2num(df.index[-1].to_pydatetime())
    y = float(df["High"].iloc[-1]) * 1.02
    ax.annotate("", xy=(x, y * 1.03), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=SIGNAL_COLOR, lw=2.4))
    return True


def finalize(fig, axes: dict, out_path: str, *, legend: bool = True,
             legend_loc: str = "lower left") -> None:
    """Axis formatting, save, close.

    The price legend sits LOWER left because draw_ma_distance() owns the upper
    left. On a base that corner is usually empty; move it if a chart disagrees.

    Always close: a cron/CI run that plots hundreds of tickers leaks figures
    otherwise.
    """
    ax_price = axes["price"]
    if legend and ax_price.get_legend_handles_labels()[0]:
        ax_price.legend(loc=legend_loc, fontsize=9, framealpha=0.9)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle=":", alpha=0.5)

    # Date formatting belongs on the BOTTOM panel only; sharex propagates the
    # limits, and labelling an interior panel leaves ticks stranded mid-figure.
    bottom = [ax for ax in axes.values() if ax is not None][-1]
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    bottom.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(bottom.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generic fallback chart
# ---------------------------------------------------------------------------
def plot_generic(instrument: "Instrument | str", df: pd.DataFrame,
                 out_path: str, signal_column: str = "Signal") -> bool:
    """Weekly candles + MA + volume + MACD + RS + signal marker. No
    strategy-specific columns are read, by design."""
    inst = Instrument.of(instrument)
    frame = prepare_frame(df)
    if frame is None:
        log.warning("plot_generic: insufficient columns for %s", inst.ticker)
        return False

    fig, axes = make_figure(inst, frame)
    draw_candles(axes["price"], frame)
    ma = draw_ma(axes["price"], frame)
    draw_ma_distance(axes["price"], frame, ma, ma_label="MA30")
    draw_volume(axes["volume"], frame)
    draw_macd(axes["macd"], frame)
    draw_rs(axes["rs"], frame, label=benchmark_label(frame))
    draw_signal_marker(axes["price"], frame, signal_column)
    finalize(fig, axes, out_path)
    return True
