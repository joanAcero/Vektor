"""
indicators.py
-------------
Pure indicator computation for CHART panels. No plotting, no strategy logic.

Why a separate module
---------------------
MACD and the relative-strength line are displayed on every chart but gate
nothing. Computing them inside src/plotter.py would put arithmetic in the
drawing layer; computing them inside a strategy would make every other strategy
inherit that strategy's choices. They live here, and run.py attaches them to
the signals frame before the chart pass.

THE WARM-UP RULE (the reason this is called where it is)
--------------------------------------------------------
run.py trims the frame to CHART_YEARS for DISPLAY, after signals are computed
on full history. Anything derived from the trimmed frame therefore starts blank
at the left edge: a 26-week EMA would be undefined for the first ~26 bars of
the visible window, and a 52-week Mansfield RS for the first ~52 -- a full year
of an empty panel on a three-year chart.

So add_display_indicators() must be called on the FULL frame, before the trim.
That ordering is not a detail; it is the whole reason this is a separate step
rather than something the plotter does for itself.

RELATIVE STRENGTH IS NOT RECOMPUTED IF A STRATEGY PUBLISHED IT
--------------------------------------------------------------
WeinsteinSetup, MomentumLeaders and BrowseAll all emit `Mansfield_RS` from
src.benchmarks.mansfield_rs with the benchmark the Screener injected. When that
column is present it is left alone, so the line on the chart is the same number
as the column in the results table and the CSV. Recomputing it here -- even
with the same function -- would create two values that could differ whenever a
strategy uses a non-default lookback, and a chart disagreeing with the table it
came from is a bug you find late and trust little.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Standard MACD. Weekly bars here, so these are weeks, not days -- the same
# (12, 26, 9) triple Weinstein-style weekly analysis uses.
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Mansfield RS lookback when this module has to compute it. Matches
# RS_PERIOD_WEEKS in strategies/weinstein_setup.py.
RS_PERIOD_WEEKS = 52

# Column the chart reads to label its RS panel. A column rather than df.attrs
# because attrs are not reliably preserved across the slicing the display trim
# performs, and a mislabelled benchmark is worse than none.
BENCHMARK_COLUMN = "Benchmark_Symbol"


def macd(close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW,
         signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """MACD line, signal line and histogram.

    EMAs use adjust=False, the recursive form every charting package uses.
    adjust=True would give slightly different early values and so a MACD that
    disagrees with ProRealTime on the same data -- which matters here, because
    these charts get cross-checked against ProRealTime by eye.
    """
    if close is None or close.empty:
        empty = pd.Series(dtype=float)
        return pd.DataFrame({"MACD": empty, "MACD_Signal": empty, "MACD_Hist": empty})

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()

    # Blank the span before `slow` bars: the recursive EMA returns a number
    # from bar one, but it is dominated by the seed value and is not a 26-week
    # average of anything. Drawing it would show a large false divergence at
    # the left edge of every chart.
    warm = min(slow, len(close))
    line.iloc[:warm] = np.nan
    sig.iloc[:warm] = np.nan

    return pd.DataFrame({"MACD": line, "MACD_Signal": sig,
                         "MACD_Hist": line - sig})


def add_display_indicators(signals: pd.DataFrame,
                           benchmark: pd.Series | None = None,
                           benchmark_symbol: str = "",
                           rs_period: int = RS_PERIOD_WEEKS) -> pd.DataFrame:
    """Attach MACD, and RS if the strategy did not already publish it.

    Call on the FULL frame, before the display trim. See the module docstring.
    Returns the same frame (mutated in place and returned for chaining).
    """
    if signals is None or signals.empty or "Close" not in signals.columns:
        return signals

    for column, values in macd(signals["Close"]).items():
        signals[column] = values

    if benchmark_symbol:
        signals[BENCHMARK_COLUMN] = benchmark_symbol

    # Only if absent -- see the module docstring on why this is not recomputed.
    if "Mansfield_RS" not in signals.columns and benchmark is not None:
        try:
            from src.benchmarks import mansfield_rs
            rs = mansfield_rs(signals["Close"], benchmark, n=rs_period)
            signals["Mansfield_RS"] = rs.reindex(signals.index)
        except Exception as e:  # noqa: BLE001 — a missing RS panel must not lose the chart
            log.debug("Mansfield RS unavailable for the chart (%s: %s).",
                      type(e).__name__, e)

    return signals
