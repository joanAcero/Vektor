"""
browse_all.py
-------------
The no-filter strategy: every name in the selected target passes, so the run
becomes "chart the whole universe" rather than "find the setups in it".

WHY THIS IS A STRATEGY AND NOT A MODE
=====================================
The alternative was a "no strategy" checkbox somewhere in run.py, which would
mean a branch bypassing generate_signals(), a second path around the Screener,
and a results table with no columns to sort by. Implementing it as a Strategy
that returns Signal=1 for everything costs one file and touches nothing:
config, the Screener, the sort menu, the sector census and the chart viewer all
work because from their side this is an ordinary strategy that happens to be
very unselective.

WHAT IT IS FOR
==============
Reviewing a universe by eye, which is the thing the detectors cannot do. Two
concrete uses:
  * Calibration. Run it over a target, page through, and see what the Weinstein
    detector WOULD have missed -- false negatives are invisible by definition
    when you only ever look at what a filter returned.
  * Small targets. On the ~30 names of the Dow, or one Finviz industry, a
    filter is beside the point; you want to look at all of them.

COST, STATED PLAINLY
====================
There is no filter, so the chart pass renders one PNG per name in the target.
On the Dow that is 30 and takes seconds. On the S&P 500 it is 500, on the
combined index selection it is over a thousand, and the run will take many
minutes and fill results/ with images. This is not a bug to be fixed with a cap
-- a silent cap would make "show me everything" quietly untrue -- but it is the
reason to point this strategy at a narrow target.

WHAT IT PUBLISHES
=================
No filter still means useful ORDERING, which is where the value is once you
have a hundred charts: the sort menu can only offer columns the strategy emits.
So it publishes the three that make a universe navigable --

    Stage           Weinstein stage of the name's own weekly chart
    Dist_MA30_Pct   how far the close sits from the 30-week MA
    Mansfield_RS    relative strength vs the market benchmark

-- plus Stage_Rank, which is not displayed but drives the default ordering
(Stage 2 first, then 1, 3, 4, and inside each by RS). That default is the
hunting order the sector monitor already uses.

Every one of the three degrades to blank rather than failing: no benchmark
means no RS, and a name with under 30 weeks of history has no meaningful MA. A
universe browser that refuses to show a recent IPO would be missing the point.

WHY THE STAGE HELPER IS SHAPED THE WAY IT IS
============================================
The first version of this file assumed classify() returned a Series and wrapped
only the CALL in a try. classify() returns something two-dimensional, so the
failure landed in the normalisation code just outside the guard and took out
every ticker in the scan with `ValueError: 2` -- 35 of 35 errored.

That is the lesson worth keeping: a fallback that only catches the failure mode
you predicted is not a fallback. _stages() below therefore (a) wraps the call
AND the coercion in one guard, and (b) accepts a DataFrame, Series, ndarray or
list, because this module does not own the stage contract and should not
assume its shape. If src/stages.py changes shape again, the Stage column goes
blank and the browse still works.

The fallback is a BLANK column, never a locally computed lookalike.
src/stages.py is the single authority for stage logic; a second implementation
here would silently disagree with the sector monitor, the Weinstein charts and
the Telegram reports, which is a far worse outcome than an empty column.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.registry import register
from src.strategy import ParamSpec, Strategy, StrategyMeta

log = logging.getLogger(__name__)

SMA_WEEKS = 30

# Column names to look for when classify() hands back a table rather than a
# single series. Most specific first.
_STAGE_COLUMN_CANDIDATES = ("stage", "stage_slug", "stage_label", "label")

# Fallback ordering used ONLY when hunt_rank() itself cannot answer. Mirrors
# hunt_rank() in src/stages.py: advance, base, top, decline, unknown. When
# stages.py answers, its ranking is used and this is not consulted.
_FALLBACK_RANK = {"stage2": 0, "stage1": 1, "stage3": 2, "stage4": 3, "unknown": 9}

_BLANK_RANK = 9.0


def _weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLCV -> weekly, Friday-anchored, as every other strategy does."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in out.columns:
        agg["Volume"] = "sum"
    return out.resample("W-FRI").agg(agg).dropna(subset=["Close"])


def _as_label_series(labels, index: pd.Index) -> pd.Series:
    """Whatever classify() returned -> a one-dimensional Series of labels.

    Deliberately shape-agnostic. This module consumes the stage contract but
    does not define it, so it handles the plausible shapes rather than encoding
    an assumption that will break the next time stages.py evolves. Raises on
    anything it cannot reduce, so the caller's guard turns it into a blank
    column instead of a crash.
    """
    if isinstance(labels, pd.DataFrame):
        lowered = {str(c).strip().lower(): c for c in labels.columns}
        column = next((lowered[c] for c in _STAGE_COLUMN_CANDIDATES if c in lowered), None)
        if column is None:
            if labels.shape[1] != 1:
                raise ValueError(
                    f"classify() returned a {labels.shape[1]}-column frame with no "
                    f"recognisable stage column (got {list(labels.columns)})")
            column = labels.columns[0]
        series = labels[column]
    elif isinstance(labels, pd.Series):
        series = labels
    else:
        array = np.asarray(labels, dtype=object)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError(f"classify() returned a {array.ndim}-D array")
        series = pd.Series(array)

    if len(series) != len(index):
        raise ValueError(
            f"classify() returned {len(series)} labels for {len(index)} bars")

    # Reuse the caller's index rather than the returned one: they should agree,
    # and if they do not, the bars are what this frame is keyed on.
    return pd.Series(series.to_numpy(), index=index, dtype=object)


def _label_text(value) -> str:
    """One stage label -> the slug form the rest of the system displays.

    Three cases, in order. A `.slug` attribute wins, since that is stages.py's
    own rendering. Failing that, a bare 1-4 becomes "stage2" -- necessary
    because pandas coerces a Series of Stage IntEnum values to int64, which
    strips the enum (and its .slug) before this code ever sees it, leaving a
    Stage column reading "2" while every other stage display in the system
    reads "stage2". Anything else is passed through as text.
    """
    if value is None:
        return ""
    slug = getattr(value, "slug", None)
    if slug:
        return str(slug)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    return f"stage{number}" if 1 <= number <= 4 else str(value)


def _stages(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(stage label, hunt rank) per bar, from src/stages.py when it will answer.

    One guard around the import, the call AND the coercion — see the module
    docstring for why splitting them was the original bug.
    """
    blank = (pd.Series("", index=close.index, dtype=object),
             pd.Series(_BLANK_RANK, index=close.index, dtype=float))
    try:
        from src.stages import classify, hunt_rank

        labels = _as_label_series(classify(close), close.index)
        text = labels.map(_label_text)

        def _rank(value) -> float:
            try:
                return float(hunt_rank(value))
            except Exception:  # noqa: BLE001 — fall back per value, not per scan
                return float(_FALLBACK_RANK.get(str(value), _BLANK_RANK))

        return text, labels.map(_rank).astype(float)

    except ImportError:
        log.debug("src.stages unavailable; Stage column left blank.")
        return blank
    except Exception as e:  # noqa: BLE001 — a stages.py change must not break browsing
        # Warning, not debug: a silently blank Stage column would look like a
        # data problem rather than a contract mismatch, and this file cannot
        # fix a contract mismatch on its own.
        log.warning("Stage classification unusable (%s: %s); Stage left blank. "
                    "If this persists, src/stages.py::classify() has changed "
                    "shape and _as_label_series() in this file needs updating.",
                    type(e).__name__, e)
        return blank


@register
class BrowseAll(Strategy):

    meta = StrategyMeta(
        key="browse_all",
        display_name="Browse all (no filter)",
        description=(
            "No filter: every name in the selected target is returned and "
            "charted. For reviewing a universe by eye — checking what the "
            "detectors are missing, or working through a small target like the "
            "Dow or a single industry where filtering is beside the point. "
            "Charts are ordered Stage 2 → 1 → 3 → 4, then by relative strength, "
            "and the sort menu can reorder them by any column. Note that on a "
            "large target this renders one chart per constituent and will take "
            "several minutes."
        ),
        signal_column="Signal",
        hit_values=(1,),
        # Deliberately parameter-free. Any parameter here would be a filter,
        # and a filtering "no filter" strategy is a contradiction. Narrow the
        # TARGET instead — that is the control this strategy pairs with.
        param_schema=(),
        display_columns=("Stage", "Dist_MA30_Pct", "Mansfield_RS"),
        # Same hunting order as the sector monitor: Stage 2 first, then 1, 3, 4,
        # and within each by relative strength descending.
        sort_by=("Stage_Rank", "Mansfield_RS"),
        sort_ascending=(True, False),
    )

    def __init__(self, **params):
        super().__init__(**params)
        self._benchmark: pd.Series | None = None

    def set_benchmark(self, benchmark: pd.Series | None) -> None:
        """Weekly benchmark closes, injected by the Screener per market.

        Same hook WeinsteinSetup and MomentumLeaders use. Without it the RS
        column is blank and the ordering falls back to stage alone — RS is a
        convenience for sorting here, never a gate.
        """
        self._benchmark = benchmark

    def _relative_strength(self, close: pd.Series) -> pd.Series:
        nan = pd.Series(np.nan, index=close.index, dtype=float)
        if self._benchmark is None:
            return nan
        try:
            from src.benchmarks import mansfield_rs
            rs = mansfield_rs(close, self._benchmark)
            # Same defensive coercion as the stage column, and for the same
            # reason: this module consumes the contract, it does not own it.
            rs = pd.Series(np.asarray(rs, dtype=float).ravel())
            if len(rs) != len(close):
                raise ValueError(f"{len(rs)} RS values for {len(close)} bars")
            return pd.Series(rs.to_numpy(), index=close.index, dtype=float)
        except Exception as e:  # noqa: BLE001
            log.debug("Mansfield RS unavailable (%s: %s).", type(e).__name__, e)
            return nan

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        weekly = _weekly(df)
        if weekly.empty:
            return weekly

        close = weekly["Close"]
        # min_periods=1 so a name with less than 30 weeks of history still gets
        # a line and still appears. The MA is meaningless over the first weeks
        # and the chart shows that plainly, which is the correct outcome for a
        # browser: excluding a recent listing would hide it entirely.
        weekly["SMA_30W"] = close.rolling(SMA_WEEKS, min_periods=1).mean()
        weekly["Dist_MA30_Pct"] = (close / weekly["SMA_30W"] - 1.0) * 100.0

        weekly["Stage"], weekly["Stage_Rank"] = _stages(close)
        weekly["Mansfield_RS"] = self._relative_strength(close)

        # The whole point: everything passes.
        weekly["Signal"] = 1
        return weekly
