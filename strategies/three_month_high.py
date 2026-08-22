"""
three_month_high.py
-------------------
Stocks that made a new 3-month high THIS WEEK.

One rule: this week's high exceeds the highest high of the previous
HIGH_LOOKBACK_WEEKS weeks. Nothing else gates the signal -- no stage filter,
no volume threshold, no relative strength. Everything else this file emits is
a COLUMN, so the screen stays a pure momentum event and the judgement stays
yours.

WHAT "HITTING A 3-MONTH HIGH" MEANS HERE
========================================
The lookback window EXCLUDES the current week. Comparing this week's high
against a window that contains it is trivially satisfied every week, which is
the easiest way to write this screen wrong.

The signal uses the weekly HIGH, the literal reading: the price traded at a
level not seen in a quarter. That admits a spike that closes back down, so
`Close_Is_3M_High` reports the stricter close-based version alongside it --
Weinstein's own breakout test is a weekly CLOSE above resistance, and a new
high the stock could not hold is a different event from one it closed on. Both
are in the output; neither is imposed on you.

WHY THE STAGE COLUMN MATTERS MORE THAN THE SIGNAL
=================================================
A 3-month high is not directional evidence on its own. The same event means
three different things depending on the chart it happens on:

    in Stage 2   continuation. The trend is intact and extending.
    in Stage 1   a base breaking out -- the setup weinstein_setup.py hunts,
                 arriving here from the opposite direction.
    in Stage 4   a bear rally. Price is below a falling 30-week MA and this is
                 a bounce inside a decline, not a turn.
    in Stage 3   a failed top attempt or a genuine resumption; ambiguous.

`Weeks_Since_Prior_High` is the other column to read: a stock making a
quarterly high for the first time in two years is a different animal from one
that prints one every fortnight. Results are sorted on it, longest-dormant
first.

FAILURE MODES
=============
* Fewer than HIGH_LOOKBACK_WEEKS + 1 weekly bars: the lookback is undefined
  and nothing signals. Fail-closed.
* Stage columns need ~82 weekly bars for the 30-week MA and its seed
  reference; below that Stage reads "-" while the signal still works. The
  signal does not depend on the classifier.
* Prices are used as delivered by the loader. A stock that has not been split-
  adjusted will print spurious highs; that is a data problem, not a rule one.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.registry import register
from src.stages import SMA_WEEKS, Stage, classify
from src.strategy import Strategy, StrategyMeta

log = logging.getLogger(__name__)

# CALIBRATION -- not book constants.
HIGH_LOOKBACK_WEEKS = 13    # one quarter of weekly bars = "3 months"
HISTORY_CAP_WEEKS = 260     # how far back to hunt for the previous equal-or-
                            # higher high before reporting the value as capped
YEAR_WEEKS = 52
VOL_AVG_WEEKS = 4           # this week's volume vs the previous 4-week mean


@register
class ThreeMonthHigh(Strategy):

    meta = StrategyMeta(
        key="three_month_high",
        display_name="3-Month High (new quarterly high this week)",
        description=(
            "Stocks whose weekly high this week exceeds the highest high of "
            f"the previous {HIGH_LOOKBACK_WEEKS} weeks. One rule, no filters: "
            "stage, volume, relative strength and distance from the 30-week "
            "MA are reported as columns rather than gates. Read the Stage "
            "column before acting -- the same new high is a continuation in "
            "Stage 2, a base breakout in Stage 1 and a bear rally in Stage 4. "
            "Listed longest-dormant first."
        ),
        signal_column="Signal",
        hit_values=(1,),
        param_schema=(),  # fixed rule: one configuration, run consistently
        display_columns=(
            "Stage", "Weeks_Since_Prior_High", "Close_Is_3M_High",
            "Prior_13W_High", "Pct_Above_Prior_High", "Pct_From_MA30",
            "Pct_Below_52W_High", "Vol_vs_Avg", "Sector", "Industry",
        ),
        sort_by=("Market", "Weeks_Since_Prior_High"),
        sort_ascending=(True, False),   # longest dormant first
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        w = df.resample("W-FRI").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()

        highs = w["High"]
        closes = w["Close"]

        # shift(1) is the whole rule: the window must not contain the bar being
        # tested, or every week is a new high.
        prior_high = highs.shift(1).rolling(HIGH_LOOKBACK_WEEKS).max()
        prior_close_high = closes.shift(1).rolling(HIGH_LOOKBACK_WEEKS).max()

        w["Prior_13W_High"] = prior_high.round(2)
        w["Pct_Above_Prior_High"] = ((highs - prior_high) / prior_high * 100).round(2)
        w["Close_Is_3M_High"] = (closes > prior_close_high).astype(int)

        w["Weeks_Since_Prior_High"] = self._weeks_since_prior_high(highs.to_numpy())

        high_52 = highs.rolling(YEAR_WEEKS).max()
        with np.errstate(divide="ignore", invalid="ignore"):
            w["Pct_Below_52W_High"] = ((high_52 - closes) / high_52 * 100).round(2)

        vol_avg = w["Volume"].shift(1).rolling(VOL_AVG_WEEKS).mean()
        w["Vol_vs_Avg"] = (w["Volume"] / vol_avg).round(2)

        # Stage context. Delegated entirely to src/stages.py -- this file
        # decides nothing about stages, and never gates on them.
        st = classify(closes)
        w["Stage_Num"] = st["stage"].to_numpy()
        w["Stage"] = [Stage(int(v)).short for v in st["stage"]]
        w["SMA_30W"] = st["ma"]
        w["SMA_10W"] = st["ma_fast"]     # so the plotter draws both lines
        ma = st["ma"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            w["Pct_From_MA30"] = np.round(
                (closes.to_numpy() - ma) / ma * 100, 2)

        w["Signal"] = np.where(
            highs.to_numpy() > prior_high.to_numpy(), 1, 0)

        return w

    @staticmethod
    def _weeks_since_prior_high(highs: np.ndarray) -> np.ndarray:
        """Weeks back to the most recent bar whose high was >= this bar's.

        Bounded at HISTORY_CAP_WEEKS: a stock at an all-time high would
        otherwise report a number that says more about how much history the
        loader returned than about the stock. A capped value therefore means
        "at least this long", and the cap is a constant you can read.

        Returns 0 where no prior bar is available at all.
        """
        n = len(highs)
        out = np.zeros(n, dtype=int)
        for i in range(1, n):
            h = highs[i]
            if not np.isfinite(h):
                continue
            start = max(0, i - HISTORY_CAP_WEEKS)
            back = 0
            for j in range(i - 1, start - 1, -1):
                back += 1
                if np.isfinite(highs[j]) and highs[j] >= h:
                    break
            else:
                back = i - start          # never exceeded within the window
            out[i] = back
        return out
