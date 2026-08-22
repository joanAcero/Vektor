"""
weinstein_simple.py
-------------------
The smallest possible Weinstein scan: securities that ARE in Stage 1 and have
been for at least MIN_STAGE1_WEEKS consecutive weeks.

That is the entire rule. No stabilization scan, no swing clusters, no prior
decline measurement, no relative strength, no volume character -- all of that
lives in weinstein_setup.py, which answers a harder question (is this a valid
base, and how far is it from breaking out?). This file answers the easy one:
what has the stage classifier been calling a base for a while?

WHY BOTH EXIST
==============
They fail differently, which is the point of running them side by side.

  weinstein_setup   measures the base directly -- a stabilization band, a
                    prior decline, resistance touches. Precise when the
                    structure resolves, silent when it does not. It can miss a
                    real base whose geometry it cannot fit.
  weinstein_simple  asks only what src/stages.py says. It cannot miss anything
                    the classifier calls Stage 1, and it inherits every one of
                    the classifier's errors -- including the churn on lateral
                    charts documented in src/stages.py. Where the two disagree
                    is where one of them is wrong, and that is worth looking at.

Because it delegates entirely, this strategy is also the fastest way to see a
change to classify() land on real names: it has no rules of its own to mask it.

NO PARAMETERS
=============
MIN_STAGE1_WEEKS is a constant, not a tunable, matching the house rule for
every Weinstein strategy here: one configuration, run consistently, so results
from different days are comparable. Ten weeks is a quarter -- long enough that
a two-month pause in a decline does not qualify, short enough that early bases
(UBER ~16w, SONY ~18w) still clear it with margin.

FAILURE MODES
=============
* Fewer than ~82 weekly bars: the 30-week MA and its 52-week seed reference
  are undefined, classify() returns UNKNOWN, and nothing signals. Fail-closed.
* A lateral chart whose stage flickers will report a small Weeks_In_Stage and
  drop out of the results, even though it has been sideways for years. That is
  the classifier's churn showing through, not a bug here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.registry import register
from src.stages import (MA_FAST_WEEKS, SMA_WEEKS, Stage, classify,
                        weeks_in_stage)
from src.strategy import Strategy, StrategyMeta

log = logging.getLogger(__name__)

MIN_STAGE1_WEEKS = 10   # a quarter of basing. Fixed, not tunable.


@register
class WeinsteinSimple(Strategy):

    meta = StrategyMeta(
        key="weinstein_simple",
        display_name=f"Weinstein Stage 1 ({MIN_STAGE1_WEEKS}+ weeks)",
        description=(
            "Everything the stage classifier currently calls Stage 1, held "
            f"for at least {MIN_STAGE1_WEEKS} consecutive weeks. One rule, "
            "delegated entirely to classify() in src/stages.py -- no "
            "stabilization scan, no support/resistance, no relative strength. "
            "It cannot miss a base the classifier sees, and it inherits every "
            "error the classifier makes. Run it beside the full Weinstein "
            "setup: the names one finds and the other does not are the ones "
            "worth opening a chart on. Listed longest-basing first."
        ),
        signal_column="Signal",
        hit_values=(1,),
        param_schema=(),  # fixed rule: one configuration, run consistently
        display_columns=(
            "Stage", "Weeks_In_Stage", "Pct_From_MA30", "MA30_Slope_Pct",
            "Base_High", "Base_Low", "Range_Width_Pct", "Pct_To_Base_High",
            "Sector", "Industry",
        ),
        sort_by=("Market", "Weeks_In_Stage"),
        sort_ascending=(True, False),   # longest base first
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        w = df.resample("W-FRI").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()

        # THE rule set. Nothing about stages is decided in this file.
        st = classify(w["Close"])
        stage = st["stage"].to_numpy()

        w["Stage_Num"] = stage
        w["Stage"] = [Stage(int(v)).short for v in stage]
        w["SMA_30W"] = st["ma"]
        w["SMA_10W"] = st["ma_fast"]          # so the plotter draws both lines
        w["MA30_Slope_Pct"] = st["slope_pct"].round(2)
        w["Weeks_In_Stage"] = weeks_in_stage(stage)

        ma = st["ma"].to_numpy()
        close = w["Close"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            w["Pct_From_MA30"] = np.round((close - ma) / ma * 100, 2)

        hi, lo = self._run_extremes(w["High"].to_numpy(), w["Low"].to_numpy(),
                                    w["Weeks_In_Stage"].to_numpy())
        w["Base_High"] = np.round(hi, 2)
        w["Base_Low"] = np.round(lo, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            w["Range_Width_Pct"] = np.round((hi - lo) / lo * 100, 2)
            # How far the top of the current base is above price. This is a
            # geometric distance, NOT a breakout level: nothing here tests
            # whether that high is real resistance. weinstein_setup does.
            w["Pct_To_Base_High"] = np.round((hi - close) / close * 100, 2)

        w["Signal"] = np.where(
            (stage == int(Stage.ONE)) &
            (w["Weeks_In_Stage"].to_numpy() >= MIN_STAGE1_WEEKS), 1, 0)

        return w

    @staticmethod
    def _run_extremes(highs: np.ndarray, lows: np.ndarray,
                      run: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Highest high and lowest low over the current stage run at each bar.

        The window is the run itself, so the range describes THIS base and not
        whatever preceded it -- the same reasoning as the cluster window in
        weinstein_setup.
        """
        n = len(run)
        hi = np.full(n, np.nan)
        lo = np.full(n, np.nan)
        for i in range(n):
            k = int(run[i])
            if k < 2:
                continue
            hi[i] = float(np.nanmax(highs[i - k + 1:i + 1]))
            lo[i] = float(np.nanmin(lows[i - k + 1:i + 1]))
        return hi, lo
