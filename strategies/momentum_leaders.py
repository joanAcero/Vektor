"""
momentum_leaders.py
-------------------
"Follow the money" strategy: stocks in a confirmed Stage-2 uptrend that are
LEADING the market (positive Mansfield relative strength) but are NOT yet
overextended above their 30-week MA.

Rationale
---------
Raw momentum screens surface names that have already run far above their mean
and carry high reversion risk. This strategy deliberately balances two opposing
forces:

  * STRENGTH  — high Mansfield RS vs the market index (outperforming: this is
                where the money is flowing).
  * RESTRAINT — price sits in a healthy band above a RISING 30W MA, not far
                extended, so we join the trend rather than chase a blow-off.

It is the natural complement to the pre-breakout Weinstein screener: that one
finds bases about to start Stage 2; this one finds Stage 2 trends still worth
following. Detection only — it does not trade.

Benchmark note
--------------
Mansfield RS needs the market index. Like WeinsteinSetup, this strategy receives
it via `set_benchmark()`, which the Screener injects per market. Without a
benchmark the RS column is NaN and the RS filter cannot pass, so the strategy
yields nothing rather than emitting misleading signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import ParamSpec, Strategy, StrategyMeta
from src.registry import register
from src.benchmarks import mansfield_rs


@register
class MomentumLeaders(Strategy):

    meta = StrategyMeta(
        key="momentum_leaders",
        display_name="Momentum Leaders (RS + healthy trend)",
        description=(
            "Finds market leaders to follow the money: stocks outperforming the "
            "index (positive Mansfield relative strength) that are in a confirmed "
            "uptrend — price above a RISING 30W MA — but NOT overextended (within "
            "a set %% above the MA). Balances strength against reversion risk; the "
            "complement to the pre-breakout screener. Detection only."
        ),
        signal_column="Signal",
        hit_values=(1,),
        param_schema=(
            ParamSpec("sma_weeks", 30, int, "Weeks for the trend MA (Weinstein uses 30)"),
            ParamSpec("slope_lookback", 5, int,
                      "Bars used to measure MA slope (must be rising)"),
            ParamSpec("max_dist_above_ma", 20.0, float,
                      "Max %% the close may sit ABOVE the MA (anti-overextension cap)"),
            ParamSpec("min_dist_above_ma", 0.0, float,
                      "Min %% above the MA (0 = just needs to be above it)"),
            ParamSpec("min_rs", 0.0, float,
                      "Min Mansfield RS (0 = merely outperforming; raise to be stricter)"),
            ParamSpec("rs_period", 52, int, "Mansfield RS lookback (weeks)"),
        ),
        # RS_Rank is filled by the Screener/consumer if desired; we expose the
        # raw metrics needed to sort "where the money is".
        display_columns=(
            "Mansfield_RS", "Dist_Above_MA_Pct", "MA_Slope_Pct",
            "Sector", "Industry",
        ),
        # Strongest leaders first: sort by RS descending.
        sort_by=("Market", "Mansfield_RS"),
        sort_ascending=(True, False),
    )

    # ---- param aliases ----------------------------------------------------
    @property
    def sma_period(self) -> int:
        return self.params["sma_weeks"]

    # ---- benchmark injection (mirrors WeinsteinSetup) ---------------------
    def __init__(self, **params):
        super().__init__(**params)
        self._benchmark_weekly = None  # set per-market by the Screener

    def set_benchmark(self, weekly_close) -> None:
        """Receive the market's benchmark weekly close series (for Mansfield RS)."""
        self._benchmark_weekly = weekly_close

    # ---- core -------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        logic = {"Open": "first", "High": "max", "Low": "min",
                 "Close": "last", "Volume": "sum"}
        w = df.resample("W-FRI").agg(logic).dropna()

        p = self.params
        w["MA"] = w["Close"].rolling(p["sma_weeks"]).mean()
        w["MA_Slope"] = w["MA"].diff(p["slope_lookback"])
        # Slope as a percentage of the MA level, so the "rising" test is
        # scale-invariant across price levels.
        w["MA_Slope_Pct"] = (w["MA_Slope"] / w["MA"]) * 100

        # Distance of price above the MA, in percent (negative = below MA).
        w["Dist_Above_MA_Pct"] = ((w["Close"] - w["MA"]) / w["MA"]) * 100

        # Mansfield relative strength vs the injected benchmark.
        w["Mansfield_RS"] = self._mansfield(w)

        # ----- conditions -----
        cond_ma_rising = w["MA_Slope"] > 0
        # Healthy band ABOVE the MA: confirms Stage 2 without chasing extension.
        cond_band = (w["Dist_Above_MA_Pct"] >= p["min_dist_above_ma"]) & \
                    (w["Dist_Above_MA_Pct"] <= p["max_dist_above_ma"])
        # Leadership: outperforming the index. NaN (no benchmark) must NOT pass,
        # otherwise the strategy would emit signals without its defining filter.
        cond_rs = w["Mansfield_RS"] >= p["min_rs"]

        # Per-condition diagnostics (consumed by the Screener's pass-count log).
        conds = {
            "ma_rising": cond_ma_rising,
            "healthy_band": cond_band,
            "rs_leading": cond_rs,
        }
        for name, series in conds.items():
            w[f"Cond_{name}"] = series.fillna(False).astype(bool)

        signal = cond_ma_rising & cond_band & cond_rs
        w["Signal"] = np.where(signal.fillna(False), 1, 0)
        return w

    # ---- helpers ----------------------------------------------------------
    def _mansfield(self, w: pd.DataFrame) -> pd.Series:
        if self._benchmark_weekly is None:
            return pd.Series(np.nan, index=w.index)
        mrs = mansfield_rs(w["Close"], self._benchmark_weekly, n=self.params["rs_period"])
        return mrs.reindex(w.index)
