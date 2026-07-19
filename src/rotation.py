"""
rotation.py
-----------
Sector selection for Weinstein Stage 1 -> Stage 2 stock breakout hunting.

Weinstein's rule (Chapter 6 of *Secrets for Profiting in Bull and Bear
Markets*) is explicit: buy stocks breaking out of Stage 1 bases in sectors
that are THEMSELVES in Stage 2 with positive Mansfield relative strength.
The sector tailwind is what makes individual breakouts stick; hunting inside
a Stage 3 top or a Stage 4 decline fights the tape.

This module produces exactly one thing: a list of sector ETFs labeled
`hunt`, `watch`, or `avoid`, for consumption by
`market_us.collect_us_by_rotation`.

Two independent checks per sector:

  1. Sector STAGE on its own weekly chart (30W MA + price + prior trend).
     This was missing from the previous version and is what turns "money
     seems to be flowing in" into a Weinstein-valid sector call: a rising
     RS line inside a Stage 4 decline is a countertrend bounce, not an
     opportunity.

  2. Sector Mansfield RS vs the market benchmark (SPY).

Combined:

  hunt   -- sector in Stage 2 AND Mansfield RS > 0
            (textbook Weinstein: leadership confirmed, tailwind in place)

  watch  -- sector in Stage 1 AND Mansfield RS slope > 0
            (sector still basing but money starting to flow in;
            not yet actionable, but earliest component stocks may lead)

  avoid  -- Stage 3, Stage 4, or Stage 1/2 without upward RS momentum

Benchmark: SPY, not ^GSPC. yfinance with `auto_adjust=True` returns total-
return prices for both SPY and the SPDR sector ETFs; ^GSPC is a price index
that excludes dividends, which introduced a systematic ~2%/yr bias favoring
high-yield sectors (XLU, XLP, XLRE, XLE) over low-yield ones (XLK, XLY) in
every Mansfield RS calculation. See `src/benchmarks.py`.

Pure logic: takes a DataLoader, returns dicts. No Flask, no printing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.benchmarks import benchmark_for, get_weekly_close, mansfield_rs

log = logging.getLogger(__name__)

# SPDR sector ETFs. These track the 11 GICS sectors, are total-return under
# yfinance auto_adjust, and are the standard Weinstein-school sector proxies.
SECTOR_ETFS: dict[str, str] = {
    "XLK":  "Technology",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
}


# ------------------------------------------------------------------
# TUNABLES  (defaults; overridable by caller)
# ------------------------------------------------------------------

# Stage detection
DEFAULT_SMA_WEEKS = 30           # Weinstein's 30W MA
DEFAULT_SLOPE_WEEKS = 5          # window for measuring MA slope
DEFAULT_PRIOR_OFFSET_WEEKS = 20  # how far back to sample "prior" trend
DEFAULT_FLAT_SLOPE_PCT = 0.15    # |slope| below this % of MA counts as "flat"

# Relative strength
DEFAULT_RS_PERIOD = 52           # Mansfield RS lookback
DEFAULT_RS_SIGNAL_WEEKS = 10     # RS signal-line MA (for the cross-up flag)
DEFAULT_RS_SLOPE_EPS = 0.05      # |RS slope| below this counts as "flat"

# History / freshness
DEFAULT_HISTORY_WEEKS = 26       # how many weeks of state to backfill


# ------------------------------------------------------------------
# NUMERICAL HELPERS
# ------------------------------------------------------------------

def _regression_slope(values: pd.Series) -> float:
    """
    Least-squares slope of `values` against a plain week index (0, 1, 2, ...).

    Uses every point in the window, unlike a naive (last - first) / n secant
    which is fooled by mid-window spikes that give back to the endpoints.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n)
    y = values.to_numpy(dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


# ------------------------------------------------------------------
# WEINSTEIN STAGE DETECTION on a sector ETF's OWN weekly chart
# ------------------------------------------------------------------

def _sector_stage(weekly_close: pd.Series,
                  sma_weeks: int = DEFAULT_SMA_WEEKS,
                  slope_weeks: int = DEFAULT_SLOPE_WEEKS,
                  prior_offset_weeks: int = DEFAULT_PRIOR_OFFSET_WEEKS,
                  flat_slope_pct: float = DEFAULT_FLAT_SLOPE_PCT) -> str:
    """
    Weinstein 4-stage classification of a sector ETF's own weekly chart:

      stage1  -- basing:    MA flat AND prior MA was clearly falling
      stage2  -- advancing: price > MA AND MA slope clearly positive
      stage3  -- topping:   MA flat AND prior MA was clearly rising
      stage4  -- declining: MA slope clearly negative AND price < MA
      unknown -- insufficient history

    `flat_slope_pct` is expressed as a percentage of the MA level, so the
    "flat" threshold is scale-invariant across sectors at any price level.
    A default of 0.15 means "|slope| < 0.15% of the MA per week".

    Note: Stage 1 vs Stage 3 look identical on a snapshot of the MA
    (both are flat). They are distinguished by what preceded them: Stage 1
    follows a Stage 4 decline, Stage 3 follows a Stage 2 advance. We check
    the slope of the MA `prior_offset_weeks` weeks ago to decide.
    """
    if len(weekly_close) < sma_weeks + prior_offset_weeks + slope_weeks + 1:
        return "unknown"

    ma = weekly_close.rolling(sma_weeks).mean().dropna()
    if len(ma) < prior_offset_weeks + slope_weeks + 1:
        return "unknown"

    price_now = float(weekly_close.iloc[-1])
    ma_now = float(ma.iloc[-1])
    if ma_now <= 0:
        return "unknown"

    # Current MA slope, as % of MA level.
    slope_now = _regression_slope(ma.iloc[-(slope_weeks + 1):])
    slope_now_pct = (slope_now / ma_now) * 100

    # Prior MA slope, ending `prior_offset_weeks` bars ago.
    end = -prior_offset_weeks
    start = end - slope_weeks - 1
    prior_window = ma.iloc[start:end]
    slope_prev = _regression_slope(prior_window)
    ma_prev_level = float(prior_window.iloc[-1])
    slope_prev_pct = (slope_prev / ma_prev_level) * 100 if ma_prev_level > 0 else 0.0

    above_ma = price_now > ma_now

    # Clear trending cases first.
    if slope_now_pct > flat_slope_pct and above_ma:
        return "stage2"
    if slope_now_pct < -flat_slope_pct and not above_ma:
        return "stage4"

    # MA is flat -- disambiguate by the prior trend.
    if slope_prev_pct < -flat_slope_pct:
        return "stage1"  # was falling, now flat = basing
    if slope_prev_pct > flat_slope_pct:
        return "stage3"  # was rising, now flat = topping

    # Prior also flat: extended chop. Fall back on price vs MA.
    return "stage1" if above_ma else "stage4"


# ------------------------------------------------------------------
# RELATIVE STRENGTH metrics vs the market benchmark
# ------------------------------------------------------------------

def _rs_metrics(rs: pd.Series,
                slope_weeks: int = DEFAULT_SLOPE_WEEKS,
                signal_weeks: int = DEFAULT_RS_SIGNAL_WEEKS) -> dict | None:
    """
    Compute RS level, slope and the fresh cross-up flag.
    Returns None if there isn't enough history.
    """
    if len(rs) < max(slope_weeks + 1, signal_weeks + 2):
        return None

    rs_now = float(rs.iloc[-1])
    rs_slope = _regression_slope(rs.iloc[-(slope_weeks + 1):])

    # Weinstein's textbook RS trigger: cross above the RS moving average.
    # Kept as a flag for information; not part of classification below,
    # so the rule stays simple and predictable. Fold it in downstream if
    # you want to relax `hunt` to include stage2 sectors with RS < 0 that
    # just crossed up.
    rs_sig = rs.rolling(signal_weeks).mean()
    crossed_up = bool(
        len(rs_sig.dropna()) >= 2
        and rs.iloc[-1] > rs_sig.iloc[-1]
        and rs.iloc[-2] <= rs_sig.iloc[-2]
    )
    return {
        "rs": round(rs_now, 2),
        "rs_slope": round(rs_slope, 3),
        "rs_crossed_up": crossed_up,
    }


# ------------------------------------------------------------------
# CLASSIFICATION: hunt / watch / avoid
# ------------------------------------------------------------------

_STATE_HUNT = "hunt"
_STATE_WATCH = "watch"
_STATE_AVOID = "avoid"

STATES: tuple[str, ...] = (_STATE_HUNT, _STATE_WATCH, _STATE_AVOID)


def _classify(stage: str, rs: float, rs_slope: float,
              rs_slope_eps: float = DEFAULT_RS_SLOPE_EPS) -> str:
    """
    Weinstein-native mapping:

      hunt   -- sector in Stage 2 AND Mansfield RS > 0
                (leadership confirmed, sector tailwind present)

      watch  -- sector in Stage 1 AND Mansfield RS slope > 0
                (sector basing, money starting to flow in;
                early stocks may lead the sector's own break to Stage 2)

      avoid  -- everything else

    Deliberately strict. Stage 3/4 sectors are never actionable for
    long Stage 1->2 stock breakouts, however positive their RS looks
    (that's just a countertrend bounce). Symmetrically, a Stage 2 sector
    that lags the market (RS <= 0) is a laggard riding the market up, not
    a leader worth hunting inside.
    """
    if stage == "stage2" and rs > 0:
        return _STATE_HUNT
    if stage == "stage1" and rs_slope > rs_slope_eps:
        return _STATE_WATCH
    return _STATE_AVOID


# ------------------------------------------------------------------
# HISTORICAL state series (for freshness / weeks_in_state)
# ------------------------------------------------------------------

def _historical_state_series(weekly_close: pd.Series, rs: pd.Series,
                             tail_weeks: int,
                             sma_weeks: int = DEFAULT_SMA_WEEKS,
                             slope_weeks: int = DEFAULT_SLOPE_WEEKS,
                             prior_offset_weeks: int = DEFAULT_PRIOR_OFFSET_WEEKS,
                             flat_slope_pct: float = DEFAULT_FLAT_SLOPE_PCT,
                             signal_weeks: int = DEFAULT_RS_SIGNAL_WEEKS,
                             rs_slope_eps: float = DEFAULT_RS_SLOPE_EPS,
                             ) -> pd.DataFrame:
    """
    Per-week (stage, rs, rs_slope, state) history. Uses the SAME rules as
    the snapshot path -- a sector's label for a given week is identical
    whether read from a snapshot or from history. Necessary for the
    `weeks_in_state` freshness metric that the sort depends on.

    Cost: O(tail_weeks) polyfits per sector. For 26 weeks x 11 sectors
    this is trivial.
    """
    idx = rs.index.intersection(weekly_close.index)
    tail_weeks = min(tail_weeks, len(idx))
    if tail_weeks <= 0:
        return pd.DataFrame(columns=["stage", "rs", "rs_slope", "state"])

    dates = idx[-tail_weeks:]
    rows = []
    for date in dates:
        px_to = weekly_close.loc[:date]
        rs_to = rs.loc[:date]
        stage = _sector_stage(px_to, sma_weeks, slope_weeks,
                              prior_offset_weeks, flat_slope_pct)
        m = _rs_metrics(rs_to, slope_weeks, signal_weeks)
        if m is None:
            continue
        state = _classify(stage, m["rs"], m["rs_slope"], rs_slope_eps)
        rows.append({
            "date": date,
            "stage": stage,
            "rs": m["rs"],
            "rs_slope": m["rs_slope"],
            "state": state,
        })
    if not rows:
        return pd.DataFrame(columns=["stage", "rs", "rs_slope", "state"])
    return pd.DataFrame(rows).set_index("date")


def _weeks_since_change(states: list[str]) -> int | None:
    """
    Weeks since the last state change in `states` (oldest -> newest).
    Returns None when no change is found within the given window --
    which is genuinely ambiguous between "very stable" and "window too
    short to observe the last flip". Callers can treat None as "at least
    len(states) weeks" for sorting/ranking.
    """
    for i in range(len(states) - 1, 0, -1):
        if states[i] != states[i - 1]:
            return (len(states) - 1) - i
    return None


# ------------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------------

def sector_rotation(loader, *,
                    rs_period: int = DEFAULT_RS_PERIOD,
                    sma_weeks: int = DEFAULT_SMA_WEEKS,
                    slope_weeks: int = DEFAULT_SLOPE_WEEKS,
                    prior_offset_weeks: int = DEFAULT_PRIOR_OFFSET_WEEKS,
                    flat_slope_pct: float = DEFAULT_FLAT_SLOPE_PCT,
                    signal_weeks: int = DEFAULT_RS_SIGNAL_WEEKS,
                    rs_slope_eps: float = DEFAULT_RS_SLOPE_EPS,
                    freshness_lookback_weeks: int = DEFAULT_HISTORY_WEEKS,
                    trail_weeks: int = 4,
                    start_date: str = "2018-01-01") -> list[dict]:
    """
    Classify the 11 SPDR sector ETFs as `hunt`, `watch` or `avoid` for
    Weinstein Stage 1 -> Stage 2 stock breakout scanning.

    Sort order:
      1. hunt before watch before avoid
      2. within each state: FRESHEST first (smallest weeks_in_state)
         -- Weinstein specifically warns against chasing sectors after
         their leaders are extended; a sector that just entered `hunt`
         has stock leaders still near their 30W MA.
      3. tiebreak on RS slope (steeper = stronger momentum).

    Returns a list of dicts, one per sector:
      {
        "etf": "XLE", "sector": "Energy",
        "state": "hunt" | "watch" | "avoid",
        "sector_stage": "stage1" | "stage2" | "stage3" | "stage4" | "unknown",
        "rs": 3.20,
        "rs_slope": 0.150,
        "rs_crossed_up": True,       # RS just crossed its own signal line
        "weeks_in_state": 6,         # None if no change seen in the window
        "trail": [                   # last `trail_weeks` weekly positions
          {"date": "2026-05-30", "rs": -1.10, "rs_slope": 0.02},
          ...
          {"date": "2026-07-18", "rs":  3.20, "rs_slope": 0.15},
        ],
      }
    """
    benchmark_symbol = benchmark_for("US")
    index_wk = get_weekly_close(loader, benchmark_symbol, start_date=start_date)
    if index_wk is None:
        log.error("Could not load benchmark %s for sector rotation.",
                  benchmark_symbol)
        return []

    rows: list[dict] = []
    for etf, sector in SECTOR_ETFS.items():
        stock_wk = get_weekly_close(loader, etf, start_date=start_date)
        if stock_wk is None:
            log.warning("No data for sector ETF %s (%s); skipping.", etf, sector)
            continue

        stage = _sector_stage(stock_wk, sma_weeks, slope_weeks,
                              prior_offset_weeks, flat_slope_pct)

        rs = mansfield_rs(stock_wk, index_wk, n=rs_period).dropna()
        m = _rs_metrics(rs, slope_weeks, signal_weeks)
        if m is None:
            log.warning("Not enough RS history for %s; skipping.", etf)
            continue

        state = _classify(stage, m["rs"], m["rs_slope"], rs_slope_eps)

        # Freshness: how many weeks the current state has held.
        hist = _historical_state_series(
            stock_wk, rs, freshness_lookback_weeks,
            sma_weeks=sma_weeks, slope_weeks=slope_weeks,
            prior_offset_weeks=prior_offset_weeks,
            flat_slope_pct=flat_slope_pct,
            signal_weeks=signal_weeks, rs_slope_eps=rs_slope_eps)
        weeks_in_state = (_weeks_since_change(hist["state"].tolist())
                          if not hist.empty else None)

        # Trail for the money-flow plot: last `trail_weeks` positions in
        # (RS, RS-slope) space. Reuses the historical series computed for
        # weeks_in_state, so no extra data fetch. Bounded above by
        # freshness_lookback_weeks (`hist` never has more than that many
        # rows) -- if you crank trail_weeks past freshness_lookback_weeks
        # you'll silently get the shorter of the two.
        trail: list[dict] = []
        if not hist.empty:
            for tdate, trow in hist.tail(trail_weeks).iterrows():
                trail.append({
                    "date": tdate.strftime("%Y-%m-%d"),
                    "rs": round(float(trow["rs"]), 2),
                    "rs_slope": round(float(trow["rs_slope"]), 3),
                })

        rows.append({
            "etf": etf,
            "sector": sector,
            "state": state,
            "sector_stage": stage,
            "rs": m["rs"],
            "rs_slope": m["rs_slope"],
            "rs_crossed_up": m["rs_crossed_up"],
            "weeks_in_state": weeks_in_state,
            "trail": trail,
        })

    state_rank = {_STATE_HUNT: 0, _STATE_WATCH: 1, _STATE_AVOID: 2}
    rows.sort(key=lambda r: (
        state_rank.get(r["state"], 9),
        # freshest first (None = unknown = treat as stale)
        r["weeks_in_state"] if r["weeks_in_state"] is not None else 999,
        # steeper RS slope wins tiebreak
        -r["rs_slope"],
    ))
    return rows


def rotation_history(loader, *,
                     weeks: int = DEFAULT_HISTORY_WEEKS,
                     rs_period: int = DEFAULT_RS_PERIOD,
                     sma_weeks: int = DEFAULT_SMA_WEEKS,
                     slope_weeks: int = DEFAULT_SLOPE_WEEKS,
                     prior_offset_weeks: int = DEFAULT_PRIOR_OFFSET_WEEKS,
                     flat_slope_pct: float = DEFAULT_FLAT_SLOPE_PCT,
                     signal_weeks: int = DEFAULT_RS_SIGNAL_WEEKS,
                     rs_slope_eps: float = DEFAULT_RS_SLOPE_EPS,
                     start_date: str = "2018-01-01") -> dict:
    """
    hunt/watch/avoid state of every sector over the last `weeks` weeks.

    Same rules as `sector_rotation`, so today's snapshot and today's cell
    in the history table are always identical. Useful for spotting WHEN a
    sector became actionable -- a sector that just flipped to `hunt` is
    a fresh opportunity; one that has been `hunt` for six months means
    its leaders are likely already extended.

    Returns:
      {
        "dates":   ["2026-02-06", ..., "2026-07-17"],   # oldest -> newest
        "sectors": [
          {"etf": "XLE", "sector": "Energy",
           "states":   ["avoid", ..., "hunt"],
           "stages":   ["stage4", ..., "stage2"],
           "current_state": "hunt",
           "current_stage": "stage2",
           "weeks_in_state": 6,
           "dates":  [...],
          },
          ...
        ]
      }
    """
    benchmark_symbol = benchmark_for("US")
    index_wk = get_weekly_close(loader, benchmark_symbol, start_date=start_date)
    if index_wk is None:
        log.error("Could not load benchmark for rotation history.")
        return {"dates": [], "sectors": []}

    dates: list[str] | None = None
    out: list[dict] = []

    for etf, sector in SECTOR_ETFS.items():
        stock_wk = get_weekly_close(loader, etf, start_date=start_date)
        if stock_wk is None:
            continue

        rs = mansfield_rs(stock_wk, index_wk, n=rs_period).dropna()
        hist = _historical_state_series(
            stock_wk, rs, weeks,
            sma_weeks=sma_weeks, slope_weeks=slope_weeks,
            prior_offset_weeks=prior_offset_weeks,
            flat_slope_pct=flat_slope_pct,
            signal_weeks=signal_weeks, rs_slope_eps=rs_slope_eps)
        if hist.empty:
            continue

        states = hist["state"].tolist()
        stages = hist["stage"].tolist()
        these_dates = [d.strftime("%Y-%m-%d") for d in hist.index]
        if dates is None:
            dates = these_dates

        out.append({
            "etf": etf,
            "sector": sector,
            "states": states,
            "stages": stages,
            "current_state": states[-1],
            "current_stage": stages[-1],
            "weeks_in_state": _weeks_since_change(states),
            "dates": these_dates,
        })

    state_rank = {_STATE_HUNT: 0, _STATE_WATCH: 1, _STATE_AVOID: 2}
    out.sort(key=lambda r: (
        state_rank.get(r["current_state"], 9),
        r["weeks_in_state"] if r["weeks_in_state"] is not None else 999,
    ))
    return {"dates": dates or [], "sectors": out}
