"""
rotation.py
-----------
Money-flow / sector-rotation monitor.

The idea (Weinstein): a stock's sector explains a large share of its move, so
you want to be hunting breakouts INSIDE sectors that money is rotating INTO —
ideally BEFORE that rotation is obvious and the leaders are already extended.

Most people watch "is the sector RS > 0 now?". That is late: by the time RS
crosses zero, much of the move has happened. This module measures three things
per sector so you can catch the turn early:

  1. rs        — current Mansfield RS vs the S&P 500 (level: leading or lagging?)
  2. rs_slope  — slope of RS over recent weeks (direction: rotating in or out?)
  3. crossed_up— RS recently crossed its own short average (the turn itself)

A sector with rs still slightly negative but a strongly positive rs_slope is
rotating INTO leadership — that is where your breakouts are still near the MA.

Two levels:
  * sector_rotation()   — the 11 SPDR sector ETFs vs SPY/^GSPC.
  * industry_rotation() — Finviz industries within a chosen sector, ranked by
                          recent relative performance (uses FinvizEngine).

Pure-ish logic: it takes a DataLoader (for prices) and, for the industry view,
a FinvizEngine. No Flask, no printing — callers format the output.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.benchmarks import get_weekly_close, mansfield_rs

log = logging.getLogger(__name__)

# SPDR sector ETFs. These track the 11 GICS sectors and have clean price data,
# which is where Mansfield RS behaves best.
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

_SP500 = "^GSPC"


def _rs_metrics(stock_wk: pd.Series, index_wk: pd.Series,
                rs_period: int, slope_weeks: int, signal_weeks: int) -> dict | None:
    """
    Compute RS level, slope, recent change and a cross-up flag for one symbol.
    Returns None if there isn't enough overlapping history.
    """
    rs = mansfield_rs(stock_wk, index_wk, n=rs_period)
    rs = rs.dropna()
    if len(rs) < max(slope_weeks, signal_weeks) + 1:
        return None

    rs_now = float(rs.iloc[-1])
    rs_prev = float(rs.iloc[-1 - slope_weeks])
    # Slope: average RS change per week over the window (units: RS points/week).
    rs_slope = (rs_now - rs_prev) / slope_weeks

    # Short average of RS; a cross above it marks the turn.
    rs_sig = rs.rolling(signal_weeks).mean()
    crossed_up = bool(
        len(rs_sig.dropna()) >= 2
        and rs.iloc[-1] > rs_sig.iloc[-1]
        and rs.iloc[-2] <= rs_sig.iloc[-2]
    )
    # 4-week RS change, a quick "acceleration" read.
    rs_change_4w = float(rs_now - rs.iloc[-5]) if len(rs) >= 5 else np.nan

    return {
        "rs": round(rs_now, 2),
        "rs_slope": round(rs_slope, 3),
        "rs_change_4w": round(rs_change_4w, 2) if np.isfinite(rs_change_4w) else None,
        "crossed_up": crossed_up,
    }


def _classify(rs: float, rs_slope: float, slope_eps: float = 0.05) -> str:
    """
    Turn (level, slope) into a human rotation state. The valuable quadrant for
    an early breakout hunter is 'Rotating In' — not yet leading, but turning up
    with a MEANINGFUL slope (flat noise near zero is treated as neutral, not a
    rotation signal). slope_eps sets how steep the RS must move to count.
    """
    leading = rs > 0
    rising = rs_slope > slope_eps
    falling = rs_slope < -slope_eps
    if leading and rising:
        return "Leading"          # already strong AND getting stronger (often late)
    if not leading and rising:
        return "Rotating In"      # <-- the early signal: money starting to flow in
    if leading and falling:
        return "Weakening"        # still strong but losing steam (rotating out)
    if not leading and falling:
        return "Lagging"          # weak and getting weaker (avoid)
    return "Neutral"              # flat: no clear rotation either way


def sector_rotation(loader, rs_period: int = 52, slope_weeks: int = 5,
                    signal_weeks: int = 10, start_date: str = "2018-01-01") -> list[dict]:
    """
    Rank the 11 SPDR sector ETFs by rotation strength vs the S&P 500.

    Returns a list of dicts (one per sector), sorted so the sectors rotating
    IN / leading with the strongest slope come first. Each dict has:
      etf, sector, rs, rs_slope, rs_change_4w, crossed_up, state.
    """
    index_wk = get_weekly_close(loader, _SP500, start_date=start_date)
    if index_wk is None:
        log.error("Could not load S&P 500 (%s) for rotation.", _SP500)
        return []

    rows: list[dict] = []
    for etf, sector in SECTOR_ETFS.items():
        stock_wk = get_weekly_close(loader, etf, start_date=start_date)
        if stock_wk is None:
            log.warning("No data for sector ETF %s (%s); skipping.", etf, sector)
            continue
        m = _rs_metrics(stock_wk, index_wk, rs_period, slope_weeks, signal_weeks)
        if m is None:
            continue
        m.update({"etf": etf, "sector": sector,
                  "state": _classify(m["rs"], m["rs_slope"])})
        rows.append(m)

    # Sort: rotating-in first (the early opportunity), then leaders, then the
    # rest. Within a state, steeper slope first so the strongest turns bubble up.
    state_rank = {"Rotating In": 0, "Leading": 1, "Neutral": 2,
                  "Weakening": 3, "Lagging": 4}
    rows.sort(key=lambda r: (state_rank.get(r["state"], 9),
                             -r["rs_slope"], -r["rs"]))
    return rows


def rotation_history(loader, weeks: int = 26, rs_period: int = 52,
                     slope_weeks: int = 5, slope_eps: float = 0.05,
                     start_date: str = "2018-01-01") -> dict:
    """
    Rotation state of every sector over the LAST `weeks` weeks.

    The single-snapshot view tells you where money is today; this tells you WHEN
    each sector changed state. Seeing "Energy flipped to Rotating In six weeks
    ago and has stayed there" is far more actionable than today's label alone —
    it shows whether a rotation is fresh (still early, leaders near their MA) or
    long in the tooth.

    Returns:
      {
        "dates":   ["2026-02-06", ...],                 # oldest -> newest
        "sectors": [
           {"etf": "XLE", "sector": "Energy",
            "rs":     [ ... one per date ... ],
            "slope":  [ ... ],
            "states": ["Lagging", ..., "Rotating In"],
            "changed_weeks_ago": 6 or None},            # weeks since last flip
           ...
        ]
      }
    """
    index_wk = get_weekly_close(loader, _SP500, start_date=start_date)
    if index_wk is None:
        log.error("Could not load S&P 500 for rotation history.")
        return {"dates": [], "sectors": []}

    dates: list[str] | None = None
    out: list[dict] = []

    for etf, sector in SECTOR_ETFS.items():
        stock_wk = get_weekly_close(loader, etf, start_date=start_date)
        if stock_wk is None:
            continue

        rs = mansfield_rs(stock_wk, index_wk, n=rs_period).dropna()
        if len(rs) < slope_weeks + 2:
            continue

        # Rolling slope over the whole series, then take the last `weeks` points.
        slope = (rs - rs.shift(slope_weeks)) / slope_weeks
        hist = pd.DataFrame({"rs": rs, "slope": slope}).dropna().tail(weeks)
        if hist.empty:
            continue

        states = [_classify(float(r), float(s), slope_eps)
                  for r, s in zip(hist["rs"], hist["slope"])]

        # How many weeks since the state last changed? (fresh rotation vs stale)
        changed_weeks_ago = None
        for i in range(len(states) - 1, 0, -1):
            if states[i] != states[i - 1]:
                changed_weeks_ago = (len(states) - 1) - i
                break

        these_dates = [d.strftime("%Y-%m-%d") for d in hist.index]
        if dates is None:
            dates = these_dates

        out.append({
            "etf": etf,
            "sector": sector,
            "rs": [round(float(v), 2) for v in hist["rs"]],
            "slope": [round(float(v), 3) for v in hist["slope"]],
            "states": states,
            "current_state": states[-1],
            "changed_weeks_ago": changed_weeks_ago,
            "dates": these_dates,
        })

    # Freshest rotations first: a sector that JUST flipped to Rotating In is the
    # highest-value signal for an early breakout hunter.
    state_rank = {"Rotating In": 0, "Leading": 1, "Neutral": 2,
                  "Weakening": 3, "Lagging": 4}
    out.sort(key=lambda r: (state_rank.get(r["current_state"], 9),
                            r["changed_weeks_ago"] if r["changed_weeks_ago"] is not None else 999))

    return {"dates": dates or [], "sectors": out}


def industry_rotation(finviz, sector_name: str, top_n: int = 0,
                      col_target: str = "Perf Quart") -> list[dict]:
    """
    Within a sector that is rotating in, find the strongest INDUSTRIES using
    Finviz group performance. This narrows the hunt to the specific corner of
    the sector leading the move.

    top_n = 0 (default) returns ALL industries, ranked strongest first.
    `col_target` is the Finviz performance column to rank by.
    """
    try:
        # A large cap effectively means "all" (Finviz has ~150 industries).
        limit = top_n if top_n and top_n > 0 else 500
        industries = finviz.get_top_industries(top_n=limit, col_target=col_target)
    except Exception as e:  # noqa: BLE001
        log.error("Finviz industry rotation failed: %s", e)
        return []
    return [{"industry": name, "rank": i + 1} for i, name in enumerate(industries)]
