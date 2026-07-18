"""
rotation.py
-----------
Money-flow / sector-rotation monitor.

The idea (Weinstein): a stock's sector explains a large share of its move, so
you want to be hunting breakouts INSIDE sectors that money is rotating INTO —
ideally BEFORE that rotation is obvious and the leaders are already extended.

Most people watch "is the sector RS > 0 now?". That is late: by the time RS
crosses zero, much of the move has happened. This module measures FOUR things
per sector so you can catch the turn early:

  1. rs                — current Mansfield RS vs the S&P 500 (level: leading or lagging?)
  2. rs_slope           — least-squares slope of RS over recent weeks (direction)
  3. rs_accel           — change in slope vs the PRIOR window (is the move
                          steepening or losing steam -- not just positive)
  4. crossed_up         — RS just crossed its own signal line THIS week (the
                          discrete trigger moment, not a continuous trend read)
  + changed_weeks_ago   — how long the CURRENT state has held (a sector that
                          flipped to Rotating In yesterday and one that's been
                          there for four months look identical on rs/rs_slope
                          alone; this is what tells them apart)

A sector with rs still slightly negative but a strongly positive rs_slope is
rotating INTO leadership. A positive rs_accel on top of that means it's not
just rotating in, it's accelerating while doing so.

Two levels:
  * sector_rotation()   — the 11 SPDR sector ETFs vs SPY/^GSPC.
  * industry_rotation() — Finviz industries within a chosen sector, ranked by
                          recent relative performance (uses FinvizEngine).
                          NOTE: this one currently ignores sector_name entirely
                          (a pre-existing bug) -- see market_us.py's
                          collect_us_by_rotation() docstring. Not touched here.

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


def _regression_slope(values: pd.Series) -> float:
    """
    Least-squares slope of `values` against a plain week index (0, 1, 2, ...).

    Replaces a naive (last - first) / n secant, which only looks at the two
    ENDPOINTS of the window -- a sector that spiked mid-window and gave it all
    back reads identically to one that moved smoothly, as long as the start
    and end points happen to match. A regression uses every point.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n)
    y = values.to_numpy(dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _rs_metrics(rs: pd.Series, slope_weeks: int, signal_weeks: int) -> dict | None:
    """
    Compute RS level, slope, acceleration, recent change and a cross-up flag.
    `rs` is the FULL historical Mansfield RS series for one symbol (already
    dropna'd) -- computed once by the caller and shared with the freshness
    calculation, rather than recomputed per metric.
    Returns None if there isn't enough overlapping history.
    """
    # Need enough history for TWO consecutive slope windows (to measure
    # acceleration), not just one.
    min_len = max(2 * slope_weeks, signal_weeks) + 1
    if len(rs) < min_len:
        return None

    rs_now = float(rs.iloc[-1])

    window_now = rs.iloc[-(slope_weeks + 1):]
    rs_slope = _regression_slope(window_now)

    # Acceleration: this window's slope vs the slope measured over the PRIOR
    # window of the same length. Positive = the rotation is steepening (money
    # flowing in faster each week, not just steadily); negative = decelerating
    # even if rs_slope is still positive. This is the "getting more and more
    # strength" read that a single slope number can't distinguish on its own.
    window_prev = rs.iloc[-(2 * slope_weeks + 1):-slope_weeks]
    rs_slope_prev = _regression_slope(window_prev)
    rs_accel = rs_slope - rs_slope_prev

    # Short average of RS; a cross above it marks the turn.
    rs_sig = rs.rolling(signal_weeks).mean()
    crossed_up = bool(
        len(rs_sig.dropna()) >= 2
        and rs.iloc[-1] > rs_sig.iloc[-1]
        and rs.iloc[-2] <= rs_sig.iloc[-2]
    )
    # 4-week RS change, a quick "acceleration" read (kept for backward compat
    # with existing consumers; rs_accel above is the more principled version).
    rs_change_4w = float(rs_now - rs.iloc[-5]) if len(rs) >= 5 else np.nan

    return {
        "rs": round(rs_now, 2),
        "rs_slope": round(rs_slope, 3),
        "rs_accel": round(rs_accel, 3),
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


def _rolling_rs_series(rs: pd.Series, slope_weeks: int, slope_eps: float,
                       tail_weeks: int | None = None) -> pd.DataFrame:
    """
    Per-week (rs, slope, state) history, using the SAME regression slope and
    _classify() as the current-snapshot calculation -- shared so a sector's
    state for a given week is identical whether read from today's snapshot or
    from history. Trims to `tail_weeks` (plus the slope_weeks needed to seed
    the first slope) BEFORE computing, so this stays cheap even though it's
    now called from sector_rotation()'s freshness lookup too, not just
    rotation_history().
    """
    if tail_weeks:
        rs = rs.tail(tail_weeks + slope_weeks)
    if len(rs) <= slope_weeks:
        return pd.DataFrame(columns=["rs", "slope", "state"])

    idx = rs.index[slope_weeks:]
    slopes = [_regression_slope(rs.iloc[i - slope_weeks:i + 1])
             for i in range(slope_weeks, len(rs))]
    aligned_rs = rs.iloc[slope_weeks:]
    states = [_classify(float(r), float(s), slope_eps) for r, s in zip(aligned_rs, slopes)]
    return pd.DataFrame({"rs": aligned_rs.to_numpy(dtype=float), "slope": slopes,
                         "state": states}, index=idx)


def _weeks_since_change(states: list[str]) -> int | None:
    """
    Weeks since the last state change in `states` (oldest -> newest), or None
    if no change was found within the available window -- that ambiguity
    (very stable vs. simply not enough history in the window) matches
    rotation_history()'s original inline logic exactly; this is that same
    logic, extracted so sector_rotation() can share it too.
    """
    for i in range(len(states) - 1, 0, -1):
        if states[i] != states[i - 1]:
            return (len(states) - 1) - i
    return None


def sector_rotation(loader, rs_period: int = 52, slope_weeks: int = 5,
                    signal_weeks: int = 10, start_date: str = "2018-01-01",
                    slope_eps: float = 0.05,
                    freshness_lookback_weeks: int = 26) -> list[dict]:
    """
    Rank the 11 SPDR sector ETFs by rotation strength vs the S&P 500.

    Returns a list of dicts (one per sector), sorted so the freshest
    rotating-in/leading sectors come first. Each dict has:
      etf, sector, rs, rs_slope, rs_accel, rs_change_4w, crossed_up, state,
      changed_weeks_ago.
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

        rs = mansfield_rs(stock_wk, index_wk, n=rs_period).dropna()
        m = _rs_metrics(rs, slope_weeks, signal_weeks)
        if m is None:
            continue

        state = _classify(m["rs"], m["rs_slope"], slope_eps)
        hist = _rolling_rs_series(rs, slope_weeks, slope_eps,
                                  tail_weeks=freshness_lookback_weeks)
        changed_weeks_ago = _weeks_since_change(hist["state"].tolist()) if not hist.empty else None

        m.update({
            "etf": etf, "sector": sector, "state": state,
            "changed_weeks_ago": changed_weeks_ago,
        })
        rows.append(m)

    # Sort: rotating-in first (the early opportunity), then leaders, then the
    # rest. Within a state, FRESHEST first (a sector that just flipped is a
    # stronger signal than one that's been there for months), then steeper
    # slope as a tiebreak. Mirrors rotation_history()'s own sort below, so the
    # two views agree on what "most interesting" means.
    state_rank = {"Rotating In": 0, "Leading": 1, "Neutral": 2,
                  "Weakening": 3, "Lagging": 4}
    rows.sort(key=lambda r: (
        state_rank.get(r["state"], 9),
        r["changed_weeks_ago"] if r["changed_weeks_ago"] is not None else 999,
        -r["rs_slope"], -r["rs"],
    ))
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
        hist = _rolling_rs_series(rs, slope_weeks, slope_eps, tail_weeks=weeks)
        if hist.empty:
            continue

        states = hist["state"].tolist()
        changed_weeks_ago = _weeks_since_change(states)
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

    NOTE (unchanged by this pass, flagged for a future one): sector_name is
    currently unused below -- finviz.get_top_industries() has no sector filter
    applied, so this returns the same market-wide ranking regardless of which
    sector you ask about. See market_us.py::collect_us_by_rotation()'s
    docstring for why the new rotation-driven scanning path deliberately
    doesn't depend on this function yet.
    """
    try:
        # A large cap effectively means "all" (Finviz has ~150 industries).
        limit = top_n if top_n and top_n > 0 else 500
        industries = finviz.get_top_industries(top_n=limit, col_target=col_target)
    except Exception as e:  # noqa: BLE001
        log.error("Finviz industry rotation failed: %s", e)
        return []
    return [{"industry": name, "rank": i + 1} for i, name in enumerate(industries)]
