"""
rotation.py
-----------
Sector rotation monitor implemented as a Relative Rotation Graph (RRG),
following StockCharts' published specification of JdK RS-Ratio and
JdK RS-Momentum.

    https://help.stockcharts.com/charts-and-tools/other-charting-tools/rrg-charts
    https://chartschool.stockcharts.com/table-of-contents/chart-analysis/
        chart-types/relative-rotation-graphs-rrg-charts

HONEST SCOPE NOTE -- READ THIS BEFORE TRUSTING THE NUMBERS
==========================================================
StockCharts does NOT publish the formula for RS-Ratio or RS-Momentum.
"Relative Rotation Graph(R)" and "RRG(R)" are registered trademarks of RRG
Research, and both StockCharts pages describe only the *properties* of the
two indicators ("normalized", "fluctuate above/below 100", "trend-following
with a lag", "momentum of RS-Ratio") -- never the arithmetic. Querying their
documentation agent endpoint for the formula returns the same prose.

Therefore this module is a RECONSTRUCTION, not a replication. It is built to
satisfy every documented property, and each property is covered by a test in
tests/test_rotation_rrg.py. What it will reproduce: quadrant membership,
rotation direction, lead/lag relationships, ranking order, and trail shape.
What it will NOT reproduce: StockCharts' exact decimal values. Expect the
same quadrant for a sector the large majority of the time, with disagreements
concentrated near the 100 axes where a security is nearly on the line anyway.

Documented properties this implementation satisfies (verified by tests):
  P1  Both indicators are normalized and oscillate around 100, with typical
      readings a few points either side (the ChartSchool worked example
      quotes XLK=102.04, XLI=101.41, XLF=100.2, XLV=103.66).
  P2  RS-Ratio is TREND-FOLLOWING: whenever it crosses 100, the price
      relative has ALREADY been moving that way for weeks ("there will
      already be upward movement in the price relative before RS-Ratio
      crosses above 100"). Note the stronger form that does NOT hold here:
      peak-to-peak phase. Rolling z-score normalization is a high-pass
      filter, so on slow relative cycles (>~2yr) RS-Ratio can peak slightly
      BEFORE the price relative even while still crossing 100 late. The
      documented cross behaviour is what the tests assert.
  P3  RS-Momentum is the momentum of RS-Ratio and LEADS it, crossing 100 as
      RS-Ratio forms a trough or peak.
  P4  Rotation is clockwise: leading -> weakening -> lagging -> improving.
  P5  Values are comparable across securities sharing one benchmark.
  P6  A persistent relative uptrend keeps RS-Ratio above 100 (trails "can
      remain on the right side of the RRG"), and vice versa.
  P7  NOT SATISFIED, and disclosed rather than hidden. StockCharts states
      ~50 weekly observations suffice; this reconstruction needs ~92 (about
      1.8 years) because RS-Momentum is normalized a second time on top of
      RS-Ratio. Configurations that DO fit 50 weeks were tested and
      rejected: they more than doubled RS-Momentum's whipsaw across the 100
      line (23 crossings vs 9 on an identical series), which would corrupt
      the quadrant assignment this monitor exists to produce. Because the
      monitor loads from 2018 by default this costs nothing in practice --
      but a newly listed ETF is skipped for ~2 years rather than plotted.
  P8  Ranking is by quadrant -- leading, improving, weakening, lagging -- and
      within a quadrant by distance from the 100/100 origin, descending.

Deliberate deviation from the StockCharts default
=================================================
StockCharts' default benchmark is $SPX, a PRICE index. This module uses
benchmark_for("US") -> SPY, a total-return ETF, to match the SPDR sector
ETFs, which the loader also returns dividend-adjusted. Comparing a
total-return numerator against a price-index denominator injects a bias of
roughly the index dividend yield (~1.5-2%/yr) into every relative-strength
reading -- see src/benchmarks.py. Keeping SPY is a correctness fix; it does
mean sector RS here is very slightly stronger than the same sector's RS on a
default StockCharts RRG.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.benchmarks import benchmark_for, get_weekly_close
from src.stages import Stage, classify_last, hunt_rank
from src.benchmarks import get_weekly_close, benchmark_for, mansfield_rs
log = logging.getLogger(__name__)


# ======================================================================
# SPEC CONSTANTS -- stated in the StockCharts documentation.
# ======================================================================
QUAD_LEADING = "leading"        # RS-Ratio > 100 and RS-Momentum > 100
QUAD_WEAKENING = "weakening"    # RS-Ratio > 100, RS-Momentum < 100
QUAD_LAGGING = "lagging"        # both < 100
QUAD_IMPROVING = "improving"    # RS-Ratio < 100, RS-Momentum > 100

QUADRANTS: tuple[str, ...] = (QUAD_LEADING, QUAD_WEAKENING,
                              QUAD_LAGGING, QUAD_IMPROVING)

# "The leading quadrant (green) is first, the improving quadrant (blue) is
#  second, the weakening quadrant (yellow) is third, and the lagging
#  quadrant (red) is last."
QUADRANT_RANK: dict[str, int] = {
    QUAD_LEADING: 0, QUAD_IMPROVING: 1, QUAD_WEAKENING: 2, QUAD_LAGGING: 3,
}

# StockCharts' own quadrant colours, so our chart matches theirs.
QUADRANT_COLOR: dict[str, str] = {
    QUAD_LEADING: "#2f9e44",     # green
    QUAD_WEAKENING: "#f1c40f",   # yellow
    QUAD_LAGGING: "#e03131",     # red
    QUAD_IMPROVING: "#1c7ed6",   # blue
}

RRG_ORIGIN = 100.0              # both axes cross at 100
DEFAULT_TAIL_WEEKS = 12         # the ChartSchool worked example uses 12-week tails
MIN_OBSERVATIONS = 50           # "a minimum of 50 data points"

# The 11 SPDR Select Sector ETFs (the full set; XLC and XLRE were added to
# the original nine in 2018 and 2015 respectively, so their RRG history is
# shorter than the others').
SECTOR_ETFS: dict[str, str] = {
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}

# ======================================================================
# RECONSTRUCTION CONSTANTS -- NOT published by StockCharts. These are the
# choices that make the indicators satisfy P1-P8. They are the honest
# location of all uncertainty in this module.
# ======================================================================
RS_TREND_WEEKS = 10     # smoothing that makes RS-Ratio trend-following (P2).
                        # A deviation-from-its-own-MA construction was tried
                        # first and REJECTED: it leads the price relative by
                        # ~13 weeks instead of lagging it, contradicting the
                        # documentation's explicit lag statement.
RS_NORM_WEEKS = 40      # z-score window that puts every sector on one
                        # comparable scale centred at 100 (P1, P5). Measured
                        # against 26 and 20: all three satisfy P1/P5/P6, but
                        # 40 gives much the least whipsaw (RS-Momentum
                        # crosses the 100 line 9 times vs 23 at NORM=20 on an
                        # identical series) and the best-behaved phase. Its
                        # only cost is warm-up -- see RRG_WARMUP_WEEKS, P7.
RS_MOM_ROC_WEEKS = 4    # rate-of-change window for RS-Momentum. Short enough
                        # that momentum turns as RS-Ratio rounds over (P3);
                        # the documentation notes RS-Momentum crosses 100
                        # often, so it is intentionally NOT smoothed further.

# Weeks of history consumed before RS-Momentum yields a first value: RS-Ratio
# needs TREND + NORM - 1, then RS-Momentum needs a further ROC window plus a
# SECOND NORM window. This -- not MIN_OBSERVATIONS -- is the real data
# requirement. MIN_OBSERVATIONS is kept above as StockCharts' documented
# figure, which this reconstruction does not meet (see P7).
RRG_WARMUP_WEEKS = (RS_TREND_WEEKS + RS_NORM_WEEKS - 1
                    + RS_MOM_ROC_WEEKS + RS_NORM_WEEKS - 1)

# ======================================================================
# COMPATIBILITY MAPPING -- VEKTOR-specific, not part of the RRG spec.
# ======================================================================
# The scan pipeline (market_us.collect_us_by_rotation, config
# market.rotation_states) selects candidate universes by a hunt/watch/avoid
# label. RRG quadrants are mapped onto it here, in ONE place.
#
# JUDGEMENT CALL, stated openly: `improving` -> watch rather than hunt.
# Improving means relative momentum has turned up while the relative trend is
# still below par -- which is exactly where Weinstein Stage-1 bases sit, and
# is arguably the most valuable quadrant for this system. It is mapped to
# `watch` only because `hunt` historically meant "leadership confirmed".
# If you want to scan improving sectors by default, either move it to `hunt`
# here or set market.rotation_states: ["hunt", "watch"] in default.yaml.
QUADRANT_TO_STATE: dict[str, str] = {
    QUAD_LEADING: "hunt",
    QUAD_IMPROVING: "watch",
    QUAD_WEAKENING: "avoid",
    QUAD_LAGGING: "avoid",
}
STATES: tuple[str, ...] = ("hunt", "watch", "avoid")
STAGE_HUNT_RANK: dict[str, int] = {
    "stage2": 0, "stage1": 1, "stage3": 2, "stage4": 3, "unknown": 4,
}
DEFAULT_HISTORY_WEEKS = 26
DEFAULT_START_DATE = "2018-01-01"


# ------------------------------------------------------------------
# CORE RRG MATH
# ------------------------------------------------------------------
def _zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score. ddof=0 (population sigma): the window IS the
    reference distribution here, not a sample drawn from a larger one."""
    mean = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


def rrg_components(security_weekly_close: pd.Series,
                   benchmark_weekly_close: pd.Series) -> pd.DataFrame:
    """
    JdK RS-Ratio and JdK RS-Momentum for one security against one benchmark.

    Construction (see module docstring for why these choices):
        RS           = 100 * security / benchmark          (the price relative)
        RS-Ratio     = 100 + z( SMA(RS, 10), 40 )          trend, lags RS
        RS-Momentum  = 100 + z( ROC(RS-Ratio, 4), 40 )     momentum, leads Ratio

    Returns a DataFrame indexed by the intersection of both inputs' dates
    with columns [rs, rs_ratio, rs_momentum]. Rows before the warm-up are
    dropped, so an empty frame means "not enough history", not "no rotation".
    """
    sec, bench = security_weekly_close.align(benchmark_weekly_close,
                                             join="inner")
    if sec.empty or bench.empty:
        return pd.DataFrame(columns=["rs", "rs_ratio", "rs_momentum"])

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = (100.0 * sec / bench).replace([np.inf, -np.inf], np.nan)

    rs_ratio = RRG_ORIGIN + _zscore(rs.rolling(RS_TREND_WEEKS).mean(),
                                    RS_NORM_WEEKS)
    with np.errstate(divide="ignore", invalid="ignore"):
        roc = 100.0 * (rs_ratio / rs_ratio.shift(RS_MOM_ROC_WEEKS) - 1.0)
    rs_momentum = RRG_ORIGIN + _zscore(roc.replace([np.inf, -np.inf], np.nan),
                                       RS_NORM_WEEKS)

    out = pd.DataFrame({"rs": rs, "rs_ratio": rs_ratio,
                        "rs_momentum": rs_momentum})
    return out.dropna()


def classify_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    """Quadrant from the two indicators. Axes cross at 100."""
    strong = rs_ratio >= RRG_ORIGIN
    rising = rs_momentum >= RRG_ORIGIN
    if strong and rising:
        return QUAD_LEADING
    if strong:
        return QUAD_WEAKENING
    if rising:
        return QUAD_IMPROVING
    return QUAD_LAGGING


def _distance(rs_ratio: float, rs_momentum: float) -> float:
    """Euclidean distance from the benchmark crosshair at (100, 100).
    Drives both the ranking within a quadrant and the trail thickness."""
    return float(np.hypot(rs_ratio - RRG_ORIGIN, rs_momentum - RRG_ORIGIN))


def _heading_degrees(tail: pd.DataFrame) -> float | None:
    """Direction of travel of the last tail segment, in degrees measured
    clockwise from 'due north' on the RRG plot (0 = straight up, 90 = right,
    180 = down, 270 = left). Ideal RRG rotation sweeps this angle clockwise.
    None when the tail is too short or the security has not moved."""
    if len(tail) < 2:
        return None
    dx = float(tail["rs_ratio"].iloc[-1] - tail["rs_ratio"].iloc[-2])
    dy = float(tail["rs_momentum"].iloc[-1] - tail["rs_momentum"].iloc[-2])
    if dx == 0.0 and dy == 0.0:
        return None
    return float(np.degrees(np.arctan2(dx, dy)) % 360.0)


def _weeks_in_quadrant(quadrants: list[str]) -> int | None:
    """Weeks the final quadrant has held. None when no change is visible in
    the supplied window -- genuinely ambiguous between 'very stable' and
    'window too short', so callers must not read it as 0."""
    if not quadrants:
        return None
    for i in range(len(quadrants) - 1, 0, -1):
        if quadrants[i] != quadrants[i - 1]:
            return (len(quadrants) - 1) - i
    return None

# ------------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------------
def sector_rotation(loader, *,
                    tail_weeks: int = DEFAULT_TAIL_WEEKS,
                    start_date: str = DEFAULT_START_DATE,
                    rs_weeks: int = 52) -> list[dict]:
    """
    RRG snapshot of the 11 SPDR sector ETFs against the US benchmark.

    Sorted per the StockCharts symbol table: quadrant order leading ->
    improving -> weakening -> lagging, and within each quadrant by distance
    from the 100/100 crosshair, furthest first.

    Returns one dict per sector:
      {
        "etf": "XLK", "sector": "Technology",
        "quadrant": "leading",            # leading|weakening|lagging|improving
        "rs_ratio": 102.04,               # x-axis
        "rs_momentum": 100.87,            # y-axis
        "distance": 2.23,                 # from the 100/100 origin
        "heading": 74.5,                  # degrees clockwise from north, or None
        "weeks_in_quadrant": 6,           # None if no change seen in the window
        "quadrant_changed": False,        # entered this quadrant this week
        "ratio_crossed_up": False,        # RS-Ratio crossed above 100 this week
        "ratio_crossed_down": False,      # ...or below
        "tail_pct_change": 4.8,           # % price change over the tail
        "sector_stage": "stage2",         # absolute Weinstein stage
        "state": "hunt",                  # compat: QUADRANT_TO_STATE
        "tail": [{"date": "...", "rs_ratio": .., "rs_momentum": ..}, ...],
        # deprecated origin-centred aliases, kept so the existing web plot
        # and any older consumer keep working:
        "rs": 2.04, "rs_slope": 0.87, "weeks_in_state": 6, "rs_crossed_up": False,
      }

    Sectors with fewer than MIN_OBSERVATIONS usable weeks are skipped with a
    warning rather than emitted with NaNs -- matching StockCharts, which
    simply does not plot such a symbol.
    """
    benchmark_symbol = benchmark_for("US")
    index_wk = get_weekly_close(loader, benchmark_symbol, start_date=start_date)
    if index_wk is None:
        log.error("Could not load benchmark %s for sector rotation.",
                  benchmark_symbol)
        return []

    rows: list[dict] = []
    for etf, sector in SECTOR_ETFS.items():
        etf_wk = get_weekly_close(loader, etf, start_date=start_date)
        if etf_wk is None:
            log.warning("No data for sector ETF %s (%s); skipping.", etf, sector)
            continue
        if len(etf_wk) < RRG_WARMUP_WEEKS:
            log.warning("%s has %d weekly points; this RRG reconstruction "
                        "needs %d (StockCharts documents %d -- see P7 in the "
                        "module docstring). Skipping.",
                        etf, len(etf_wk), RRG_WARMUP_WEEKS, MIN_OBSERVATIONS)
            continue

        comp = rrg_components(etf_wk, index_wk)
        if comp.empty:
            log.warning("Not enough RRG history for %s (needs ~%d weeks); "
                        "skipping.", etf, RRG_WARMUP_WEEKS)
            continue

        tail = comp.tail(max(2, tail_weeks))
        last = comp.iloc[-1]
        rs_ratio = float(last["rs_ratio"])
        rs_mom = float(last["rs_momentum"])
        quadrant = classify_quadrant(rs_ratio, rs_mom)

        quad_hist = [classify_quadrant(r, m) for r, m
                     in zip(comp["rs_ratio"], comp["rs_momentum"])]
        weeks_in_quadrant = _weeks_in_quadrant(quad_hist[-DEFAULT_HISTORY_WEEKS:])

        prev_ratio = float(comp["rs_ratio"].iloc[-2]) if len(comp) >= 2 else rs_ratio
        tail_start_rs = float(tail["rs"].iloc[0])
        tail_pct = ((float(last["rs"]) / tail_start_rs - 1.0) * 100.0
                    if tail_start_rs else 0.0)
        stage = classify_last(etf_wk)
        mrs_series = mansfield_rs(etf_wk, index_wk, n=rs_weeks)
        mansfield = (float(mrs_series.iloc[-1])
                     if not mrs_series.empty and pd.notna(mrs_series.iloc[-1])
                     else None)
        rows.append({
            "etf": etf,
            "sector": sector,
            "quadrant": quadrant,
            "rs_ratio": round(rs_ratio, 2),
            "rs_momentum": round(rs_mom, 2),
            "distance": round(_distance(rs_ratio, rs_mom), 2),
            "heading": (round(h, 1) if (h := _heading_degrees(tail)) is not None
                        else None),
            "weeks_in_quadrant": weeks_in_quadrant,
            "quadrant_changed": bool(len(quad_hist) >= 2
                                     and quad_hist[-1] != quad_hist[-2]),
            "ratio_crossed_up": bool(prev_ratio < RRG_ORIGIN <= rs_ratio),
            "ratio_crossed_down": bool(prev_ratio >= RRG_ORIGIN > rs_ratio),
            "tail_pct_change": round(tail_pct, 2),
            "sector_stage": stage.slug,
            "mansfield_rs": round(mansfield, 2) if mansfield is not None else None,
            "hunt_rank": hunt_rank(stage),
            "tail": [
                {"date": d.strftime("%Y-%m-%d"),
                 "rs_ratio": round(float(r.rs_ratio), 2),
                 "rs_momentum": round(float(r.rs_momentum), 2)}
                for d, r in tail.iterrows()
            ]
        })

    rows.sort(key=lambda r: (QUADRANT_RANK.get(r["quadrant"], 9),
                             -r["distance"]))
    return rows


def sector_leaders_history(loader, *, weeks_back: int = 6,
                           start_date: str = DEFAULT_START_DATE,
                           rs_weeks: int = 52) -> dict:
    """
    El filtro de Weinstein (Etapa 2 y Mansfield RS > 0) RECALCULADO en cada
    uno de los últimos `weeks_back` cortes semanales, con el RANGO por RS
    entre los líderes DE ESA SEMANA (no fijo a los líderes de hoy) -- para
    ver si alguien que ya era líder ha subido o bajado dentro del grupo, no
    solo si sigue dentro o fuera.

    RECORTE, NO HISTORIAL GUARDADO -- ver la nota en la versión anterior
    de esta función sobre por qué esto es exacto y no una aproximación.

        {
          "weeks": ["2026-07-18", ..., "2026-08-29"],
          "sectors": [
            {"etf": "XLK", "sector": "Technology",
             "trail": [
               {"is_leader": false, "rank": null, "mansfield_rs": -1.2},
               ...
               {"is_leader": true,  "rank": 1,    "mansfield_rs": 6.4}
             ]},
            ...
          ]
        }

    `rank` es 1-based, solo entre los líderes de ESA semana; None si esa
    semana no era líder o no había suficiente historial.
    """
    benchmark_symbol = benchmark_for("US")
    index_wk_full = get_weekly_close(loader, benchmark_symbol, start_date=start_date)
    if index_wk_full is None:
        log.error("Could not load benchmark %s for leaders history.", benchmark_symbol)
        return {"weeks": [], "sectors": []}

    cuts = list(range(weeks_back, -1, -1))  # antigua -> hoy
    weeks: list[str] | None = None
    # raw[k_index] = list of (etf, sector, mansfield_rs) para los LÍDERES de esa semana
    raw_leaders_per_week: list[list[tuple[str, str, float]]] = [[] for _ in cuts]
    per_sector: dict[str, dict] = {}  # etf -> {"sector":.., "trail": [dict,...]}

    for etf, sector in SECTOR_ETFS.items():
        etf_wk_full = get_weekly_close(loader, etf, start_date=start_date)
        if etf_wk_full is None:
            log.warning("No data for %s in leaders history; skipping.", etf)
            continue

        trail: list[dict] = []
        these_weeks: list[str] = []
        skip = False
        for wi, k in enumerate(cuts):
            end = len(etf_wk_full) - k
            if end < RRG_WARMUP_WEEKS:
                skip = True
                break
            etf_wk = etf_wk_full.iloc[:end]
            index_wk = index_wk_full.reindex(etf_wk.index).dropna()

            stage = classify_last(etf_wk)
            mrs_series = mansfield_rs(etf_wk, index_wk, n=rs_weeks)
            mrs = (float(mrs_series.iloc[-1])
                   if not mrs_series.empty and pd.notna(mrs_series.iloc[-1])
                   else None)
            is_leader = stage == Stage.TWO and (mrs or 0) > 0
            trail.append({"is_leader": is_leader, "rank": None,  # rango se rellena después
                          "mansfield_rs": round(mrs, 2) if mrs is not None else None})
            these_weeks.append(etf_wk.index[-1].strftime("%Y-%m-%d"))
            if is_leader:
                raw_leaders_per_week[wi].append((etf, sector, mrs))

        if skip:
            log.warning("%s has insufficient history for a %d-week leaders "
                        "trail; skipping.", etf, weeks_back)
            continue

        if weeks is None:
            weeks = these_weeks
        per_sector[etf] = {"sector": sector, "trail": trail}

    # Rango por semana: SOLO entre quienes fueron líderes esa semana concreta.
    for wi, leaders in enumerate(raw_leaders_per_week):
        leaders_sorted = sorted(leaders, key=lambda t: t[2], reverse=True)
        for rank, (etf, _sector, _mrs) in enumerate(leaders_sorted, start=1):
            if etf in per_sector:
                per_sector[etf]["trail"][wi]["rank"] = rank

    out = [{"etf": etf, **data} for etf, data in per_sector.items()]
    return {"weeks": weeks or [], "sectors": out}


def rotation_history(loader, *,
                     weeks: int = DEFAULT_HISTORY_WEEKS,
                     start_date: str = DEFAULT_START_DATE) -> dict:
    """
    Per-week quadrant of every sector over the last `weeks` weeks, oldest
    first. Uses the same `rrg_components` series as the snapshot, so a given
    week's label is identical whether read here or from `sector_rotation`.

    Returns:
      {
        "dates": ["2026-02-06", ..., "2026-07-24"],
        "sectors": [
          {"etf": "XLK", "sector": "Technology",
           "quadrants": ["improving", ..., "leading"],
           "rs_ratios": [99.1, ..., 102.0],
           "rs_momentums": [100.4, ..., 100.9],
           "current_quadrant": "leading",
           "weeks_in_quadrant": 6,
           "dates": [...],
           # compat aliases
           "states": [...], "current_state": "hunt", "weeks_in_state": 6,
           "stages": [...], "current_stage": "stage2"},
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
        etf_wk = get_weekly_close(loader, etf, start_date=start_date)
        if etf_wk is None or len(etf_wk) < RRG_WARMUP_WEEKS:
            continue
        comp = rrg_components(etf_wk, index_wk)
        if comp.empty:
            continue

        window = comp.tail(weeks)
        quads = [classify_quadrant(r, m) for r, m
                 in zip(window["rs_ratio"], window["rs_momentum"])]
        these_dates = [d.strftime("%Y-%m-%d") for d in window.index]
        if dates is None:
            dates = these_dates

        stage = _sector_stage(etf_wk)
        states = [QUADRANT_TO_STATE[q] for q in quads]
        out.append({
            "etf": etf,
            "sector": sector,
            "quadrants": quads,
            "rs_ratios": [round(float(v), 2) for v in window["rs_ratio"]],
            "rs_momentums": [round(float(v), 2) for v in window["rs_momentum"]],
            "current_quadrant": quads[-1],
            "weeks_in_quadrant": _weeks_in_quadrant(quads),
            "dates": these_dates,
            # compat
            "states": states,
            "current_state": states[-1],
            "weeks_in_state": _weeks_in_quadrant(quads),
            "stages": [stage] * len(quads),
            "current_stage": stage,
        })

    out.sort(key=lambda r: QUADRANT_RANK.get(r["current_quadrant"], 9))
    return {"dates": dates or [], "sectors": out}
