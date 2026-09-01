"""
benchmarks.py
-------------
Market benchmarks, regime detection and Mansfield relative strength.

Mansfield RS is the Weinstein-school measure of whether a stock (or sector
ETF) is outperforming its market index:

    RP  = (stock / index) * 100
    MRS = ((RP / SMA(RP, n)) - 1) * 100

MRS > 0 means the stock is stronger than the index relative to its own
recent norm; MRS < 0 means it is lagging.

Benchmark choice matters for correctness. yfinance with `auto_adjust=True`
returns TOTAL-RETURN prices (dividends reinvested) for ETFs like SPY and
the SPDR sector ETFs (XL*), but ^GSPC is a PRICE INDEX -- dividends are
excluded from its history no matter how it's fetched. Comparing a total-
return numerator to a price-index denominator introduces a systematic bias
of roughly the S&P 500's dividend yield (~1.5-2%/yr) into every Mansfield
RS reading, favoring high-yield sectors (XLU, XLP, XLRE, XLE) and
penalising low-yield ones (XLK, XLY).

Fix: benchmark to SPY, which under auto_adjust is total-return like the
sectors. All Mansfield RS calculations become apples-to-apples.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_loader import DataLoader

log = logging.getLogger(__name__)

US_MARKET_INDICES: dict[str, tuple[str, str]] = {
    # code -> (display name, ETF symbol)
    "sp500":    ("S&P 500",     "SPY"),
    "nasdaq":   ("Nasdaq 100",  "QQQ"),
    "dowjones": ("Dow Jones",   "DIA"),
    "russell2000": ("Russell 2000", "IWM"),
}

BENCHMARKS: dict[str, str] = {
    "US": "SPY",
}
_DEFAULT_BENCHMARK = "SPY"


def benchmark_for(market_code: str) -> str:
    return BENCHMARKS.get(market_code, _DEFAULT_BENCHMARK)


def _to_weekly_close(df: pd.DataFrame) -> pd.Series:
    """Resample a daily OHLCV frame to a weekly (W-FRI) close series."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df["Close"].resample("W-FRI").last().dropna()


def get_weekly_close(loader: DataLoader, symbol: str,
                     start_date: str = "2018-01-01") -> pd.Series | None:
    """Fetch a symbol and return its weekly close series (None if unavailable)."""
    df = loader.get_data(symbol, start_date=start_date)
    if df is None or df.empty or "Close" not in df.columns:
        log.warning("Benchmark data unavailable for %s.", symbol)
        return None
    wk = _to_weekly_close(df)
    return wk if not wk.empty else None

def market_state_for_symbol(loader: DataLoader, symbol: str) -> dict:
    """
    market_state()'s calculation, but on an EXPLICIT symbol instead of one
    derived from a market_code via benchmark_for(). market_state() below is
    now a one-line wrapper over this for the single-benchmark callers
    (detect_regime, run.py's gate) that still only care about one symbol.
    """
    from src.stages import weekly_ma, ma_slope_pct, TREND_SLOPE_PCT

    wk = get_weekly_close(loader, symbol)
    if wk is None:
        return {"regime": "unknown", "label": "unknown", "benchmark": symbol,
                "close": None, "reason": "no benchmark data"}

    ma = weekly_ma(wk)
    slope = ma_slope_pct(ma)
    bull = (wk > ma) & (slope > TREND_SLOPE_PCT)
    bear = (wk < ma) & (slope < -TREND_SLOPE_PCT)

    if bool(bull.iloc[-1]):
        regime, label = "bull", "alcista"
    elif bool(bear.iloc[-1]):
        regime, label = "bear", "bajista"
    else:
        side = pd.Series(np.where(bull, "bull", np.where(bear, "bear", np.nan)),
                         index=wk.index).ffill()
        last_side = side.iloc[-1] if not side.empty else None
        if last_side == "bull":
            regime, label = "amber_from_bull", "correccion"
        elif last_side == "bear":
            regime, label = "amber_from_bear", "base"
        else:
            regime, label = "unknown", "unknown"

    reason = None if regime != "unknown" else "insufficient weekly history"
    return {"regime": regime, "label": label, "benchmark": symbol,
            "close": float(wk.iloc[-1]), "reason": reason}


def market_state(loader: DataLoader, market_code: str = "US") -> dict:
    """Thin wrapper over market_state_for_symbol() for single-benchmark
    callers (detect_regime, run.py) that still think in market_code terms."""
    return market_state_for_symbol(loader, benchmark_for(market_code))


def us_market_states(loader: DataLoader) -> list[dict]:
    """market_state_for_symbol() for every index in US_MARKET_INDICES, in
    that dict's declared order. Backs the four-way semaphore on the Sector
    Rotation tab."""
    out = []
    for code, (name, symbol) in US_MARKET_INDICES.items():
        state = market_state_for_symbol(loader, symbol)
        out.append({"code": code, "name": name, **state})
    return out


def detect_regime(loader: DataLoader, market_code: str) -> dict:
    """
    Coarse bull/bear/neutral gate for run.py's long-only scanners.

    Thin wrapper over market_state(): "bull"/"bear" pass straight through,
    both amber flavors and "unknown" collapse to "neutral" -- this gate only
    needs to know whether to skip scanning, not which kind of pause the
    market is in. Shares the MA30/slope primitives with market_state()
    (and therefore with src/stages.py) rather than defining its own
    threshold, which is the one thing worth keeping from my earlier attempt
    at consolidating this with classify_last() -- that attempt was wrong
    for the reason explained in market_state()'s docstring, but sharing a
    threshold instead of a state machine is still a correct simplification.
    """
    state = market_state(loader, market_code)
    if state["regime"] in ("bull", "bear"):
        regime = state["regime"]
    else:
        regime = "neutral"
    reason = {
        "bull": "benchmark above a rising 30W MA",
        "bear": "benchmark below a falling 30W MA",
    }.get(state["regime"], state.get("reason") or f"benchmark reading: {state['label']}")

    return {"regime": regime, "benchmark": state["benchmark"], "reason": reason}

def mansfield_rs(stock_weekly_close: pd.Series, index_weekly_close: pd.Series,
                 n: int = 52) -> pd.Series:
    """
    Mansfield relative strength of a stock vs an index, on weekly closes.

        RP  = (stock / index) * 100
        MRS = ((RP / SMA(RP, n)) - 1) * 100

    Returns a Series aligned to the intersection of both inputs' dates.
    """
    stock, index = stock_weekly_close.align(index_weekly_close, join="inner")
    if stock.empty or index.empty:
        return pd.Series(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        rp = (stock / index) * 100.0
        rp_sma = rp.rolling(n).mean()
        mrs = ((rp / rp_sma) - 1.0) * 100.0

    return mrs.replace([np.inf, -np.inf], np.nan)