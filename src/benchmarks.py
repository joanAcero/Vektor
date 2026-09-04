from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_loader import DataLoader

log = logging.getLogger(__name__)

US_MARKET_INDICES: dict[str, tuple[str, str]] = {
    # code -> (display name, ETF symbol)
    "rsp":         ("S&P 500 Equal Weight", "RSP"),
    "sp500":       ("S&P 500",              "SPY"),
    "nasdaq":      ("Nasdaq 100",           "QQQ"),
    "dowjones":    ("Dow Jones",            "DIA"),
    "russell2000": ("Russell 2000",         "IWM"),
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

def market_state_for_symbol(loader: DataLoader, symbol: str,
                            timeframe: str = "long") -> dict:
    """
    Market state calculation for a given symbol and timeframe.

    Timeframe:
      - "long": Uses 30-week MA (weekly_ma), representing Weinstein's primary trend.
      - "medium": Uses 10-week MA (weekly_ma_fast), representing the medium-term trend.

    Regimes:
      - "bull": Close > MA and MA slope > +0.5% (Alcista)
      - "bear": Close < MA and MA slope < -0.5% (Bajista)
      - "amber_from_bull": In consolidation, last confirmed trend was bull (Corrección)
      - "amber_from_bear": In consolidation, last confirmed trend was bear (Formando base)
      - "unknown": Insufficient data
    """
    from src.stages import weekly_ma, weekly_ma_fast, ma_slope_pct, TREND_SLOPE_PCT

    wk = get_weekly_close(loader, symbol)
    if wk is None:
        return {"regime": "unknown", "label": "unknown", "benchmark": symbol,
                "close": None, "reason": "no benchmark data"}

    if timeframe == "medium":
        ma = weekly_ma_fast(wk)
    else:
        ma = weekly_ma(wk)

    slope = ma_slope_pct(ma)
    bull = (wk > ma) & (slope > TREND_SLOPE_PCT)
    bear = (wk < ma) & (slope < -TREND_SLOPE_PCT)

    if bool(bull.iloc[-1]):
        regime, label = "bull", "alcista"
    elif bool(bear.iloc[-1]):
        regime, label = "bear", "bajista"
    else:
        side = pd.Series(None, index=wk.index, dtype=object)
        side[bull] = "bull"
        side[bear] = "bear"
        side = side.ffill()
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


def market_state(loader: DataLoader, market_code: str = "US",
                 timeframe: str = "long") -> dict:
    """Thin wrapper over market_state_for_symbol() for single-benchmark
    callers (detect_regime, run.py) that still think in market_code terms."""
    return market_state_for_symbol(loader, benchmark_for(market_code), timeframe=timeframe)


def us_market_states(loader: DataLoader, timeframe: str = "long") -> list[dict]:
    """market_state_for_symbol() for every index in US_MARKET_INDICES, in
    that dict's declared order for the specified timeframe ('long' or 'medium')."""
    out = []
    for code, (name, symbol) in US_MARKET_INDICES.items():
        state = market_state_for_symbol(loader, symbol, timeframe=timeframe)
        out.append({"code": code, "name": name, **state})
    return out


def us_market_states_all(loader: DataLoader) -> dict[str, list[dict]]:
    """Convenience helper returning both long-term and medium-term states."""
    return {
        "long_term": us_market_states(loader, timeframe="long"),
        "medium_term": us_market_states(loader, timeframe="medium"),
    }


def detect_regime(loader: DataLoader, market_code: str) -> dict:
    """
    Coarse bull/bear/neutral gate for run.py's long-only scanners.

    Thin wrapper over market_state(): "bull"/"bear" pass straight through,
    both amber flavors and "unknown" collapse to "neutral" -- this gate only
    needs to know whether to skip scanning, not which kind of pause the
    market is in.
    """
    state = market_state(loader, market_code, timeframe="long")
    if state["regime"] == "bull":
        regime = "bull"
    elif state["regime"] == "bear":
        regime = "bear"
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