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


BENCHMARKS: dict[str, str] = {
    # US: SPY (total-return ETF) not ^GSPC (price index), so Mansfield RS
    # against the SPDR sector ETFs -- which are also total-return -- is
    # unbiased. See module docstring.
    "US": "SPY",
    # European markets still use the EURO STOXX 50 price index for now.
    # Same caveat applies: comparing to iShares MSCI Europe (IEUR, EXW1.DE)
    # or similar total-return ETFs would be more consistent. Left as-is
    # because we don't have a strong preferred total-return proxy.
    "DE": "^STOXX50E",
    "GB": "^STOXX50E",
    "FR": "^STOXX50E",
    "IT": "^STOXX50E",
    "ES": "^STOXX50E",
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


def detect_regime(loader: DataLoader, market_code: str,
                  sma_weeks: int = 30, slope_weeks: int = 5) -> dict:
    """
    Classify the market regime from the benchmark:
      bull    -- benchmark above its 30W SMA and the SMA rising
      bear    -- benchmark below its 30W SMA and the SMA falling
      neutral -- anything else

    A long-only screener can skip scanning in a bear regime.
    """
    symbol = benchmark_for(market_code)
    wk = get_weekly_close(loader, symbol)
    if wk is None or len(wk) < sma_weeks + slope_weeks:
        return {"regime": "neutral", "benchmark": symbol,
                "reason": "insufficient benchmark history"}

    sma = wk.rolling(sma_weeks).mean()
    last_close = float(wk.iloc[-1])
    last_sma = float(sma.iloc[-1])
    slope = float(sma.iloc[-1] - sma.iloc[-1 - slope_weeks])

    above = last_close > last_sma
    rising = slope > 0

    if above and rising:
        regime, reason = "bull", "benchmark above 30W SMA, SMA rising"
    elif (not above) and (not rising):
        regime, reason = "bear", "benchmark below 30W SMA, SMA falling"
    else:
        regime, reason = "neutral", "benchmark and SMA disagree"

    return {"regime": regime, "benchmark": symbol, "reason": reason,
            "close": last_close, "sma": last_sma, "slope": slope}


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