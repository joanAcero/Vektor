"""
data_loader.py
--------------
Price data access with an honest on-disk cache.

Fixes over the original:
  * The original always re-downloaded and never read the cache, while the
    docstring claimed otherwise. Here the cache is actually consulted and is
    only refreshed when older than ``max_age``.
  * ``auto_adjust`` is set explicitly (its yfinance default flipped to True in
    0.2.28+, silently changing prices for anyone relying on the old default).
    We use adjusted prices because Weinstein-style trend analysis must be
    split/dividend-consistent -- but because adjusted history is rewritten on
    every corporate action, the cache is time-boxed rather than permanent.
  * The download path used to accept any frame containing the five OHLCV
    names, even with EXTRA columns (e.g. 'Close'/'Close.1' from a MultiIndex
    flattened without checking which ticker it belonged to). That let a
    malformed or mis-attributed download get written straight to the cache,
    where nothing downstream could tell a Series-shaped column from a
    corrupted 2-column frame until pandas raised deep inside stages.py or
    rotation.py (align()/classify() crashes several calls removed from the
    real cause). The download path's validation now mirrors the cache-read
    validation exactly -- same _OHLCV set, no extras allowed -- and a
    MultiIndex is checked against the REQUESTED ticker before being
    flattened, not after. A response labelled with the wrong ticker is now
    discarded instead of being cached under the wrong symbol with no error
    raised anywhere.
  * ``threads=False`` on yf.download(). yfinance's default (threads=True)
    stores each ticker's result in an internal global dict (shared._DFS)
    that is not thread-safe -- see ranaroussi/yfinance#2557. Calling
    get_data() in a tight loop for many tickers (exactly what sector_rotation
    and us_market_states do) triggered this: XLE's request coming back
    labelled QQQ, DIA's coming back labelled XLE, etc., every single ticker,
    every run. threads=False is the documented workaround and is the actual
    fix for the mislabelling; the ticker-label check above is what makes
    that failure mode visible in the first place and stays on as a permanent
    safety net, not a workaround to be removed once this is confirmed fixed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataLoader:
    def __init__(self, data_dir: str = "data", max_age_hours: float = 12.0,
                 auto_adjust: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_hours * 3600
        self.auto_adjust = auto_adjust

    def _cache_path(self, ticker: str) -> Path:
        safe = ticker.replace("^", "_").replace("=", "_").replace("/", "_")
        return self.data_dir / f"{safe}.csv"

    def _fresh(self, path: Path) -> bool:
        return path.exists() and (time.time() - path.stat().st_mtime) < self.max_age_seconds

    def get_data(self, ticker: str, start_date: str,
                 end_date: str | None = None) -> pd.DataFrame | None:
        path = self._cache_path(ticker)

        if self._fresh(path):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if not df.empty and set(df.columns) == set(_OHLCV):
                    log.debug("Cache hit: %s", ticker)
                    return df
                log.warning("Cached %s has unexpected columns %s; re-downloading.",
                            ticker, list(df.columns))
            except Exception:  # noqa: BLE001 -- corrupt cache, fall through to download
                log.warning("Corrupt cache for %s; re-downloading.", ticker)

        try:
            df = yf.download(
                ticker, start=start_date, end=end_date,
                progress=False, auto_adjust=self.auto_adjust,
                threads=False,
            )
        except Exception as e:  # noqa: BLE001
            log.error("Download failed for %s: %s", ticker, e)
            return None

        if df is None or df.empty:
            log.warning("No data returned for %s.", ticker)
            return None

        if isinstance(df.columns, pd.MultiIndex):
            # yfinance labels a single-ticker download (Field, Ticker). Under
            # rate-limiting we've observed it return a frame carrying a
            # DIFFERENT ticker's data. Check the label BEFORE collapsing the
            # MultiIndex -- collapsing first and validating names only would
            # still pass, since the field names match regardless of whose
            # data they hold.
            labels = df.columns.get_level_values(1).unique().tolist()
            if labels != [ticker]:
                log.error("%s: download returned data labelled %s, not %s; "
                          "discarding rather than caching under the wrong "
                          "symbol.", ticker, labels, ticker)
                return None
            df.columns = df.columns.get_level_values(0)

        if set(df.columns) != set(_OHLCV):
            log.error("%s: downloaded columns %s don't match expected %s; "
                      "discarding.", ticker, list(df.columns), _OHLCV)
            return None

        try:
            df.to_csv(path)
        except OSError as e:
            log.warning("Could not write cache for %s: %s", ticker, e)

        return df
