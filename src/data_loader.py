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
    split/dividend-consistent — but because adjusted history is rewritten on
    every corporate action, the cache is time-boxed rather than permanent.
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
                if not df.empty:
                    log.debug("Cache hit: %s", ticker)
                    return df
            except Exception:  # noqa: BLE001 — corrupt cache, fall through to download
                log.warning("Corrupt cache for %s; re-downloading.", ticker)

        try:
            df = yf.download(
                ticker, start=start_date, end=end_date,
                progress=False, auto_adjust=self.auto_adjust,
            )
        except Exception as e:  # noqa: BLE001
            log.error("Download failed for %s: %s", ticker, e)
            return None

        if df is None or df.empty:
            log.warning("No data returned for %s.", ticker)
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        missing = [c for c in _OHLCV if c not in df.columns]
        if missing:
            log.error("%s missing expected columns %s; skipping.", ticker, missing)
            return None

        try:
            df.to_csv(path)
        except OSError as e:
            log.warning("Could not write cache for %s: %s", ticker, e)

        return df
