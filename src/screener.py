"""
screener.py
-----------
Runs a strategy across a list of tickers. Strategy-agnostic: it knows nothing
about Weinstein or any specific column. It reads the signal column, hit values
and output columns from ``strategy.meta``.

Fixes over the original:
  * The original used a bare ``except Exception: continue`` that silently
    discarded every per-ticker error — fatal for a detection tool. Here errors
    are logged with the ticker, counted, and (when strict=True) re-raised, so a
    bug in a new strategy surfaces immediately instead of yielding a false "0
    setups found".
  * No hard-coded result columns: whatever the strategy emits on its last row
    is carried through.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data_loader import DataLoader
from src.strategy import Strategy

log = logging.getLogger(__name__)


class Screener:
    def __init__(self, data_loader: DataLoader, start_date: str = "2020-01-01",
                 strict: bool = False, progress_cb=None):
        self.loader = data_loader
        self.start_date = start_date
        self.strict = strict  # if True, re-raise strategy errors (dev mode)
        # Optional callback(done, total, matched) for progress reporting (e.g.
        # the web UI). Logging happens regardless.
        self.progress_cb = progress_cb

    def scan(self, strategy: Strategy, tickers: list[str],
             market_label: str = "US",
             benchmark: "pd.Series | None" = None) -> pd.DataFrame:
        meta = strategy.meta
        log.info("[%s] Scanning %d candidates with %s",
                 market_label, len(tickers), strategy.name)

        # Optional: hand the strategy the market's benchmark (for relative
        # strength). Only strategies that declare set_benchmark use it; others
        # are unaffected. Fetched once per market, not per ticker.
        if benchmark is not None and hasattr(strategy, "set_benchmark"):
            strategy.set_benchmark(benchmark)

        results: list[dict] = []
        errors = 0
        scanned = 0                       # tickers that produced a usable frame
        cond_pass: dict[str, int] = {}    # per-filter pass counts (last bar)

        total = len(tickers)
        # Log progress roughly every 2%, at least every 10 tickers, so a long
        # full-market scan (thousands of names) shows a % instead of going dark.
        step = max(10, total // 50) if total else 1

        for done, ticker in enumerate(tickers, start=1):
            if done % step == 0 or done == total:
                pct = 100 * done / total if total else 100
                log.info("[%s] Progress: %d/%d (%.0f%%) — %d matched so far",
                         market_label, done, total, pct, len(results))
                if self.progress_cb is not None:
                    try:
                        self.progress_cb(done, total, len(results))
                    except Exception:  # noqa: BLE001 — never let a callback break the scan
                        pass

            df = self.loader.get_data(ticker, start_date=self.start_date)
            if df is None or df.empty:
                continue

            try:
                signals = strategy.generate_signals(df)
            except Exception:  # noqa: BLE001
                errors += 1
                log.exception("Strategy error on %s", ticker)
                if self.strict:
                    raise
                continue

            if signals is None or signals.empty or len(signals) < 2:
                continue
            if meta.signal_column not in signals.columns:
                log.error("%s: strategy did not emit signal column %r; skipping.",
                          ticker, meta.signal_column)
                if self.strict:
                    raise KeyError(meta.signal_column)
                continue

            last = signals.iloc[-1]
            scanned += 1

            # Tally each diagnostic Cond_* column on the last bar.
            for col in signals.columns:
                if col.startswith("Cond_"):
                    cond_pass[col] = cond_pass.get(col, 0) + (1 if bool(last[col]) else 0)

            if int(last[meta.signal_column]) not in meta.hit_values:
                continue

            row = {"Market": market_label, "Ticker": ticker}
            if "Close" in last.index:
                row["Price"] = last["Close"]
            # Carry through every column the strategy declared for display.
            for col in meta.display_columns:
                if col in last.index:
                    row[col] = last[col]
            results.append(row)

        if errors:
            log.warning("[%s] %d/%d tickers errored during scan.",
                        market_label, errors, len(tickers))

        # Diagnostic summary: how many of the scanned tickers passed each filter
        # on the latest bar. The filter with the LOWEST pass count is the one
        # rejecting the most candidates — relax that one first.
        if cond_pass and scanned:
            log.info("[%s] Filter pass counts (of %d scanned, %d matched all):",
                     market_label, scanned, len(results))
            for col, cnt in sorted(cond_pass.items(), key=lambda kv: kv[1]):
                name = col[len("Cond_"):]
                log.info("    %-20s %4d / %d  (%.0f%%)",
                         name, cnt, scanned, 100 * cnt / scanned)

        return pd.DataFrame(results)
