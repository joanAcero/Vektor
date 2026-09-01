"""
market_intl.py
--------------
DYNAMIC international ticker sourcing. Fetches live index constituents from
Wikipedia (via pandas.read_html) instead of hardcoded lists, so new index
members are picked up automatically.

Public interface (consumed by run.py, unchanged):
    collect_intl_candidates(codes) -> (tickers, market_map, sector_map)

Design notes:
  * Each market is fetched independently and wrapped in try/except, so one
    broken Wikipedia page degrades that single market to an empty list (logged)
    rather than failing the whole run.
  * Tickers are normalised and suffixed for yfinance (e.g. SAP -> SAP.DE),
    except where the page already writes the suffix (NESN.SW).
  * sector_map is populated when the page exposes a usable sector/industry
    column; otherwise it falls back to the market name. Sector here is
    best-effort metadata, not a scan filter.
  * This is a SCRAPED source with no API key. It can break when Wikipedia
    changes a table's layout; when it does, update src/market_config.py.

TABLE SELECTION IS BY SHAPE, NOT BY POSITION
============================================
Same rule as src/indices.py for the US pages, and for the same reason: these
pages carry several tables and their order is not stable across edits. A table
qualifies only if it has one of the market's symbol columns AND at least
`min_rows` rows; among the qualifying tables the largest wins. A page that
yields nothing raises a warning naming what was actually found, so a layout
change reads as a layout change rather than as a market with no setups.

A small in-memory cache avoids re-fetching the same page within one run.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from src.market_config import MARKETS, MarketConfig

log = logging.getLogger(__name__)

# yfinance is the eventual consumer; a realistic User-Agent reduces the chance
# Wikipedia/storage layers reject an "I'm a bot" default.
_STORAGE_OPTS = {"User-Agent": "Mozilla/5.0 (compatible; VektorScreener/0.1)"}

# Column-name fragments we'll accept as a sector/industry label, lowercased.
_SECTOR_HINTS = ("sector", "industry", "icb")


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column present (case-insensitive)."""
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _find_sector_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if any(h in str(c).lower() for h in _SECTOR_HINTS):
            return c
    return None


@lru_cache(maxsize=None)
def read_wiki_tables(url: str) -> tuple[pd.DataFrame, ...]:
    """Fetch and cache all tables on a Wikipedia page, keyed by URL.

    Public because src/market_us.py fetches the S&P 500 list the same way. One
    function knows how to ask Wikipedia politely (realistic User-Agent) and
    cache the answer for the run; a second copy would drift on one or both.
    """
    tables = pd.read_html(url, storage_options=_STORAGE_OPTS)
    return tuple(tables)


def _select_table(tables: tuple[pd.DataFrame, ...],
                  cfg: MarketConfig) -> pd.DataFrame | None:
    """The constituents table, identified by SHAPE.

    Qualifies: has a symbol column and at least cfg.min_rows rows. Among the
    qualifying tables the largest wins -- on these pages the constituents list
    is the long one and everything that shares its columns (index changes,
    former members) is short.
    """
    best: pd.DataFrame | None = None
    too_small = 0

    for df in tables:
        if _pick_column(df, cfg.symbol_cols) is None:
            continue
        if len(df) < cfg.min_rows:
            too_small += 1
            continue
        if best is None or len(df) > len(best):
            best = df

    if best is None and too_small:
        log.warning("%s: %d table(s) had a symbol column but fewer than %d rows. "
                    "Either the page was trimmed or min_rows is set too high in "
                    "src/market_config.py.", cfg.code, too_small, cfg.min_rows)
    return best


def _fetch_one_market(code: str) -> tuple[list[str], dict[str, str], dict[str, str]]:
    cfg = MarketConfig(code)
    tickers: list[str] = []
    market_map: dict[str, str] = {}
    sector_map: dict[str, str] = {}

    tables = read_wiki_tables(cfg.wiki_url)
    df = _select_table(tables, cfg)
    if df is None:
        log.warning("%s: no constituents table found at %s (looked for columns "
                    "%s in %d table(s); the layout has probably changed). "
                    "Skipping this market.",
                    code, cfg.wiki_url, cfg.symbol_cols, len(tables))
        return tickers, market_map, sector_map

    sym_col = _pick_column(df, cfg.symbol_cols)
    sec_col = _find_sector_column(df)

    for _, row in df.iterrows():
        raw = row.get(sym_col)
        if pd.isna(raw):
            continue
        full = cfg.full_ticker(raw)
        if not full or full in market_map:
            continue
        market_map[full] = code
        sector_map[full] = (str(row[sec_col]) if sec_col and not pd.isna(row.get(sec_col))
                            else cfg.name)
        tickers.append(full)

    log.info("%s: %d constituents fetched.", code, len(tickers))
    return tickers, market_map, sector_map


def collect_intl_candidates(codes: list[str]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    if not codes:
        codes = [c for c in MARKETS if c != "US"]
    log.info("International markets (live fetch): %s", codes)

    all_tickers: list[str] = []
    market_map: dict[str, str] = {}
    sector_map: dict[str, str] = {}

    for code in codes:
        if code not in MARKETS:
            log.warning("Unknown market code %r; skipping.", code)
            continue
        try:
            t, m, s = _fetch_one_market(code)
        except Exception:  # noqa: BLE001 — isolate one bad page from the rest
            log.exception("Failed to fetch constituents for %s; skipping.", code)
            continue
        for tk in t:
            if tk not in market_map:
                all_tickers.append(tk)
                market_map[tk] = m[tk]
                sector_map[tk] = s[tk]

    log.info("Total international candidates: %d", len(all_tickers))
    return all_tickers, market_map, sector_map
