"""
market_us.py
------------
US ticker sourcing for the runner.

Public interface (consumed by run.py):
    collect_us_candidates(top_n, perf_col) -> (tickers: list[str], meta_df: DataFrame)

meta_df carries [Ticker, Sector, Industry] for enrichment. The runner maps those
columns defensively, so a partial meta_df (e.g. Finviz dropped a column) will not
crash the run.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.finviz_engine import FinvizEngine

log = logging.getLogger(__name__)


def collect_us_by_sector(top_n: int = 3,
                         perf_col: str = "Perf Week") -> tuple[list[str], pd.DataFrame]:
    """Collect candidates from the top-N performing broad SECTORS (~11 exist)."""
    finviz = FinvizEngine()
    sectors = finviz.get_top_sectors(top_n=top_n, col_target=perf_col)
    if not sectors:
        log.error("Could not obtain sectors from Finviz.")
        return [], pd.DataFrame(columns=["Ticker", "Sector", "Industry"])
    log.info("Top sectors: %s", sectors)
    frames = []
    for sector in sectors:
        details = finviz.get_ticker_details_in_sector(sector)
        if details is not None and not details.empty:
            frames.append(details)
    if not frames:
        log.warning("No tickers for the selected sectors.")
        return [], pd.DataFrame(columns=["Ticker", "Sector", "Industry"])
    meta_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="Ticker")
    tickers = meta_df["Ticker"].tolist()
    log.info("Total US candidates (sectors): %d", len(tickers))
    return tickers, meta_df


def collect_explicit_tickers(tickers: list[str]) -> tuple[list[str], pd.DataFrame]:
    """
    Pass-through for a user-supplied list of tickers (e.g. ["RACE", "ATKR"]).
    No Finviz lookup; Sector/Industry are left blank (filled later if known).
    """
    clean = [t.strip().upper() for t in tickers if t and t.strip()]
    clean = list(dict.fromkeys(clean))  # dedupe, keep order
    meta_df = pd.DataFrame({"Ticker": clean, "Sector": "", "Industry": ""})
    log.info("Explicit tickers to scan: %s", clean)
    return clean, meta_df


def collect_us_candidates(top_n: int = 0,
                          perf_col: str = "Perf Week") -> tuple[list[str], pd.DataFrame]:
    """
    Collect US candidates. By default (top_n <= 0) scans the FULL market —
    every liquid stock, no industry pre-filter. Pass a positive top_n to
    restrict to the best-performing industries instead.
    """
    finviz = FinvizEngine()

    if top_n is None or top_n <= 0:
        meta_df = finviz.get_all_market_details()
        if meta_df.empty:
            log.error("Could not obtain the market universe from Finviz.")
            return [], pd.DataFrame(columns=["Ticker", "Sector", "Industry"])
        tickers = meta_df["Ticker"].tolist()
        log.info("Total US candidates (full market): %d", len(tickers))
        return tickers, meta_df

    industries = finviz.get_top_industries(top_n=top_n, col_target=perf_col)
    if not industries:
        log.error("Could not obtain industries from Finviz.")
        return [], pd.DataFrame(columns=["Ticker", "Sector", "Industry"])
    log.info("Top industries: %s", industries)

    frames: list[pd.DataFrame] = []
    for industry in industries:
        details = finviz.get_ticker_details_in_industry(industry)
        if details is not None and not details.empty:
            frames.append(details)

    if not frames:
        log.warning("No tickers for the selected industries.")
        return [], pd.DataFrame(columns=["Ticker", "Sector", "Industry"])

    meta_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="Ticker")
    tickers = meta_df["Ticker"].tolist()
    log.info("Total US candidates: %d", len(tickers))
    return tickers, meta_df
