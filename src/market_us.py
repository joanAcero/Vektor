"""
market_us.py
------------
US ticker sourcing for the runner.

Public interface (consumed by run.py):
    collect_us_candidates(top_n, perf_col) -> (tickers: list[str], meta_df: DataFrame)
    collect_us_by_sector(top_n, perf_col)  -> (tickers, meta_df)
    collect_us_by_rotation(loader, states) -> (tickers, meta_df)
    collect_explicit_tickers(tickers)      -> (tickers, meta_df)

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


# ---------------------------------------------------------------------------
# src/rotation.py's SECTOR_ETFS uses SPDR/GICS-style sector names. Finviz's
# own screener taxonomy differs for five of the eleven -- passing rotation's
# "sector" straight into a Finviz Sector filter silently returns ZERO tickers
# for any of these. This mapping is my best-known match of Finviz's standard
# sector labels, but I cannot hit Finviz from this environment to confirm it.
#
# VERIFY before trusting this: finviz.get_all_market_details()["Sector"].unique()
# and diff against the right-hand column below. Fix any mismatches here.
# ---------------------------------------------------------------------------
FINVIZ_SECTOR_NAME: dict[str, str] = {
    "Technology": "Technology",
    "Energy": "Energy",
    "Financials": "Financial",
    "Health Care": "Healthcare",
    "Industrials": "Industrials",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Utilities": "Utilities",
    "Materials": "Basic Materials",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
}


def collect_us_by_rotation(loader, states: list[str] | tuple[str, ...] = ("Rotating In",),
                           ) -> tuple[list[str], pd.DataFrame]:
    """
    Sector-rotation-driven candidate sourcing: run the sector money-flow
    monitor first (src/rotation.py::sector_rotation), keep only sectors
    currently in one of `states` (default: just "Rotating In" -- the early
    signal), and collect every liquid ticker Finviz lists under those sectors.

    Deliberately sector-level only, not sector -> industry -> ticker:
    src/rotation.py::industry_rotation() currently ignores the sector_name
    argument it's given (it calls finviz.get_top_industries() with no sector
    filter applied at all -- a pre-existing bug, not something this function
    works around). Narrowing further to "strongest industries within this
    sector" isn't something the codebase can reliably do until that's fixed;
    this function only relies on get_ticker_details_in_sector(), which IS
    correctly sector-filtered (collect_us_by_sector already depends on it).
    """
    from src.rotation import sector_rotation

    finviz = FinvizEngine()

    sectors = sector_rotation(loader)
    if not sectors:
        log.error("Sector rotation returned nothing; cannot drive candidate selection.")
        return [], pd.DataFrame(columns=["Ticker", "Sector", "Industry"])

    picked = [s for s in sectors if s["state"] in states]
    if not picked:
        log.info("No sectors currently in state(s) %s; nothing to scan.", list(states))
        return [], pd.DataFrame(columns=["Ticker", "Sector", "Industry"])

    log.info("Rotation-driven sectors (%s): %s",
             "/".join(states), [s["sector"] for s in picked])

    frames: list[pd.DataFrame] = []
    for s in picked:
        finviz_sector = FINVIZ_SECTOR_NAME.get(s["sector"])
        if finviz_sector is None:
            log.warning("No Finviz-name mapping for sector %r; skipping it. Add it to "
                       "FINVIZ_SECTOR_NAME in market_us.py if this is a real sector.",
                       s["sector"])
            continue
        details = finviz.get_ticker_details_in_sector(finviz_sector)
        if details is not None and not details.empty:
            frames.append(details)
        else:
            log.warning("No tickers returned for sector %r (Finviz name %r) -- verify "
                       "FINVIZ_SECTOR_NAME against Finviz's actual sector labels if this "
                       "looks wrong.", s["sector"], finviz_sector)

    if not frames:
        log.warning("No tickers found across rotation-selected sectors.")
        return [], pd.DataFrame(columns=["Ticker", "Sector", "Industry"])

    meta_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="Ticker")
    tickers = meta_df["Ticker"].tolist()
    log.info("Total US candidates (rotation-driven): %d", len(tickers))
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
