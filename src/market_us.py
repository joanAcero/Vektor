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


def collect_us_by_rotation(loader,
                           quadrants: Sequence[str]) -> tuple[list[str], pd.DataFrame]:
    """
    Sector-rotation-driven candidate sourcing: run the sector money-flow
    monitor first (src/rotation.py::sector_rotation), keep only the sectors
    currently sitting in one of `quadrants`, and collect every liquid ticker
    Finviz lists under those sectors.
    """
    from src.rotation import sector_rotation

    empty = pd.DataFrame(columns=["Ticker", "Sector", "Industry"])
    wanted = list(quadrants)
    if not wanted:
        log.error("collect_us_by_rotation called with an empty quadrant list; "
                  "check market.us_rotation_quadrants in your config.")
        return [], empty

    finviz = FinvizEngine()

    sectors = sector_rotation(loader)
    if not sectors:
        log.error("Sector rotation returned nothing; cannot drive candidate "
                  "selection.")
        return [], empty

    picked = [s for s in sectors if s["quadrant"] in wanted]
    if not picked:
        # Legitimate outcome, not an error: in a strong trend every sector can
        # sit in one or two quadrants. Log what WAS there so an empty scan is
        # distinguishable from a misconfigured one.
        census: dict[str, int] = {}
        for s in sectors:
            census[s["quadrant"]] = census.get(s["quadrant"], 0) + 1
        log.info("No sectors currently in quadrant(s) %s; nothing to scan. "
                 "Present this week: %s", wanted,
                 ", ".join(f"{n} {q}" for q, n in sorted(census.items())))
        return [], empty

    log.info("Rotation-driven sectors (%s): %s", "/".join(wanted),
             [f"{s['etf']} {s['sector']} [{s['sector_stage']}]" for s in picked])

    frames: list[pd.DataFrame] = []
    for s in picked:
        finviz_sector = FINVIZ_SECTOR_NAME.get(s["sector"])
        if finviz_sector is None:
            log.warning("No Finviz-name mapping for sector %r; skipping it. Add "
                        "it to FINVIZ_SECTOR_NAME in market_us.py if this is a "
                        "real sector.", s["sector"])
            continue
        details = finviz.get_ticker_details_in_sector(finviz_sector)
        if details is not None and not details.empty:
            frames.append(details)
        else:
            log.warning("No tickers returned for sector %r (Finviz name %r) -- "
                        "verify FINVIZ_SECTOR_NAME against Finviz's actual "
                        "sector labels if this looks wrong.",
                        s["sector"], finviz_sector)

    if not frames:
        log.warning("No tickers found across rotation-selected sectors.")
        return [], empty

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



SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Wikipedia writes class shares with a dot (BRK.B, BF.B); Yahoo wants a hyphen
# (BRK-B, BF-B). Silently wrong tickers download as empty frames and vanish
# from the scan with no error, so the translation is explicit and logged.
_SP500_SYMBOL_FIXUPS = str.maketrans({".": "-"})


def collect_us_sp500() -> tuple[list[str], pd.DataFrame]:
    """
    The S&P 500 constituent list, fetched live from Wikipedia.

    A fixed, liquid, survivorship-free-going-forward universe: no Finviz
    performance ranking, no sector rotation, no stage filter. Useful as a
    CONTROL. Every other US source narrows the universe before the strategy
    sees it, which means a change in results can come from the selector rather
    than the strategy. This one does not move, so run it when you want to know
    whether a rule change actually changed anything.

    Returns (tickers, meta_df) in the same shape as the other US collectors,
    so run.py's enrichment step needs no special case.

    TWO CAVEATS, both worth knowing before you compare outputs across sources:

    * SECTOR VOCABULARY DIFFERS. The Sector column here holds GICS names from
      Wikipedia ("Information Technology", "Consumer Discretionary"); the
      Finviz-driven sources hold Finviz's names ("Technology", "Consumer
      Cyclical"). Sector is display metadata rather than a scan filter, so
      nothing breaks, but do not join or group across sources on it.

    * IT IS A SCRAPED SOURCE. Wikipedia tables are community-edited HTML and
      the column names change without notice. When that happens this raises
      with the columns it actually found, rather than returning a short list
      that would look like a quiet market.
    """
    from src.market_intl import read_wiki_tables

    empty = pd.DataFrame(columns=["Ticker", "Sector", "Industry"])

    try:
        tables = read_wiki_tables(SP500_WIKI_URL)
    except Exception:  # noqa: BLE001
        log.exception("Could not fetch the S&P 500 constituents page (%s).",
                      SP500_WIKI_URL)
        return [], empty

    df = next((t for t in tables if "Symbol" in t.columns), None)
    if df is None:
        found = [list(t.columns)[:6] for t in tables[:3]]
        log.error("No S&P 500 table with a 'Symbol' column at %s. Wikipedia's "
                  "layout has probably changed; first tables seen: %s",
                  SP500_WIKI_URL, found)
        return [], empty

    sector_col = "GICS Sector" if "GICS Sector" in df.columns else None
    industry_col = ("GICS Sub-Industry" if "GICS Sub-Industry" in df.columns
                    else None)
    if sector_col is None:
        log.warning("S&P 500 table has no 'GICS Sector' column; Sector will be "
                    "blank. Columns present: %s", list(df.columns))

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    translated = 0
    for _, r in df.iterrows():
        raw = r.get("Symbol")
        if pd.isna(raw):
            continue
        ticker = str(raw).strip().replace("\xa0", "").upper()
        if not ticker:
            continue
        fixed = ticker.translate(_SP500_SYMBOL_FIXUPS)
        if fixed != ticker:
            translated += 1
        if fixed in seen:
            continue
        seen.add(fixed)
        rows.append({
            "Ticker": fixed,
            "Sector": str(r[sector_col]) if sector_col and not pd.isna(r.get(sector_col)) else "",
            "Industry": str(r[industry_col]) if industry_col and not pd.isna(r.get(industry_col)) else "",
        })

    if not rows:
        log.warning("S&P 500 table parsed but yielded no tickers.")
        return [], empty

    meta_df = pd.DataFrame(rows)
    tickers = meta_df["Ticker"].tolist()

    if translated:
        log.info("S&P 500: rewrote %d class-share symbol(s) from Wikipedia's "
                 "dot form to Yahoo's hyphen form (e.g. BRK.B -> BRK-B).",
                 translated)
    if len(tickers) < 450:
        # A partial parse looks exactly like a quiet market downstream. Say so.
        log.warning("S&P 500: only %d constituents parsed (expected ~503). The "
                    "Wikipedia table may have changed shape.", len(tickers))
    log.info("Total US candidates (S&P 500): %d", len(tickers))
    return tickers, meta_df


def collect_us_named_group(kind: str, name: str) -> tuple[list[str], pd.DataFrame]:
    """
    Collect every liquid ticker in ONE named Finviz sector or industry.

    The counterpart to collect_us_by_sector / collect_us_candidates, which pick
    groups FOR you by performance. This is for when you already know which
    group you want to look inside — "show me every basing name in Utilities"
    is a different question from "show me the strongest groups", and answering
    it by setting top_n high enough that Utilities happens to be included also
    drags in everything above it.

    `kind` is "sector" or "industry"; `name` is Finviz's own label for the
    group, exactly as it appears in its group screener. It is NOT validated
    against a list here: Finviz owns that vocabulary, it changes, and a
    hardcoded copy is the same trap FINVIZ_SECTOR_NAME above already documents.
    A name Finviz does not recognise comes back as an empty result and is
    reported as such — visible and cheap to fix — rather than silently
    scanning something else.
    """
    kind = str(kind).strip().lower()
    name = str(name).strip()
    if kind not in ("sector", "industry"):
        raise ValueError(f"kind must be 'sector' or 'industry', got {kind!r}")
    if not name:
        raise ValueError("collect_us_named_group() needs a group name.")

    finviz = FinvizEngine()
    log.info("Scanning a single %s: %s", kind, name)
    details = (finviz.get_ticker_details_in_sector(name) if kind == "sector"
               else finviz.get_ticker_details_in_industry(name))

    if details is None or details.empty:
        log.warning("Finviz returned no tickers for %s %r. Check the spelling "
                    "against Finviz's own group names — an unknown group and an "
                    "empty group look identical from here.", kind, name)
        return [], empty_details()

    meta_df = details.drop_duplicates(subset="Ticker")
    tickers = meta_df["Ticker"].tolist()
    log.info("Total US candidates (%s=%s): %d", kind, name, len(tickers))
    return tickers, meta_df


def collect_us_named_group(kind: str, name: str) -> tuple[list[str], pd.DataFrame]:
    """
    Collect every liquid ticker in ONE named Finviz sector or industry.

    The counterpart to collect_us_by_sector / collect_us_candidates, which pick
    groups FOR you by performance. This is for when you already know which
    group you want to look inside — "show me every basing name in Utilities"
    is a different question from "show me the strongest groups", and answering
    it by setting top_n high enough that Utilities happens to be included also
    drags in everything above it.

    `kind` is "sector" or "industry"; `name` is Finviz's own label for the
    group, exactly as it appears in its group screener. It is NOT validated
    against a list here: Finviz owns that vocabulary, it changes, and a
    hardcoded copy is the same trap FINVIZ_SECTOR_NAME above already documents.
    A name Finviz does not recognise comes back as an empty result and is
    reported as such — visible and cheap to fix — rather than silently
    scanning something else.
    """
    kind = str(kind).strip().lower()
    name = str(name).strip()
    if kind not in ("sector", "industry"):
        raise ValueError(f"kind must be 'sector' or 'industry', got {kind!r}")
    if not name:
        raise ValueError("collect_us_named_group() needs a group name.")

    finviz = FinvizEngine()
    log.info("Scanning a single %s: %s", kind, name)
    details = (finviz.get_ticker_details_in_sector(name) if kind == "sector"
               else finviz.get_ticker_details_in_industry(name))

    if details is None or details.empty:
        log.warning("Finviz returned no tickers for %s %r. Check the spelling "
                    "against Finviz's own group names — an unknown group and an "
                    "empty group look identical from here.", kind, name)
        return [], empty_details()

    meta_df = details.drop_duplicates(subset="Ticker")
    tickers = meta_df["Ticker"].tolist()
    log.info("Total US candidates (%s=%s): %d", kind, name, len(tickers))
    return tickers, meta_df

