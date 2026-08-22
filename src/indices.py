"""
indices.py
----------
US index constituents as a scan universe.

Public interface (consumed by run.py):
    collect_us_indices(keys) -> (tickers: list[str], meta_df: DataFrame)
    INDICES                  -> the registry, for the UI's checkbox list

meta_df carries Ticker, Name, Sector, Industry, Market -- the same contract
every other collector returns. Three of the four indices publish GICS sector
and sub-industry alongside the ticker, so this path gives company names and
sectors for free, exactly like the Quality holdings path and unlike the Finviz
screener paths where Industry is often blank.

WHY WIKIPEDIA AND NOT AN ETF
===========================
The obvious source for the S&P SmallCap 600 is a tracking ETF such as Vanguard's
VIOO. Its profile page renders holdings in JavaScript -- the served HTML
contains a title and nothing else -- and Vanguard publishes no documented CSV
endpoint the way iShares does. Scraping it would mean either a headless browser
or an undocumented internal API that can change without notice.

Wikipedia's constituent tables are the better source here for three reasons
beyond mere availability:
  * They name the INDEX, not a fund tracking it. An ETF's holdings differ from
    its index -- sampling, cash, recent creations -- and you are screening the
    index.
  * They carry GICS sector and sub-industry per row, which no ETF file does.
  * The repo already depends on this mechanism for the European indices, so
    this is one parsing path to maintain rather than two.

The cost is honest: Wikipedia can be edited, and a table's column headings do
occasionally get reworded. _pick_table() below therefore identifies the table
by SHAPE (a ticker-like column plus a plausible row count) rather than by
position, and a page that stops parsing fails loudly instead of silently
returning three rows.

S&P 500 IS DELEGATED, NOT REIMPLEMENTED
=======================================
src/market_us.py already has collect_us_sp500(), which works. This module calls
it rather than parsing the S&P 500 page itself -- two implementations of one
universe would eventually disagree, and the existing one has been exercised.
The Wikipedia spec for it below is a FALLBACK used only if that import fails,
so a tree without it still works.

Self-check without running a scan:
    python -m src.indices sp600
    python -m src.indices --all
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path("cache/indices")
# Index membership changes a few times a year (plus quarterly rebalances), so a
# day-old list is not stale in any way that matters. The TTL is about not
# hitting Wikipedia on every scan.
CACHE_MAX_AGE_HOURS = 24.0
REQUEST_TIMEOUT = 30
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; VektorScreener/0.1)"}

# Same contract as src/holdings.py::HOLDING_COLS.
INDEX_COLS = ("Ticker", "Name", "Sector", "Industry", "Market")

# Column headings seen across these pages, in preference order. Matched
# case-insensitively against the table's own headings.
_TICKER_HEADINGS = ("symbol", "ticker", "ticker symbol")
_NAME_HEADINGS = ("security", "company", "company name", "name")
_SECTOR_HEADINGS = ("gics sector", "sector")
_INDUSTRY_HEADINGS = ("gics sub-industry", "gics sub industry", "industry")


@dataclass(frozen=True)
class IndexSpec:
    key: str
    label: str          # shown in the UI
    wiki_url: str
    min_rows: int       # sanity floor for _pick_table; well under the real count
    delegate: str | None = None   # "module:function" preferred over parsing


INDICES: dict[str, IndexSpec] = {
    "sp500": IndexSpec(
        key="sp500",
        label="S&P 500",
        wiki_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        min_rows=400,
        delegate="src.market_us:collect_us_sp500",
    ),
    "nasdaq100": IndexSpec(
        key="nasdaq100",
        label="Nasdaq-100",
        wiki_url="https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
        min_rows=80,
    ),
    "dow30": IndexSpec(
        key="dow30",
        label="Dow Jones Industrial Average",
        wiki_url="https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        min_rows=20,
    ),
    "sp600": IndexSpec(
        key="sp600",
        label="S&P SmallCap 600",
        wiki_url="https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        min_rows=400,
    ),
}


def index_options() -> list[dict[str, str]]:
    """[{key, label}] for the UI, so the checkbox list is never hardcoded in
    the frontend."""
    return [{"key": spec.key, "label": spec.label} for spec in INDICES.values()]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _heading_match(columns, candidates) -> str | None:
    lowered = {str(c).strip().lower(): c for c in columns}
    for want in candidates:
        if want in lowered:
            return lowered[want]
    return None


def _pick_table(tables: list[pd.DataFrame], spec: IndexSpec) -> pd.DataFrame:
    """The constituents table, identified by SHAPE not by position.

    These pages carry several tables -- recent changes, historical components,
    performance -- and their order is not stable across edits. Indexing
    tables[0] is how a scrape silently starts returning the wrong list, so the
    rule is: has a ticker-like column, and has enough rows to plausibly BE the
    index. min_rows sits well under the true count, so a table that has lost
    most of its rows fails rather than passing quietly.
    """
    best = None
    for table in tables:
        ticker_col = _heading_match(table.columns, _TICKER_HEADINGS)
        if ticker_col is None or len(table) < spec.min_rows:
            continue
        if best is None or len(table) > len(best[0]):
            best = (table, ticker_col)
    if best is None:
        raise ValueError(
            f"no table on {spec.wiki_url} had a ticker column and at least "
            f"{spec.min_rows} rows — the page layout has probably changed")
    return best[0].rename(columns={best[1]: "Ticker"})


def _normalise_ticker(raw) -> str | None:
    """Wikipedia's symbol -> Yahoo's. BRK.B -> BRK-B, as in the US holdings
    path; anything with whitespace or a footnote marker is not a symbol."""
    text = str(raw).strip().upper()
    if not text or text in ("-", "NAN", "NONE"):
        return None
    text = text.split("[")[0].strip()   # drop [1]-style citation markers
    if " " in text or "/" in text:
        return None
    return text.replace(".", "-")


def _to_frame(table: pd.DataFrame) -> pd.DataFrame:
    name_col = _heading_match(table.columns, _NAME_HEADINGS)
    sector_col = _heading_match(table.columns, _SECTOR_HEADINGS)
    industry_col = _heading_match(table.columns, _INDUSTRY_HEADINGS)

    rows, skipped = [], []
    for _, row in table.iterrows():
        ticker = _normalise_ticker(row["Ticker"])
        if ticker is None:
            skipped.append(str(row["Ticker"]))
            continue
        rows.append({
            "Ticker": ticker,
            "Name": str(row[name_col]).strip() if name_col else "",
            "Sector": str(row[sector_col]).strip() if sector_col else "",
            "Industry": str(row[industry_col]).strip() if industry_col else "",
            "Market": "US",
        })
    if skipped:
        log.debug("Skipped %d unparseable symbol(s): %s",
                  len(skipped), ", ".join(skipped[:10]))
    return pd.DataFrame(rows, columns=list(INDEX_COLS)).drop_duplicates(subset="Ticker")


def _fetch_wikipedia(spec: IndexSpec) -> pd.DataFrame:
    resp = requests.get(spec.wiki_url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    frame = _to_frame(_pick_table(tables, spec))
    if len(frame) < spec.min_rows:
        raise ValueError(
            f"{spec.label}: parsed only {len(frame)} symbols, expected at least "
            f"{spec.min_rows}")
    return frame


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.csv"


def _delegated(spec: IndexSpec) -> pd.DataFrame | None:
    """Use the existing collector when there is one. See the module docstring:
    one universe, one implementation."""
    if not spec.delegate:
        return None
    module_name, func_name = spec.delegate.split(":")
    try:
        module = __import__(module_name, fromlist=[func_name])
        func = getattr(module, func_name)
    except (ImportError, AttributeError) as e:
        log.warning("%s: %s unavailable (%s); parsing %s instead.",
                    spec.key, spec.delegate, e, spec.wiki_url)
        return None
    _, meta_df = func()
    if meta_df is None or meta_df.empty:
        log.warning("%s: %s returned nothing; parsing %s instead.",
                    spec.key, spec.delegate, spec.wiki_url)
        return None
    out = meta_df.copy()
    for col in INDEX_COLS:
        if col not in out.columns:
            out[col] = "US" if col == "Market" else ""
    return out[list(INDEX_COLS)]


def fetch_index(key: str, *, force: bool = False) -> pd.DataFrame:
    """Constituents of one index, from cache when fresh, else downloaded.

    A failed download falls back to the cached copy REGARDLESS of age, with a
    warning: last month's constituents are a far better answer than no scan.
    """
    spec = INDICES.get(key)
    if spec is None:
        raise KeyError(f"Unknown index {key!r}. Known: {sorted(INDICES)}")

    cache = _cache_path(key)
    fresh = (cache.exists()
             and (time.time() - cache.stat().st_mtime) < CACHE_MAX_AGE_HOURS * 3600)
    if fresh and not force:
        try:
            return pd.read_csv(cache).fillna("")
        except (OSError, ValueError, pd.errors.ParserError) as e:
            log.warning("%s: cached list unusable (%s); refetching.", key, e)

    frame = None
    try:
        frame = _delegated(spec)
        if frame is None:
            frame = _fetch_wikipedia(spec)
    except Exception as e:  # noqa: BLE001 — network, parse and shape failures alike
        log.error("%s: could not obtain constituents: %s", key, e)

    if frame is not None and not frame.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
        return frame

    if cache.exists():
        log.warning("%s: using the CACHED constituent list — the fetch failed. "
                    "Membership may be out of date.", key)
        return pd.read_csv(cache).fillna("")

    raise RuntimeError(
        f"Could not obtain the constituents of {spec.label} and no cached copy "
        f"exists. Check {spec.wiki_url} still carries a table with a Symbol "
        f"column, and adjust INDICES['{key}'] in src/indices.py if it moved.")


def collect_us_indices(keys) -> tuple[list[str], pd.DataFrame]:
    """Union of the selected indices — the collector contract run.py expects.

    Overlap is real and expected: every Dow member is in the S&P 500, and most
    of the Nasdaq-100 is too. Deduplicating on Ticker means selecting both
    costs nothing and scans nothing twice.
    """
    wanted = [str(k).strip().lower() for k in (keys or []) if str(k).strip()]
    unknown = [k for k in wanted if k not in INDICES]
    if unknown:
        raise ValueError(f"Unknown index/indices {unknown}. "
                         f"Known: {sorted(INDICES)}")
    if not wanted:
        log.error("No index selected; nothing to scan.")
        return [], pd.DataFrame(columns=list(INDEX_COLS))

    frames = []
    for key in dict.fromkeys(wanted):          # dedupe, keep order
        try:
            frame = fetch_index(key)
        except (KeyError, RuntimeError) as e:
            # One failing index must not lose the others: a scan of three out
            # of four with a loud warning beats no scan at all.
            log.error("%s: skipped (%s)", key, e)
            continue
        log.info("%s: %d constituent(s).", INDICES[key].label, len(frame))
        frames.append(frame)

    if not frames:
        return [], pd.DataFrame(columns=list(INDEX_COLS))

    meta_df = (pd.concat(frames, ignore_index=True)
                 .drop_duplicates(subset="Ticker")
                 .reset_index(drop=True))
    tickers = meta_df["Ticker"].tolist()
    log.info("Total US candidates (indices %s): %d after dedupe.",
             ", ".join(wanted), len(tickers))
    return tickers, meta_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s: %(message)s")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    keys = sorted(INDICES) if ("--all" in sys.argv or not args) else args
    for k in keys:
        frame = fetch_index(k, force=force)
        print(f"\n{INDICES[k].label} — {len(frame)} constituents")
        print(frame.head(10).to_string(index=False))
        blanks = (frame["Sector"].astype(str).str.strip() == "").sum()
        print(f"  ... rows without a sector: {blanks}")
