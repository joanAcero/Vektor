"""
market_config.py
----------------
Metadata for international markets. This does NOT hardcode ticker lists. Each
entry describes WHERE to fetch the live constituent universe and HOW to turn a
local symbol into a yfinance ticker. The list of stocks is fetched at runtime
(see market_intl.py), so newly-added index members are picked up automatically.

Per-market fields:
  name        : human-readable market name.
  suffix      : appended to a local symbol to form the yfinance ticker
                (e.g. "SAN" + ".MC" -> "SAN.MC"). Not appended when the page
                already writes it (Swiss pages write "NESN.SW").
  wiki_url    : English-Wikipedia page listing the index constituents.
  symbol_cols : ordered candidate column names that may hold the local ticker.
                The first one present on the page is used.
  name_col    : column holding the company name (fallback / sector-less label).
  min_rows    : the smallest row count a table may have and still be believed
                to BE the constituents table. Set well UNDER the true index
                size: it exists to reject a small look-alike table, not to
                assert the exact membership count.

WHY min_rows EXISTS (and why `table_match` no longer does)
=========================================================
These pages carry several tables -- index changes, annual performance,
milestones, navboxes. Picking "the first table that has a ticker-ish column"
is how a scrape starts silently returning the wrong list, so the rule is the
same one src/indices.py already applies to the US pages: a table must have a
ticker column AND enough rows, and among the qualifying tables the LARGEST
wins.

The old `table_match` fallback was dead code with a failure mode: when no
table had a symbol column it returned a table chosen by a substring match on
its headers, market_intl.py then read a column that did not exist, and every
row was skipped. The result was an empty market with NO warning -- a broken
page and a market with no setups looked identical. Removed rather than fixed:
`min_rows` covers the real case.

WHY THIS IS STILL FRAGILE (say so in the README too): Wikipedia tables are
community-edited HTML. Column names and layouts change without notice, and
not every page exposes a ticker column. When a page can't be parsed, that ONE
market degrades to an empty list (logged loudly), the others keep working, and
you update the hints here. That is the documented cost of a zero-API-key,
scraped source.
"""

from __future__ import annotations

# Order roughly by how reliable the Wikipedia table's ticker column tends to be.
MARKETS: dict[str, dict] = {
    "DE": {
        "name": "Germany — DAX 40 (XETRA)",
        "suffix": ".DE",
        "wiki_url": "https://en.wikipedia.org/wiki/DAX",
        "symbol_cols": ["Ticker", "Symbol"],
        "name_col": "Company",
        "min_rows": 30,
    },
    "GB": {
        "name": "United Kingdom — FTSE 100 (LSE)",
        "suffix": ".L",
        "wiki_url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "symbol_cols": ["Ticker", "EPIC", "Symbol"],
        "name_col": "Company",
        "min_rows": 80,
    },
    "FR": {
        "name": "France — CAC 40 (Euronext Paris)",
        "suffix": ".PA",
        "wiki_url": "https://en.wikipedia.org/wiki/CAC_40",
        "symbol_cols": ["Ticker", "Symbol"],
        "name_col": "Company",
        "min_rows": 30,
    },
    "CH": {
        # The SMI constituents table exposes Rank | Name | Industry | Ticker |
        # Canton, and writes the ticker WITH the Yahoo suffix ("NESN.SW").
        # full_ticker() below already refuses to double-append, so both
        # conventions parse.
        #
        # KNOWN LIMITATION, not a bug: this is 20 names, and Nestlé, Roche and
        # Novartis are more than half its capitalisation. As a SCREENING
        # universe it is thin -- fine as a watchlist of Swiss blue chips,
        # useless for finding Stage 1->2 transitions you had not already heard
        # of. The 50-name SMI Expanded is SMI + SMI MID, published as two
        # separate Wikipedia pages, so covering it needs this schema to accept
        # several URLs per market. See the note in the delivery message.
        "name": "Switzerland — SMI 20 (SIX)",
        "suffix": ".SW",
        "wiki_url": "https://en.wikipedia.org/wiki/Swiss_Market_Index",
        "symbol_cols": ["Ticker", "Symbol"],
        "name_col": "Name",
        "min_rows": 15,
    },
    "IT": {
        "name": "Italy — FTSE MIB (Borsa Italiana)",
        "suffix": ".MI",
        "wiki_url": "https://en.wikipedia.org/wiki/FTSE_MIB",
        "symbol_cols": ["Ticker", "Symbol"],
        "name_col": "Company",
        "min_rows": 30,
    },
    "ES": {
        # The IBEX 35 page historically lacks a clean ticker column, so
        # symbol_cols may not match; market_intl.py logs and skips if so. Kept
        # here so the market stays selectable and is easy to fix when the page
        # changes. (You can also point wiki_url at a page that does list
        # tickers.)
        "name": "Spain — IBEX 35 (BME)",
        "suffix": ".MC",
        "wiki_url": "https://en.wikipedia.org/wiki/IBEX_35",
        "symbol_cols": ["Ticker", "Symbol"],
        "name_col": "Company",
        "min_rows": 25,
    },
}

# Smallest row count accepted when a market omits `min_rows`. Deliberately
# non-zero: an unguarded market is the case this whole mechanism exists to
# prevent.
DEFAULT_MIN_ROWS = 10


class MarketConfig:
    """Thin accessor over one market's metadata."""

    def __init__(self, code: str):
        if code not in MARKETS:
            raise KeyError(f"Unknown market code {code!r}. Known: {sorted(MARKETS)}")
        self.code = code
        self._cfg = MARKETS[code]

    @property
    def name(self) -> str:
        return self._cfg.get("name", self.code)

    @property
    def suffix(self) -> str:
        return self._cfg.get("suffix", "")

    @property
    def wiki_url(self) -> str:
        return self._cfg["wiki_url"]

    @property
    def symbol_cols(self) -> list[str]:
        return list(self._cfg.get("symbol_cols", []))

    @property
    def name_col(self) -> str:
        return self._cfg.get("name_col", "Company")

    @property
    def min_rows(self) -> int:
        return int(self._cfg.get("min_rows", DEFAULT_MIN_ROWS))

    def full_ticker(self, base: str) -> str:
        # Wikipedia tables are inconsistent: some list a bare local symbol
        # ("ADS"), others already include the yfinance suffix ("ADS.DE",
        # "NESN.SW"). Appending unconditionally produced "ADS.DE.DE", which
        # Yahoo rejects. So only append the suffix when it isn't already there.
        clean = str(base).strip().split()[0].replace("\xa0", "")
        if not self.suffix or clean.endswith(self.suffix):
            return clean
        return f"{clean}{self.suffix}"
