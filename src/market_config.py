"""
market_config.py
----------------
Metadata for international markets. NOTE: this no longer hardcodes ticker
lists. Each entry describes WHERE to fetch the live constituent universe and
HOW to turn a local symbol into a yfinance ticker. The actual list of stocks is
fetched at runtime (see market_intl.py), so newly-added index members are
picked up automatically.

Per-market fields:
  name        : human-readable market name.
  suffix      : appended to a local symbol to form the yfinance ticker
                (e.g. "SAN" + ".MC" -> "SAN.MC").
  wiki_url    : English-Wikipedia page listing the index constituents.
  table_match : a substring that must appear in the target table, used to pick
                the right table on pages that contain several.
  symbol_cols : ordered candidate column names that may hold the local ticker.
                The first one present on the page is used.
  name_col    : column holding the company name (fallback / sector-less label).
  name_col    : column holding the company name (fallback / sector-less label).

WHY THIS IS FRAGILE (be honest in the README too): Wikipedia tables are
community-edited HTML. Column names and layouts change without notice, and not
every page exposes a ticker column. When a page can't be parsed, that ONE market
degrades to an empty list (logged), the others keep working, and you update the
hints here. This is the documented cost of a zero-API-key, scraped source.
"""

from __future__ import annotations

# Order roughly by how reliable the Wikipedia table's ticker column tends to be.
MARKETS: dict[str, dict] = {
    "DE": {
        "name": "Germany — DAX 40 (XETRA)",
        "suffix": ".DE",
        "wiki_url": "https://en.wikipedia.org/wiki/DAX",
        "table_match": "Ticker",
        "symbol_cols": ["Ticker", "Symbol"],
        "name_col": "Company",
    },
    "GB": {
        "name": "United Kingdom — FTSE 100 (LSE)",
        "suffix": ".L",
        "wiki_url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "table_match": "Ticker",
        "symbol_cols": ["Ticker", "EPIC", "Symbol"],
        "name_col": "Company",
    },
    "FR": {
        "name": "France — CAC 40 (Euronext Paris)",
        "suffix": ".PA",
        "wiki_url": "https://en.wikipedia.org/wiki/CAC_40",
        "table_match": "Ticker",
        "symbol_cols": ["Ticker", "Symbol"],
        "name_col": "Company",
    },
    "IT": {
        "name": "Italy — FTSE MIB (Borsa Italiana)",
        "suffix": ".MI",
        "wiki_url": "https://en.wikipedia.org/wiki/FTSE_MIB",
        "table_match": "Ticker",
        "symbol_cols": ["Ticker", "Symbol"],
        "name_col": "Company",
    },
    "ES": {
        # The IBEX 35 Wikipedia page historically lacks a clean ticker column,
        # so symbol_cols may not match; market_intl.py will log and skip if so.
        # Kept here so the market is selectable and easy to fix when the page
        # changes. (You can also point wiki_url at a page that does list tickers.)
        "name": "Spain — IBEX 35 (BME)",
        "suffix": ".MC",
        "wiki_url": "https://en.wikipedia.org/wiki/IBEX_35",
        "table_match": "Ticker",
        "symbol_cols": ["Ticker", "Symbol"],
        "name_col": "Company",
    },
}


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
    def table_match(self) -> str:
        return self._cfg.get("table_match", "")

    @property
    def symbol_cols(self) -> list[str]:
        return list(self._cfg.get("symbol_cols", []))

    @property
    def name_col(self) -> str:
        return self._cfg.get("name_col", "Company")

    def full_ticker(self, base: str) -> str:
        # Wikipedia tables are inconsistent: some list a bare local symbol
        # ("ADS"), others already include the yfinance suffix ("ADS.DE").
        # Appending unconditionally produced "ADS.DE.DE", which Yahoo rejects.
        # So only append the suffix when it isn't already present.
        clean = str(base).strip().split()[0].replace("\xa0", "")
        if not self.suffix or clean.endswith(self.suffix):
            return clean
        return f"{clean}{self.suffix}"
