"""
finviz_engine.py
----------------
Wraps finvizfinance for US-market industry/sector data. Ported from the
original with two changes: print() -> logging, and defensive handling so a
Finviz schema change degrades gracefully instead of raising KeyError deep in
the scan (original bug #4).

Ticker detail frames now carry the COMPANY NAME. The screener Overview view has
always returned it in a column called "Company"; the previous code selected
["Ticker", "Sector", "Industry"] and threw it away. Keeping it costs no extra
request.

NAMING COLLISION, worth knowing before editing this file: the GROUP performance
view (Performance().screener_view(group=...), used by _get_group_top) also has a
column called "Name", but there it is the SECTOR or INDUSTRY name, not a company.
The two frames never meet -- _normalise_details is only applied to screener
results -- but do not "unify" them.
"""

from __future__ import annotations

import logging

import pandas as pd
from finvizfinance.group.performance import Performance
from finvizfinance.screener.overview import Overview as ScreenerOverview

log = logging.getLogger(__name__)

_LIQUIDITY_FILTERS = {
    "Market Cap.": "+Small (over $300mln)",
    "Average Volume": "Over 300K",
}

# The metadata contract for every ticker-detail frame this module returns.
# src/market_us.py imports this so the shape is declared in exactly one place.
DETAIL_COLS = ("Ticker", "Name", "Sector", "Industry")


def empty_details() -> pd.DataFrame:
    """An empty frame with the detail contract's columns."""
    return pd.DataFrame(columns=list(DETAIL_COLS))


class FinvizEngine:

    @staticmethod
    def _clean_pct(x):
        if isinstance(x, str):
            return float(x.strip("%")) / 100
        return x

    def _get_group_top(self, group: str, top_n: int, col_target: str,
                       label: str) -> list[str]:
        log.info("Querying %s on Finviz (Performance view)...", label)
        try:
            df = Performance().screener_view(group=group)
        except Exception as e:  # noqa: BLE001
            log.error("Finviz query failed (%s): %s", label, e)
            return []

        if col_target not in df.columns:
            fallback = next(
                (c for c in ("Perf Quart", "Perf Month", "Perf Year")
                 if c in df.columns), None)
            if fallback is None:
                log.error("No usable performance column in Finviz output.")
                return []
            log.warning("Column %r missing; falling back to %r.", col_target, fallback)
            col_target = fallback

        df[col_target] = df[col_target].apply(self._clean_pct)
        df_sorted = df.sort_values(by=col_target, ascending=False)
        # "Name" here is the sector/industry name -- see the module docstring.
        return df_sorted["Name"].head(top_n).tolist()

    @staticmethod
    def _fix_ticker_dup(df: pd.DataFrame) -> pd.DataFrame:
        """
        Work around a known finvizfinance parsing bug where every ticker comes
        back with its FIRST LETTER DUPLICATED (e.g. MSFT -> MMSFT, KLAC -> KKLAC,
        NOK -> NNOK). The bug is systematic — it affects every row — so we only
        apply the fix when the vast majority of rows show the doubled-first-char
        pattern, which avoids mangling a legitimate ticker in a clean DataFrame.

        A ticker is "doubled" if it has >=2 chars and ticker[0] == ticker[1].
        We additionally require the collapsed form to be non-empty. Because some
        real tickers *could* legitimately start with a repeated letter, we guard
        by the systematic-majority check rather than fixing individual rows.
        """
        if df.empty or "Ticker" not in df.columns:
            return df
        tickers = df["Ticker"].astype(str)
        doubled = tickers.str.len().ge(2) & (tickers.str[0] == tickers.str[1])
        frac = float(doubled.mean()) if len(tickers) else 0.0
        # If most rows are doubled, this is the bug — collapse the first char on
        # the affected rows. (In a clean DataFrame frac ~ 0 and we do nothing.)
        if frac >= 0.8:
            log.warning("finvizfinance first-letter-duplication bug detected "
                        "(%.0f%% of tickers); collapsing.", frac * 100)
            fixed = tickers.where(~doubled, tickers.str[1:])
            df = df.copy()
            df["Ticker"] = fixed
        return df

    @staticmethod
    def _normalise_details(df: pd.DataFrame) -> pd.DataFrame:
        """Reduce a screener frame to the DETAIL_COLS contract.

        The company name arrives as "Company"; anything the view did not return
        is created empty, so downstream code can index the columns
        unconditionally and Instrument.label() just omits what is missing.
        """
        if df is None or df.empty:
            return empty_details()
        out = df.copy()
        if "Name" not in out.columns and "Company" in out.columns:
            out = out.rename(columns={"Company": "Name"})
        for col in DETAIL_COLS:
            if col not in out.columns:
                log.debug("Finviz result had no %r column; filling empty.", col)
                out[col] = ""
        return out[list(DETAIL_COLS)]

    def _screener_filter(self, filters_dict: dict) -> pd.DataFrame:
        try:
            fs = ScreenerOverview()
            fs.set_filter(filters_dict=filters_dict)
            df = fs.screener_view()
        except Exception as e:  # noqa: BLE001
            log.error("Finviz screener failed: %s", e)
            return pd.DataFrame()
        # finvizfinance returns None (and prints "No ticker found.") when a
        # filter matches nothing — no exception is raised. Normalise to an empty
        # DataFrame so every caller can rely on a real DataFrame.
        if df is None:
            return pd.DataFrame()
        return self._fix_ticker_dup(df)

    def get_top_industries(self, top_n: int = 5,
                           col_target: str = "Perf Week") -> list[str]:
        return self._get_group_top("Industry", top_n, col_target, "Industries")

    def get_top_sectors(self, top_n: int = 3,
                        col_target: str = "Perf Week") -> list[str]:
        return self._get_group_top("Sector", top_n, col_target, "Sectors")

    def get_ticker_details_in_sector(self, sector_name: str) -> pd.DataFrame:
        """Return DETAIL_COLS for liquid stocks in a sector."""
        log.info("Fetching ticker details for sector: %s", sector_name)
        df = self._screener_filter({"Sector": sector_name, **_LIQUIDITY_FILTERS})
        result = self._normalise_details(df)
        if result.empty:
            return result
        result["Sector"] = sector_name  # normalise
        return result

    def get_ticker_details_in_industry(self, industry_name: str) -> pd.DataFrame:
        """Return DETAIL_COLS for liquid stocks in an industry."""
        log.info("Fetching ticker details for industry: %s", industry_name)
        df = self._screener_filter({"Industry": industry_name, **_LIQUIDITY_FILTERS})
        result = self._normalise_details(df)
        if result.empty:
            return result
        result["Industry"] = industry_name  # normalise
        return result

    def get_all_market_details(self) -> pd.DataFrame:
        """
        Return DETAIL_COLS for ALL liquid stocks in the market, without any
        industry pre-filter. This is the universe for a full-market scan. Note:
        this can be thousands of names; finvizfinance paginates, so it may take
        a while and issue many requests.
        """
        log.info("Fetching FULL market universe from Finviz (liquidity filters only)...")
        result = self._normalise_details(self._screener_filter(dict(_LIQUIDITY_FILTERS)))
        log.info("Full market universe: %d tickers.", len(result))
        return result
