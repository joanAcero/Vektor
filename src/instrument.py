"""
instrument.py
-------------
What the system knows about a security other than its prices: ticker, company
name, sector, industry, home market. Plus the one function that decides the
column order of a results table.

Why a type and not a dict
-------------------------
Three renderers need the SAME human label for the same security: the chart
suptitle (src/plotter.py and every strategy plot() override), the webapp chart
card caption (web/index.html) and the Telegram caption (daily_report.py). Held
as a dict, the format string is written once per renderer and drifts on the
first change. Instrument.label() is the single source of truth for that string.

Metadata is OPTIONAL by construction. Finviz supplies name/sector/industry for
US names; explicit-ticker mode and the international markets may supply neither.
Every field except `ticker` therefore defaults to empty and label() drops what
it does not have -- a chart must never print "None - UBER - ".

Display only: chart FILENAMES stay keyed on the ticker (chart_{TICKER}.png), so
nothing on disk depends on metadata that may be missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

# Market metadata, as opposed to strategy output. These columns describe the
# SECURITY, not the setup, so a strategy should not have to declare them in
# StrategyMeta.display_columns -- every new strategy would have to remember to,
# and forgetting silently loses the sector from the table and the sort menu.
# They are appended centrally by results_columns() below.
METADATA_COLUMNS = ("Name", "Sector", "Industry")


def results_columns(strategy_display_columns: Iterable[str],
                    available: Iterable[str]) -> list[str]:
    """Ordered column list for the results table, the CSV and the API payload.

    Identity, then price, then whatever the strategy declared, then any market
    metadata the strategy did not already position itself. Deduplicated and
    filtered to what the frame actually has -- so a strategy that still lists
    "Sector" in display_columns keeps its chosen position and does not get a
    second copy at the end.

    One function, three callers (CLI table, CSV, API payload), because a table
    and its payload disagreeing about columns shows up as "the sort menu offers
    a column the table doesn't have".
    """
    ordered = ["Market", "Ticker", "Name", "Price",
               *strategy_display_columns, *METADATA_COLUMNS]
    have = set(available)
    return [c for c in dict.fromkeys(ordered) if c in have]


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    market: str = ""

    # ---- construction -------------------------------------------------
    @classmethod
    def of(cls, value: "Instrument | str") -> "Instrument":
        """Coerce a bare ticker string to an Instrument.

        Renderers accept `Instrument | str` and call this on entry, so a caller
        that has no metadata -- or that has not been migrated yet -- still works
        and simply gets a ticker-only label. This is a boundary coercion, NOT a
        dual API: no code path downstream of of() ever handles a string. It also
        means src/plotter.py can be deployed before run.py is, instead of the
        two having to land in the same commit.
        """
        return value if isinstance(value, cls) else cls(ticker=str(value))

    @classmethod
    def from_row(cls, row) -> "Instrument":
        """Build from a results row (pandas Series or any mapping).

        Missing keys, None and NaN all collapse to "" so that label() has a
        single notion of "unknown" to test.
        """
        def _get(key: str) -> str:
            try:
                value = row[key]
            except (KeyError, IndexError, TypeError):
                return ""
            if value is None:
                return ""
            try:
                if pd.isna(value):
                    return ""
            except (TypeError, ValueError):  # arrays / non-scalars
                pass
            text = str(value).strip()
            return "" if text.lower() in ("nan", "none") else text

        return cls(
            ticker=_get("Ticker"),
            name=_get("Name"),
            sector=_get("Sector"),
            industry=_get("Industry"),
            market=_get("Market"),
        )

    # ---- display ------------------------------------------------------
    def label(self) -> str:
        """`Name - TICKER - Sector`, minus whatever is unknown."""
        return " \u2014 ".join(p for p in (self.name, self.ticker, self.sector) if p)

    def __str__(self) -> str:  # so f"{inst}" is never "Instrument(ticker=...)"
        return self.label() or self.ticker
