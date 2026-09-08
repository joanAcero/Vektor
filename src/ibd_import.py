"""
src/ibd_import.py
-----------------
Reader for the CSV / XLSX exports produced by IBD Digital's Screen Center
(IBD 50, Sector Leaders, IPO Leaders, custom screens, ...).

WHAT THIS TAKES, AND WHAT IT DELIBERATELY THROWS AWAY
=====================================================
Two columns survive: the ticker and the IBD Composite Rating. Everything else
in the export -- Price, Price $ Change, Vol. (1000s), EPS/RS/SMR/Acc-Dis -- is
read and dropped.

Price and volume are dropped because they are a snapshot frozen at download
time. Keeping them would put a second price series in the repo next to
DataLoader's, and the point of one source of truth per concept is that there is
never a question about which one a number came from.

The individual ratings are dropped because the Composite already contains them:
IBD builds it from EPS, RS, SMR, Acc/Dis and distance from the 52-week high.
Carrying the parts next to the whole invites filtering on both.

IBD Composite is a 1-99 PERCENTILE RANK against the whole US universe. It is an
ordering, not a magnitude, and it is not comparable to any continuous indicator
in this repo -- in particular it is not Mansfield RS, which is a ratio.

THIS IS A UNIVERSE SOURCE, NOT A SIGNAL
=======================================
It answers "which tickers do I look at", never "which ones are setups".

FILE LAYOUT
===========
    data/ibd/ibd50_2026-09-05.csv
    data/ibd/sector_leaders_2026-09-05.xlsx

List name and as-of date come from the filename, so a new IBD screen is
registered by dropping a file in the directory -- no list maintained by hand.
The date matters: the export carries no date of its own, and applying today's
IBD 50 to a past date is look-ahead bias. `data/ibd/` belongs in .gitignore;
the data is licensed to your subscription.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

IBD_DATA_DIR = Path("data/ibd")
SUPPORTED_SUFFIXES = (".csv", ".txt", ".xlsx", ".xls")

#: Header slugs that mean "the ticker".
_SYMBOL_HEADERS = ("symbol", "ticker")

#: Header slugs that mean "the IBD Composite Rating", across screens.
_COMPOSITE_HEADERS = ("ibd_comp_rating", "comp_rating", "composite_rating",
                      "ibd_composite_rating", "composite")

#: Yahoo-compatible ticker shape after normalisation.
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9\-.]{0,9}$")

#: '<name>_<YYYY-MM-DD>' or '<name>_<YYYYMMDD>' filename stems.
_STEM_RE = re.compile(r"^(?P<name>.+?)[_\-](?P<date>\d{4}-\d{2}-\d{2}|\d{8})$")

#: Column name used downstream. Title-cased with a space, matching the
#: Ticker/Sector/Industry convention the runner's meta_df already uses.
COMPOSITE_COLUMN = "IBD Composite"


@dataclass(frozen=True)
class IbdSnapshot:
    """One IBD screen export, as of one date."""

    list_name: str
    as_of: date
    source_path: Path
    dated_filename: bool  # False => as_of came from mtime; not point-in-time
    frame: pd.DataFrame   # columns: Ticker, IBD Composite

    @property
    def tickers(self) -> list[str]:
        """Tickers in the order IBD ranked them."""
        return self.frame["Ticker"].tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"IbdSnapshot({self.list_name!r}, as_of={self.as_of}, "
                f"n={len(self.frame)})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(header: object) -> str:
    """'IBD Comp. Rating' -> 'ibd_comp_rating'; 'Vol. (1000s)' -> 'vol_1000s'."""
    text = str(header).strip().lower().replace("%", " pct ").replace("$", " ")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalize_ticker(raw: object) -> str | None:
    """Map an IBD ticker onto the Yahoo convention.

    IBD writes class shares with a dot (BRK.B); Yahoo wants a hyphen (BRK-B).
    A wrong ticker downloads as an empty frame and vanishes from the scan with
    no error, so the translation is explicit. Anything that does not look like
    a ticker -- footer rows, blanks, the "Data provided by..." line -- returns
    None and is dropped rather than propagated into a download.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    ticker = str(raw).strip().replace("\xa0", "").upper()
    if not ticker:
        return None
    ticker = ticker.replace(".", "-")
    if not _SYMBOL_RE.match(ticker):
        return None
    return ticker


def _parse_stem(stem: str) -> tuple[str, date | None]:
    """'ibd50_2026-09-05' -> ('ibd50', date(2026, 9, 5))."""
    match = _STEM_RE.match(stem.strip())
    if not match:
        return stem.strip().lower(), None
    raw = match.group("date")
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%d" if "-" in raw else "%Y%m%d").date()
    except ValueError:
        return stem.strip().lower(), None
    return match.group("name").strip().lower(), parsed


def _read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV/XLSX export with every cell as text."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    # sep=None sniffs comma vs tab: IBD's "Export to Excel" and "Download CSV"
    # buttons do not agree on the delimiter.
    return pd.read_csv(path, dtype=str, sep=None, engine="python")


def _pick_column(columns: list, candidates: tuple[str, ...]) -> object | None:
    slugs = {_slugify(c): c for c in columns}
    for candidate in candidates:
        if candidate in slugs:
            return slugs[candidate]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_ibd_export(path: str | Path,
                    *,
                    list_name: str | None = None,
                    as_of: date | None = None) -> IbdSnapshot:
    """Parse one IBD export down to Ticker + IBD Composite."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"IBD export not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported IBD export format {path.suffix!r}; "
                         f"expected one of {SUPPORTED_SUFFIXES}")

    stem_name, stem_date = _parse_stem(path.stem)
    dated = stem_date is not None or as_of is not None
    resolved_date = as_of or stem_date
    if resolved_date is None:
        resolved_date = datetime.fromtimestamp(path.stat().st_mtime).date()
        log.warning("IBD: %s has no date in its filename; falling back to the "
                    "file mtime (%s). Rename it to '<list>_YYYY-MM-DD%s' if you "
                    "want the snapshot to be point-in-time.",
                    path.name, resolved_date, path.suffix)

    raw = _read_table(path)
    if raw.empty:
        raise ValueError(f"IBD export {path.name} contains no rows")

    columns = list(raw.columns)
    symbol_col = _pick_column(columns, _SYMBOL_HEADERS)
    if symbol_col is None:
        raise ValueError(f"IBD export {path.name} has no Symbol column; "
                         f"columns found: {columns}")

    out = pd.DataFrame({"Ticker": raw[symbol_col].map(normalize_ticker)})

    composite_col = _pick_column(columns, _COMPOSITE_HEADERS)
    if composite_col is None:
        # Not fatal: some screens do not export it and the tickers are still
        # the point. Said once, rather than a silently empty column.
        log.warning("IBD: %s has no Composite Rating column; %s will be blank. "
                    "Columns found: %s", path.name, COMPOSITE_COLUMN, columns)
        out[COMPOSITE_COLUMN] = pd.NA
    else:
        out[COMPOSITE_COLUMN] = pd.to_numeric(
            raw[composite_col].astype("string").str.replace(r"[,\s]", "", regex=True),
            errors="coerce",
        )

    dropped = int(out["Ticker"].isna().sum())
    if dropped:
        log.info("IBD: dropped %d non-ticker row(s) from %s (footers, blanks).",
                 dropped, path.name)
    out = out[out["Ticker"].notna()].copy()
    if out.empty:
        raise ValueError(f"IBD export {path.name} yielded no usable tickers")

    duped = out["Ticker"][out["Ticker"].duplicated()].tolist()
    if duped:
        log.warning("IBD: duplicate tickers in %s: %s -- keeping the first.",
                    path.name, duped)
        out = out.drop_duplicates(subset="Ticker", keep="first")

    out["Ticker"] = out["Ticker"].astype(str)
    out = out.reset_index(drop=True)

    log.info("IBD: %s -> %d tickers (as of %s).", path.name, len(out), resolved_date)
    return IbdSnapshot(list_name=list_name or stem_name,
                       as_of=resolved_date,
                       source_path=path,
                       dated_filename=dated,
                       frame=out)


def _iter_exports(directory: Path):
    """Yield (list_name, snapshot_date, path) for every export on disk."""
    if not directory.is_dir():
        return
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        name, parsed = _parse_stem(path.stem)
        if parsed is None:
            parsed = datetime.fromtimestamp(path.stat().st_mtime).date()
        yield name, parsed, path


def available_ibd_lists(directory: str | Path = IBD_DATA_DIR) -> dict[str, list[str]]:
    """Discover the IBD screens present on disk.

    Returns {list_name: [ISO dates, newest first]}. The UI picker is built from
    this, so dropping a new export in the directory adds a source with no code
    change and no list to maintain.
    """
    found: dict[str, list[date]] = {}
    for name, snapshot_date, _ in _iter_exports(Path(directory)):
        found.setdefault(name, []).append(snapshot_date)
    return {name: [d.isoformat() for d in sorted(dates, reverse=True)]
            for name, dates in sorted(found.items())}


def load_ibd_list(list_name: str,
                  *,
                  directory: str | Path = IBD_DATA_DIR,
                  as_of: date | None = None) -> IbdSnapshot:
    """Load one IBD screen by name.

    With as_of=None the newest snapshot wins. With as_of set, the newest
    snapshot AT OR BEFORE that date wins -- the only selection rule that does
    not leak future membership into a past date.
    """
    directory = Path(directory)
    wanted = list_name.strip().lower()
    candidates = [(d, p) for name, d, p in _iter_exports(directory) if name == wanted]

    if not candidates:
        raise FileNotFoundError(
            f"No export for IBD list {wanted!r} in {directory}. Available: "
            f"{sorted(available_ibd_lists(directory))}")

    if as_of is not None:
        candidates = [c for c in candidates if c[0] <= as_of]
        if not candidates:
            raise FileNotFoundError(
                f"No {wanted!r} snapshot at or before {as_of} in {directory}")

    snapshot_date, path = max(candidates, key=lambda item: item[0])
    return read_ibd_export(path, list_name=wanted, as_of=snapshot_date)


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) != 2:
        print("usage: python -m src.ibd_import <path-to-ibd-export>")
        raise SystemExit(2)
    snap = read_ibd_export(sys.argv[1])
    print(snap, "| dated filename:", snap.dated_filename)
    print(snap.frame.to_string(index=False))
