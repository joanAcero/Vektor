"""
holdings.py
-----------
ETF holdings as a scan universe. Currently the two MSCI Quality Factor funds,
but the module is fund-agnostic: adding another iShares product is one entry in
FUNDS below, not new code.

Public interface (consumed by run.py):
    collect_quality(fund_key) -> (tickers: list[str], meta_df: DataFrame)

meta_df carries Ticker, Name, Sector, Industry, Market. Industry is always
empty -- iShares publishes GICS sector but not industry -- and every other
field comes straight from the fund's own file, so Quality mode is the only
source in the system that gives international names a company name AND a
sector. Both were previously blank outside the US.

Self-check without running a scan:
    python -m src.holdings intl_quality

WHY A FUND FILE AND NOT A SCRAPE
================================
This is the issuer's own daily holdings file: authoritative, dated, and stable
in format. It is a better universe than a Wikipedia table and much better than
a screener, because membership already encodes the quality screen (high return
on equity, stable earnings, low debt) that you would otherwise have to
reconstruct from fundamentals.

Two things that follow from that, and are NOT bugs:
  * The universe is ~125 names, not thousands. A scan finding three setups in
    it is not a broken scan.
  * Membership changes only at the semi-annual rebalance, so a cached file a
    few days old is as good as a fresh one. The cache below reflects that.

THE ENDPOINTS
=============
US    the product page's own `latest-holdings.csv` link. Clean and dateless.
EMEA  the BlackRock document API, which is what the site's download button
      actually calls:

        /varnish-api/uk-retail01-product-data/product-data/api/v1/
            get-fund-document?...&portfolioId=272024&asOfDate=YYYYMMDD
            &component=holdings

      `asOfDate` is a REAL constraint, not decoration: it must name a date for
      which holdings were published. Today's date fails on weekends, on
      holidays, and before the day's file is posted -- the working example was
      dated two days back. So the date is not computed, it is SEARCHED: the
      dateless form is tried first (in case the API defaults to latest), then
      today, then each preceding day up to AS_OF_LOOKBACK_DAYS. First response
      that parses wins and is cached for a day, so the walk-back costs a
      handful of requests once, not per scan.

Each fund carries a tuple of URL TEMPLATES tried in order. A template
containing {as_of} is expanded across the date walk-back; one without it is
requested as-is. If everything fails, the error names the product page and
tells you where to paste a working URL.

TICKER MAPPING -- WHAT THE REAL FILE TAUGHT US
==============================================
Two conventions in the European file that a plausible-looking guess gets wrong:

  * Share classes are written with a SPACE: `NOVO B`, `ATCO A`, `TEL2 B`,
    `COLO B`, `INDU C`. Yahoo writes them with a hyphen: NOVO-B.CO,
    ATCO-A.ST. An earlier version of this module rejected any ticker
    containing a space outright, which would have dropped every Nordic share
    class from the universe without a word in the log.
  * Some LSE tickers carry a TRAILING DOT: `RR.` (Rolls-Royce), `NG.`
    (National Grid). Appending the suffix naively yields `RR..L`, which is not
    a symbol; the dot is a placeholder and has to go first.

Both are handled in _yahoo_symbol(). The remaining risk is that an issuer
ticker legitimately differs from Yahoo's - there is no rule that can fix that,
so TICKER_OVERRIDES exists. Note the failure is SILENT: an unresolvable symbol
returns no price data and the Screener skips it. If a holding you expect never
appears in results, run the self-check above and look for it in the output.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path("cache/holdings")
# Semi-annual rebalance, so a day-old file is not stale in any way that
# matters. This TTL is about not hammering the issuer on every scan, not about
# freshness of the data.
CACHE_MAX_AGE_HOURS = 24.0
REQUEST_TIMEOUT = 30
# How far back the asOfDate search goes. Eight days covers a weekend plus a
# public holiday plus the publication lag with room to spare.
AS_OF_LOOKBACK_DAYS = 8

# A default python-requests UA gets refused by some BlackRock edge nodes.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; VektorScreener/0.1)"}

# Columns every collector in this project returns. Mirrors DETAIL_COLS in
# src/finviz_engine.py plus Market, which the international funds need and the
# Finviz path gets from the runner instead.
HOLDING_COLS = ("Ticker", "Name", "Sector", "Industry", "Market")


@dataclass(frozen=True)
class FundSpec:
    key: str
    etf_ticker: str
    name: str                       # exactly as the issuer writes it; shown in the UI
    region: str                     # "us" | "emea" -- decides ticker normalisation
    product_url: str
    csv_urls: tuple[str, ...]       # templates, tried in order; {as_of} = YYYYMMDD


_BLACKROCK_API = ("https://www.blackrock.com/varnish-api/{site}-product-data"
                  "/product-data/api/v1/get-fund-document"
                  "?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite={target}"
                  "&locale={locale}&portfolioId={pid}&userType=individual"
                  "&component=holdings")

FUNDS: dict[str, FundSpec] = {
    "us_quality": FundSpec(
        key="us_quality",
        etf_ticker="QUAL",
        name="iShares MSCI USA Quality Factor ETF",
        region="us",
        product_url=("https://www.ishares.com/us/products/256101/"
                     "ishares-msci-usa-quality-factor-etf"),
        csv_urls=(
            # Taken verbatim from the "Download Holdings CSV" link on the US
            # product page. Dateless, so nothing to search.
            "https://www.ishares.com/us/products/256101/"
            "ishares-msci-usa-quality-factor-etf/latest-holdings.csv",
            # Same document API as EMEA, by analogy. Untested; harmless as a
            # fallback because the line above works.
            _BLACKROCK_API.format(site="blk-one01", target="us-ishares",
                                  locale="en_US", pid="256101"),
            _BLACKROCK_API.format(site="blk-one01", target="us-ishares",
                                  locale="en_US", pid="256101") + "&asOfDate={as_of}",
        ),
    ),
    "intl_quality": FundSpec(
        key="intl_quality",
        etf_ticker="IEFQ",
        name="iShares Edge MSCI Europe Quality Factor UCITS ETF",
        region="emea",
        product_url=("https://www.ishares.com/uk/individual/en/products/272024/"
                     "ishares-msci-europe-quality-factor-ucits-etf"),
        csv_urls=(
            # Confirmed working (the site's own download call, minus the date).
            _BLACKROCK_API.format(site="uk-retail01", target="ishares-uk",
                                  locale="en_GB", pid="272024"),
            _BLACKROCK_API.format(site="uk-retail01", target="ishares-uk",
                                  locale="en_GB", pid="272024") + "&asOfDate={as_of}",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Ticker normalisation
# ---------------------------------------------------------------------------
# Exchange first, because the exchange IS the venue and the Yahoo suffix
# encodes the venue. Location (country of risk) is only a fallback: for a
# European fund it can name the country of incorporation rather than the
# listing -- Rio Tinto and Coca-Cola HBC are the usual examples -- which would
# send us to the wrong listing.
#
# Matched as case-insensitive substrings. Keyed on the CITY, not on the
# operator: the operator names collide across continents and the city names do
# not. An earlier version keyed on operator and carried generic ("nasdaq", "")
# and ("nyse", "") catch-alls for US venues -- which matched "Nasdaq Stockholm"
# and mapped every Swedish holding to a US listing. BOL failed loudly because
# no US security is called BOL; EQT did NOT fail, it silently resolved to EQT
# Corporation of Pennsylvania and got scanned as if it were EQT AB. The
# catch-alls are gone: region == "us" already yields an empty suffix in
# _yahoo_symbol(), so this table never needed to describe US venues at all.
#
# Operator-only strings ("Nasdaq Omx Nordic", plain "Euronext") are absent on
# purpose -- each spans several countries, so they must fall through to
# Location rather than guess.
_SUFFIX_BY_EXCHANGE: tuple[tuple[str, str], ...] = (
    ("stockholm", ".ST"),
    ("copenhagen", ".CO"),
    ("helsinki", ".HE"),
    ("oslo", ".OL"),
    ("amsterdam", ".AS"),
    ("brussels", ".BR"),
    ("lisbon", ".LS"),
    ("dublin", ".IR"),
    ("irish", ".IR"),
    ("paris", ".PA"),
    ("london", ".L"),
    ("swiss", ".SW"),
    ("borsa italiana", ".MI"),
    ("milan", ".MI"),
    ("madrid", ".MC"),
    ("xetra", ".DE"),
    ("deutsche boerse", ".DE"),
    ("frankfurt", ".DE"),
    ("wiener", ".VI"),
    ("vienna", ".VI"),
)

_SUFFIX_BY_LOCATION: dict[str, str] = {
    "united states": "",
    "germany": ".DE",
    "france": ".PA",
    "netherlands": ".AS",
    "belgium": ".BR",
    "portugal": ".LS",
    "spain": ".MC",
    "italy": ".MI",
    "united kingdom": ".L",
    "switzerland": ".SW",
    "sweden": ".ST",
    "denmark": ".CO",
    "norway": ".OL",
    "finland": ".HE",
    "austria": ".VI",
    "ireland": ".IR",
}

# Market label for the results table / chart title, derived from the resolved
# suffix so it can never disagree with the symbol we actually request.
_MARKET_BY_SUFFIX: dict[str, str] = {
    "": "US", ".DE": "DE", ".PA": "FR", ".AS": "NL", ".BR": "BE",
    ".LS": "PT", ".MC": "ES", ".MI": "IT", ".L": "GB", ".SW": "CH",
    ".ST": "SE", ".CO": "DK", ".OL": "NO", ".HE": "FI", ".VI": "AT",
    ".IR": "IE",
}

# Manual escape hatch for anything the rules get wrong. Keyed on the symbol
# this module builds, valued with the Yahoo symbol it should have been. Empty
# by design: add entries only from observed failures, never from speculation.
TICKER_OVERRIDES: dict[str, str] = {}

# Currency codes appear as cash rows. Anything else non-equity is caught by the
# Asset Class / Sector filters in _parse, which is where futures and FX
# forwards (BZFUT, VHU6, contract codes that change monthly) get dropped --
# blacklisting those by name would need editing every quarter.
_CURRENCY_TICKERS = {"USD", "EUR", "GBP", "CHF", "SEK", "DKK", "NOK", "JPY",
                     "CAD", "AUD", "PLN", "CZK", "HUF", "ILS", "TRY"}
_PLACEHOLDER_TICKERS = {"-", "", "CASH", "MARGIN", "XTSLA"}


def _suffix_for(exchange: str, location: str) -> str | None:
    """Yahoo suffix for a listing, or None if we cannot tell."""
    ex = str(exchange).strip().lower()
    for pattern, suffix in _SUFFIX_BY_EXCHANGE:
        if pattern in ex:
            return suffix
    loc = str(location).strip().lower()
    if loc in _SUFFIX_BY_LOCATION:
        log.debug("Exchange %r unrecognised; falling back to location %r.",
                  exchange, location)
        return _SUFFIX_BY_LOCATION[loc]
    return None


def _yahoo_symbol(raw_ticker: str, exchange: str, location: str,
                  region: str) -> tuple[str, str] | None:
    """(symbol, market) for a holding, or None if it is not a mappable equity.

    See the module docstring for the two conventions this has to undo.
    """
    base = str(raw_ticker).strip().upper()
    if base in _PLACEHOLDER_TICKERS or base in _CURRENCY_TICKERS:
        return None

    # `RR.` -> `RR`  (trailing dot is a placeholder, not part of the symbol)
    base = base.rstrip(".")
    # `NOVO B` -> `NOVO-B`, and any interior dot to the same separator.
    base = base.replace(" ", "-").replace(".", "-")
    if not base:
        return None

    if region == "us":
        suffix = ""
    else:
        suffix = _suffix_for(exchange, location)
        # STRUCTURAL GUARD. A European fund holds European listings, so an
        # empty suffix -- a bare symbol, i.e. a US listing -- cannot be right;
        # it means a rule matched something it should not have. Refusing here
        # converts that into a visible skip instead of a scan of the wrong
        # company: the bug this catches produced a complete, plausible chart
        # of EQT Corporation while the fund held EQT AB. A skipped name costs
        # you one line in the log; a wrong one costs you a decision.
        if suffix == "":
            log.warning("%s: exchange=%r location=%r resolved to a US listing in a "
                        "non-US fund. Refusing; fix the mapping in src/holdings.py.",
                        base, exchange, location)
            return None
        if suffix is None:
            log.debug("No suffix for exchange=%r location=%r (ticker %s); skipping.",
                      exchange, location, base)
            return None

    symbol = TICKER_OVERRIDES.get(f"{base}{suffix}", f"{base}{suffix}")
    return symbol, _MARKET_BY_SUFFIX.get(suffix, "?")


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------
def _cache_path(fund: FundSpec) -> Path:
    return CACHE_DIR / f"{fund.key}.csv"


def _candidate_urls(fund: FundSpec):
    """Expand the templates. Dateless ones yield once; {as_of} ones yield the
    date walk-back, newest first."""
    today = date.today()
    for template in fund.csv_urls:
        if "{as_of}" not in template:
            yield template
            continue
        for back in range(AS_OF_LOOKBACK_DAYS + 1):
            day = today - timedelta(days=back)
            yield template.format(as_of=day.strftime("%Y%m%d"))


def _decode(resp: requests.Response) -> str:
    """Bytes -> text without mangling accented company names.

    utf-8-sig first (handles the BOM these files often carry), cp1252 second,
    which is what BlackRock's older EMEA exports use. Guessing wrong here turns
    Nestlé into Nestlé rather than failing, so it is worth being explicit.
    """
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return resp.content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return resp.content.decode("utf-8", errors="replace")


def _download(fund: FundSpec) -> str | None:
    """First candidate URL that returns something holdings-shaped."""
    tried = 0
    for url in _candidate_urls(fund):
        tried += 1
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:  # noqa: PERF203
            log.debug("%s: %s -> %s", fund.key, url, e)
            continue
        text = _decode(resp)
        if _header_row(text) is None:
            # A 200 that is actually an HTML error/consent page, or a date with
            # no published file. Cheap check, and it is the failure mode that
            # would otherwise reach pandas as an unreadable parse error.
            log.debug("%s: %s returned 200 with no holdings table.", fund.key, url)
            continue
        log.info("%s: holdings fetched after %d attempt(s) from %s",
                 fund.key, tried, url)
        return text
    log.warning("%s: all %d candidate URL(s) failed.", fund.key, tried)
    return None


def _header_row(text: str) -> int | None:
    """Index of the column-header line. iShares files open with a block of
    fund metadata (name, inception, NAV) before the table starts."""
    for i, line in enumerate(text.splitlines()[:60]):
        fields = {f.strip().strip('"') for f in line.split(",")[:8]}
        if "Ticker" in fields and "Name" in fields:
            return i
    return None


def _parse(text: str, fund: FundSpec) -> pd.DataFrame:
    """Raw CSV text -> HOLDING_COLS frame. Raises ValueError if unparseable."""
    start = _header_row(text)
    if start is None:
        raise ValueError("no header row with Ticker + Name found in the file")

    df = pd.read_csv(StringIO(text), skiprows=start, thousands=",",
                     on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    # Drop everything that is not a share. Two independent filters because the
    # files are not consistent about which column carries the flag, and both
    # are cheap.
    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.strip().str.casefold() == "equity"]
    if "Sector" in df.columns:
        df = df[~df["Sector"].astype(str).str.contains("cash|derivative", case=False,
                                                       na=False)]

    exchange_col = "Exchange" if "Exchange" in df.columns else None
    location_col = "Location" if "Location" in df.columns else None

    rows, skipped = [], []
    for _, row in df.iterrows():
        resolved = _yahoo_symbol(
            row.get("Ticker", ""),
            row[exchange_col] if exchange_col else "",
            row[location_col] if location_col else "",
            fund.region,
        )
        if resolved is None:
            skipped.append(str(row.get("Ticker", "")).strip())
            continue
        symbol, market = resolved
        rows.append({
            "Ticker": symbol,
            "Name": str(row.get("Name", "")).strip(),
            "Sector": str(row.get("Sector", "")).strip(),
            "Industry": "",   # iShares publishes sector only
            "Market": market,
        })

    out = pd.DataFrame(rows, columns=list(HOLDING_COLS))
    out = out.drop_duplicates(subset="Ticker").reset_index(drop=True)
    if skipped:
        # Expected to be a handful of cash and FX lines. A long list means the
        # exchange/location mapping has a hole worth filling -- which is why
        # the tickers are NAMED rather than counted, and why this is a warning
        # rather than info. A universe quietly missing ten names looks exactly
        # like a universe that legitimately has ten fewer.
        log.warning("%s: %d row(s) not mapped to a symbol: %s",
                    fund.key, len(skipped), ", ".join(skipped[:25]))
    if out.empty:
        raise ValueError("parsed the file but no tradable equity rows survived")
    return out


def fetch_holdings(fund_key: str, *, force: bool = False) -> pd.DataFrame:
    """Holdings for `fund_key`, from cache when fresh, else downloaded.

    A failed download falls back to the cached copy REGARDLESS of age, with a
    warning: for a semi-annually rebalanced fund, last week's constituents are
    a far better answer than no scan at all. Only a failure with no cache at
    all raises.
    """
    fund = FUNDS.get(fund_key)
    if fund is None:
        raise KeyError(f"Unknown fund {fund_key!r}. Known: {sorted(FUNDS)}")

    cache = _cache_path(fund)
    fresh = (cache.exists()
             and (time.time() - cache.stat().st_mtime) < CACHE_MAX_AGE_HOURS * 3600)

    if fresh and not force:
        try:
            return _parse(cache.read_text(encoding="utf-8"), fund)
        except (OSError, ValueError) as e:
            log.warning("%s: cached file unusable (%s); refetching.", fund.key, e)

    text = _download(fund)
    if text is not None:
        try:
            parsed = _parse(text, fund)
        except ValueError as e:
            log.error("%s: downloaded file did not parse: %s", fund.key, e)
        else:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
            return parsed

    if cache.exists():
        log.warning("%s: using the CACHED holdings file — the download failed. "
                    "Constituents may be out of date.", fund.key)
        return _parse(cache.read_text(encoding="utf-8"), fund)

    raise RuntimeError(
        f"Could not obtain holdings for {fund.name} ({fund.etf_ticker}) and no "
        f"cached copy exists.\nTemplates tried:\n  "
        + "\n  ".join(fund.csv_urls)
        + f"\n({AS_OF_LOOKBACK_DAYS + 1} dates attempted for any template with "
          f"an asOfDate.)\nOpen {fund.product_url}, use the holdings download, "
          f"copy the URL the browser actually requests, and add it to "
          f"FUNDS['{fund.key}'].csv_urls in src/holdings.py."
    )


def collect_quality(fund_key: str) -> tuple[list[str], pd.DataFrame]:
    """(tickers, meta_df) for a quality fund — the collector contract run.py
    expects from every source."""
    fund = FUNDS[fund_key]
    meta_df = fetch_holdings(fund_key)
    tickers = meta_df["Ticker"].tolist()
    log.info("%s (%s): %d holding(s) to scan.",
             fund.name, fund.etf_ticker, len(tickers))
    return tickers, meta_df


def fund_label(fund_key: str) -> str:
    """`Quality (<ETF name> holdings)` — the string the UI shows."""
    return f"Quality ({FUNDS[fund_key].name} holdings)"


if __name__ == "__main__":
    # Self-check: fetch and print, without touching the scanner. Use this after
    # changing a URL or a mapping rule.
    #   python -m src.holdings intl_quality
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)-7s %(name)s: %(message)s")
    key = sys.argv[1] if len(sys.argv) > 1 else "intl_quality"
    frame = fetch_holdings(key, force="--force" in sys.argv)
    print(f"\n{FUNDS[key].name} — {len(frame)} holdings\n")
    print(frame.to_string(index=False))
    # The market census is the cheapest test for a mis-resolved suffix: a
    # European fund reporting any US rows, or a country count that does not
    # match the fund's geography, means a mapping rule fired wrongly.
    print("\nBy market:")
    for market, n in frame["Market"].value_counts().items():
        print(f"  {market:>3}  {n}")
