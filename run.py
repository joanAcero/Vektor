"""
run.py
------
Entry point. Config-driven runner plus a thin CLI. The orchestration here is
strategy-agnostic: it asks the strategy's meta for display columns and sort
order, and delegates plotting to the strategy (falling back to the generic
plotter).

Usage:
    python run.py --config config/default.yaml
    python run.py --config config/default.yaml --strategy weinstein
    python run.py --list                      # list registered strategies

TARGET SOURCES
==============
US            industries | sectors | rotation | index | quality | ticker
International indices | quality

  * sectors / industries take cfg.us_group to scan ONE named Finviz group;
    empty means the top-N by cfg.us_perf_col.
  * index takes cfg.us_indices, a list unioned by src/indices.py.

Every source returns (tickers, meta_df), fed through _enrich(), so a new
universe is a new collector and nothing else. The one exception is the
international `indices` path, which still returns the older triple from
src/market_intl.py; it is adapted here rather than left as a second contract.

ONE BENCHMARK PER RUN
=====================
_benchmark_symbol() below is the single place that decides what every Mansfield
RS in the run is measured against -- the results column, the CSV and the chart
panel all trace back to it, so they cannot disagree.

It is deliberately per-RUN and not per-STOCK. Mansfield RS is used to RANK
names against each other (the results table sorts on it, and momentum_leaders
gates on it); numbers computed against different denominators are not
comparable, so a per-stock benchmark would silently make that ranking
meaningless. It would also degrade rather than improve accuracy today: src/
benchmarks.py maps only DE/GB/FR/IT/ES, so a Swiss or Swedish holding would
fall through to the SPY default and be measured against the wrong market
entirely. Per-market RS becomes worth doing when BENCHMARKS covers every
listing venue AND the ranking question is separated from the display question;
until then one benchmark, named on the chart, is the honest version.

COMPUTE ON FULL HISTORY, PLOT THREE YEARS
=========================================
Charts show CHART_YEARS of weekly bars, trimmed AFTER generate_signals() and
AFTER add_display_indicators(). Both orderings matter: computing signals on two
or three years would truncate the 30-week MA warm-up and weinstein_setup's
156-week windows, and computing MACD or RS after the trim would leave the first
26 (or 52) bars of the visible window blank.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import RunConfig, load_config
from src.data_loader import DataLoader
from src.indicators import add_display_indicators
from src.instrument import Instrument, METADATA_COLUMNS, results_columns
from src.plotter import plot_generic
from src.registry import load_strategies, get_registry
from src.screener import Screener

log = logging.getLogger("vektor")

# Display window for every chart. Three years of weekly bars (~156) shows a
# long base together with the decline that preceded it -- the proportionality
# Weinstein's width-vs-decline test depends on -- which two years often cut in
# half. Not a config knob: a per-run window would make two charts of the same
# setup non-comparable.
CHART_YEARS = 3

# Metadata attached after the scan. Market is included because the holdings and
# index sources know the listing country per row, which is finer than the
# single label the Screener stamps on the whole run.
ENRICH_COLUMNS = (*METADATA_COLUMNS, "Market")

# Benchmark proxy per index key, used when us_source == "index" so RS is
# measured against the index actually being scanned. Total-return ETFs, not
# price indices, for the reason src/benchmarks.py documents at length: mixing a
# total-return numerator with a price-index denominator biases every RS reading
# by roughly the dividend yield.
#
# Keys mirror src/indices.py::INDICES. Adding an index there without adding it
# here is not an error -- the run falls back to the US default and says so in
# the log -- but it is worth doing.
INDEX_BENCHMARK = {
    "sp500": "SPY",
    "nasdaq100": "QQQ",
    "dow30": "DIA",
    "sp600": "IJR",
}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _benchmark_symbol(cfg: RunConfig) -> str:
    """The one symbol every Mansfield RS in this run is measured against.

    Rules, in order:
      * International -> the benchmark for the first selected market code
        (EURO STOXX 50 for all of DE/GB/FR/IT/ES today), and the same for the
        European quality fund, which is pan-European by construction.
      * US `index` with EXACTLY ONE index selected -> that index's proxy, so a
        Nasdaq-100 scan is judged against QQQ rather than against SPY. With
        several selected the universe is a union with no single index behind
        it, so it falls back to the broad market -- the alternative would be
        picking one of them arbitrarily and labelling the chart with it.
      * Everything else (sectors, industries, rotation, quality, ticker) ->
        the US default, SPY.
    """
    from src.benchmarks import benchmark_for

    if cfg.market_mode != "us":
        codes = cfg.intl_codes or ["DE"]
        return benchmark_for(codes[0])

    if getattr(cfg, "us_source", "") == "index":
        indices = list(getattr(cfg, "us_indices", []) or [])
        if len(indices) == 1:
            symbol = INDEX_BENCHMARK.get(indices[0])
            if symbol:
                return symbol
            log.info("No benchmark proxy known for index %r; using %s. Add one "
                     "to INDEX_BENCHMARK in run.py.",
                     indices[0], benchmark_for("US"))
        elif len(indices) > 1:
            log.info("%d indices selected; relative strength is measured "
                     "against %s rather than any one of them.",
                     len(indices), benchmark_for("US"))

    return benchmark_for("US")


def _enrich(res: pd.DataFrame, meta_df: pd.DataFrame | None) -> pd.DataFrame:
    """Attach whatever metadata the source supplied, and only that.

    Defensive on purpose: a source that cannot supply Name (explicit tickers,
    Wikipedia constituents abroad) simply leaves the column out, and every
    consumer downstream degrades to a shorter label rather than failing.
    """
    if res.empty or meta_df is None or meta_df.empty or "Ticker" not in meta_df:
        return res
    idx = meta_df.drop_duplicates(subset="Ticker").set_index("Ticker")
    for col in ENRICH_COLUMNS:
        if col in idx.columns:
            mapped = res["Ticker"].map(idx[col])
            # Keep what the Screener already stamped where the map has no
            # answer -- that matters for Market, which is never blank.
            res[col] = mapped.fillna(res[col]) if col in res.columns else mapped.fillna("")
    return res


def _named_group(kind: str, name: str):
    """Collect one named Finviz sector/industry.

    Imported HERE rather than at module scope on purpose: a missing name in the
    top-level import tuple takes down every US source with an ImportError,
    which is exactly how `sp500` broke the whole US path once already.
    """
    try:
        from src.market_us import collect_us_named_group
    except ImportError:
        log.error(
            "market.group=%r needs collect_us_named_group() in src/market_us.py, "
            "which is not there. Add it, or clear the group to scan the "
            "top-ranked groups instead.", name)
        return [], pd.DataFrame()
    return collect_us_named_group(kind, name)


def _scan(cfg: RunConfig, screener: Screener, strategy,
          benchmark: pd.Series | None) -> pd.DataFrame:
    """Dispatch to the right target source, scan, enrich.

    `benchmark` is resolved once by the caller and passed in, so the series the
    Screener injects into the strategy is the identical object the chart pass
    labels and draws.
    """
    from src.benchmarks import detect_regime

    if cfg.market_mode == "us":
        try:
            from src.market_us import (
                collect_explicit_tickers,
                collect_us_by_rotation,
                collect_us_by_sector,
                collect_us_candidates,
            )
        except ImportError:
            log.exception("US target source (src/market_us.py) failed to import.")
            return pd.DataFrame()

        source = getattr(cfg, "us_source", "industries")
        group = getattr(cfg, "us_group", "")

        # Explicit-ticker mode: diagnose specific names. Skip the regime gate so
        # you always see the requested stock regardless of market conditions.
        if source == "ticker":
            if not cfg.us_tickers:
                log.error("us_source='ticker' but no tickers provided.")
                return pd.DataFrame()
            tickers, meta_df = collect_explicit_tickers(cfg.us_tickers)
        else:
            regime = detect_regime(screener.loader, "US")
            log.info("US regime: %s (%s)", regime["regime"], regime.get("reason", ""))
            if regime["regime"] == "bear":
                log.warning("US market regime is bearish — long setups skipped.")
                return pd.DataFrame()

            if source == "sectors":
                if group:
                    tickers, meta_df = _named_group("sector", group)
                else:
                    tickers, meta_df = collect_us_by_sector(
                        cfg.us_top_n_industries, cfg.us_perf_col)
            elif source == "industries":
                if group:
                    tickers, meta_df = _named_group("industry", group)
                else:
                    tickers, meta_df = collect_us_candidates(
                        cfg.us_top_n_industries, cfg.us_perf_col)
            elif source == "rotation":
                tickers, meta_df = collect_us_by_rotation(
                    screener.loader, cfg.us_rotation_quadrants)
            elif source == "index":
                from src.indices import collect_us_indices
                tickers, meta_df = collect_us_indices(cfg.us_indices)
            elif source == "quality":
                from src.holdings import collect_quality
                tickers, meta_df = collect_quality("us_quality")
            else:
                tickers, meta_df = collect_us_candidates(
                    cfg.us_top_n_industries, cfg.us_perf_col)

        if not tickers:
            return pd.DataFrame()

        res = screener.scan(strategy, tickers, market_label="US", benchmark=benchmark)
        return _enrich(res, meta_df)

    # ---- international ----------------------------------------------------
    codes = cfg.intl_codes or ["DE"]
    regime = detect_regime(screener.loader, codes[0])
    log.info("International regime (%s): %s (%s)",
             regime["benchmark"], regime["regime"], regime.get("reason", ""))
    if regime["regime"] == "bear":
        log.warning("International benchmark regime is bearish — long setups "
                    "skipped. (Short scanning not implemented yet.)")
        return pd.DataFrame()

    intl_source = getattr(cfg, "intl_source", "indices")

    if intl_source == "quality":
        from src.holdings import collect_quality
        tickers, meta_df = collect_quality("intl_quality")
    else:
        try:
            from src.market_intl import collect_intl_candidates
        except ImportError:
            log.error("International source (src/market_intl.py) not found. "
                      "Provide collect_intl_candidates(codes) -> "
                      "(tickers, market_map, sector_map).")
            return pd.DataFrame()
        tickers, market_map, sector_map = collect_intl_candidates(cfg.intl_codes)
        # Adapt the older triple to the (tickers, meta_df) contract every other
        # source uses. Wikipedia gives no company name on this path.
        meta_df = pd.DataFrame({
            "Ticker": tickers,
            "Sector": [sector_map.get(t, "") for t in tickers],
            "Market": [market_map.get(t, "?") for t in tickers],
        })

    if not tickers:
        return pd.DataFrame()

    res = screener.scan(strategy, tickers, market_label="INTL", benchmark=benchmark)
    return _enrich(res, meta_df)


def _display_window(df: pd.DataFrame, years: int = CHART_YEARS) -> pd.DataFrame:
    """Last `years` of bars, measured back from the final bar (not from today,
    so a stale feed still yields a full-looking chart rather than a short one).

    Display only -- callers must have computed signals AND display indicators
    on the full frame already.
    """
    if df is None or df.empty:
        return df
    index = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    cutoff = index.max() - pd.DateOffset(years=years)
    return df.loc[index >= cutoff]


def _chart(strategy, instrument: Instrument, signals: pd.DataFrame,
           out_path: str, signal_column: str,
           benchmark: pd.Series | None, benchmark_symbol: str) -> None:
    """Attach display indicators on the FULL frame, trim, then plot.

    The order is the point. add_display_indicators() needs the pre-trim history
    for its 26-week EMA and 52-week RS average; the strategy may plot itself,
    but either renderer receives the already-trimmed view.
    """
    enriched = add_display_indicators(signals, benchmark=benchmark,
                                      benchmark_symbol=benchmark_symbol)
    view = _display_window(enriched)
    if not strategy.plot(instrument, view, out_path):
        plot_generic(instrument, view, out_path, signal_column)


def run(cfg: RunConfig) -> pd.DataFrame:

    load_strategies("strategies")
    strategy = cfg.build_strategy()
    log.info("Strategy: %s | params: %s", strategy.name, strategy.params)

    loader = DataLoader(max_age_hours=cfg.cache_max_age_hours)
    screener = Screener(loader, start_date=cfg.data_start, strict=cfg.strict)

    # One benchmark, resolved once, used by the scan and by every chart.
    from src.benchmarks import get_weekly_close
    benchmark_symbol = _benchmark_symbol(cfg)
    benchmark = get_weekly_close(loader, benchmark_symbol)
    if benchmark is None:
        log.warning("Benchmark %s unavailable; relative strength will be blank "
                    "in the results and absent from the charts.", benchmark_symbol)
    else:
        log.info("Relative strength benchmark: %s", benchmark_symbol)

    result = _scan(cfg, screener, strategy, benchmark)

    # Ticker (diagnostic) mode: always produce charts for the requested names,
    # even if no signal fired — the whole point is to SEE the detected base and
    # understand why it did or didn't match.
    diagnostic = getattr(cfg, "us_source", "") == "ticker"
    if result.empty and not diagnostic:
        log.info("No setups matched today.")
        return result

    meta = strategy.meta
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if diagnostic:
        if benchmark is not None and hasattr(strategy, "set_benchmark"):
            strategy.set_benchmark(benchmark)
        for ticker in cfg.us_tickers:
            df = loader.get_data(ticker, start_date=cfg.data_start)
            if df is None or df.empty:
                log.warning("No data for %s.", ticker)
                continue
            signals = strategy.generate_signals(df)
            sig = int(signals[meta.signal_column].iloc[-1]) if meta.signal_column in signals else 0
            log.info("%s: Signal=%d", ticker, sig)
            chart_path = str(out_dir / f"chart_{ticker}.png")
            _chart(strategy, Instrument(ticker=ticker), signals, chart_path,
                   meta.signal_column, benchmark, benchmark_symbol)
        log.info("Diagnostic charts written to %s/", out_dir)
        if result.empty:
            return result

    display_cols = results_columns(meta.display_columns, result.columns)

    sort_by = [c for c in meta.sort_by if c in result.columns]
    if sort_by:
        asc = list(meta.sort_ascending[:len(sort_by)]) or [True] * len(sort_by)
        result = result.sort_values(by=sort_by, ascending=asc)

    # The CLI's table stays: it is this entry point's only visible output. The
    # web UI drops it in favour of the charts, but a terminal run that printed
    # nothing would be indistinguishable from a run that found nothing.
    print("\n" + result[display_cols].to_string(index=False) + "\n")

    csv_path = out_dir / f"{meta.key}_setups.csv"
    result.to_csv(csv_path, index=False)
    log.info("CSV written: %s", csv_path)

    # Sector census, so a cron/Telegram run says WHERE the setups are and not
    # only how many. Same figures the web UI shows above its chart viewer.
    if "Sector" in result.columns:
        census = result["Sector"].replace("", "Unclassified").fillna("Unclassified")
        counts = census.value_counts()
        log.info("By sector: %s",
                 ", ".join(f"{name} {n}" for name, n in counts.items()))

    for _, row in result.iterrows():
        ticker = row["Ticker"]
        df = loader.get_data(ticker, start_date=cfg.data_start)
        if df is None or df.empty:
            continue
        signals = strategy.generate_signals(df)
        chart_path = str(out_dir / f"chart_{ticker}.png")
        _chart(strategy, Instrument.from_row(row), signals, chart_path,
               meta.signal_column, benchmark, benchmark_symbol)
    log.info("Charts written to %s/", out_dir)

    log.info("Done. %d setup(s).", len(result))
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="VEKTOR — setup screener (detection only).")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--strategy", help="Override strategy key from config.")
    p.add_argument("--strict", action="store_true", default=None,
                   help="Re-raise strategy errors (debug).")
    p.add_argument("--list", action="store_true", help="List registered strategies and exit.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    _setup_logging(args.verbose)

    if args.list:
        load_strategies("strategies")
        reg = get_registry()
        if not reg:
            print("No strategies registered.")
            return 0
        print("Registered strategies:")
        for key, cls in sorted(reg.items()):
            print(f"  {key:14s} {cls.meta.display_name}")
        return 0

    # Discover before loading config: config validation resolves the strategy
    # key against the registry, so the registry must be populated first.
    load_strategies("strategies")

    overrides = {
        "strategy": args.strategy,
        "strict": args.strict,
    }
    try:
        cfg = load_config(args.config, overrides)
    except (FileNotFoundError, ValueError, KeyError) as e:
        log.error("Config error: %s", e)
        return 2

    try:
        run(cfg)
    except KeyboardInterrupt:
        log.warning("Interrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
