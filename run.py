"""
run.py
------
Entry point. Replaces the interactive main.py with a config-driven runner and a
thin CLI. The orchestration here is strategy-agnostic: it asks the strategy's
meta for display columns and sort order, and delegates plotting to the strategy
(falling back to the generic plotter).

Usage:
    python run.py --config config/default.yaml
    python run.py --config config/default.yaml --strategy weinstein --no-charts
    python run.py --list                      # list registered strategies
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import RunConfig, load_config
from src.data_loader import DataLoader
from src.plotter import plot_generic
from src.registry import load_strategies, get_registry
from src.screener import Screener

log = logging.getLogger("vektor")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _scan(cfg: RunConfig, screener: Screener, strategy) -> pd.DataFrame:
    """
    Dispatch to the right market path. The US/international ticker-sourcing
    engines are kept behind these calls; if they are not present in your tree,
    this raises a clear error rather than failing deep in a stack trace.
    """
    from src.benchmarks import benchmark_for, detect_regime, get_weekly_close

    if cfg.market_mode == "us":
        try:
            from src.market_us import (collect_us_candidates, collect_us_by_sector,
                                       collect_explicit_tickers)
        except ImportError:
            log.error("US market source (src/market_us.py) not found.")
            return pd.DataFrame()

        source = getattr(cfg, "us_source", "industries")

        # Explicit-ticker mode: diagnose specific names. Skip the regime gate so
        # you always see the requested stock regardless of market conditions.
        if source == "ticker":
            if not cfg.us_tickers:
                log.error("us_source='ticker' but no tickers provided.")
                return pd.DataFrame()
            tickers, meta_df = collect_explicit_tickers(cfg.us_tickers)
        else:
            # Market regime from the benchmark index. Long-only: skip if bearish.
            regime = detect_regime(screener.loader, "US")
            log.info("US regime: %s (%s)", regime["regime"], regime.get("reason", ""))
            if regime["regime"] == "bear":
                log.warning("US market regime is bearish — long setups skipped.")
                return pd.DataFrame()

            if source == "sectors":
                tickers, meta_df = collect_us_by_sector(cfg.us_top_n_industries, cfg.us_perf_col)
            else:  # "industries" (or full market when top_n<=0)
                tickers, meta_df = collect_us_candidates(cfg.us_top_n_industries, cfg.us_perf_col)

        if not tickers:
            return pd.DataFrame()

        bench = get_weekly_close(screener.loader, benchmark_for("US"))
        res = screener.scan(strategy, tickers, market_label="US", benchmark=bench)
        # Enrich defensively: only map columns Finviz actually returned.
        if not res.empty and meta_df is not None and not meta_df.empty:
            idx = meta_df.set_index("Ticker")
            for col in ("Sector", "Industry"):
                if col in idx.columns:
                    res[col] = res["Ticker"].map(idx[col]).fillna("")
        return res

    # international
    try:
        from src.market_intl import collect_intl_candidates
    except ImportError:
        log.error("International source (src/market_intl.py) not found. "
                  "Provide collect_intl_candidates(codes) -> (tickers, market_map, sector_map).")
        return pd.DataFrame()

    # European markets here all benchmark to EURO STOXX 50; use the first
    # selected market's benchmark (they coincide for DE/GB/FR/IT/ES).
    codes = cfg.intl_codes or ["DE"]
    regime = detect_regime(screener.loader, codes[0])
    log.info("International regime (%s): %s (%s)",
             regime["benchmark"], regime["regime"], regime.get("reason", ""))
    if regime["regime"] == "bear":
        log.warning("International benchmark regime is bearish — long setups "
                    "skipped. (Short scanning not implemented yet.)")
        return pd.DataFrame()

    tickers, market_map, sector_map = collect_intl_candidates(cfg.intl_codes)
    if not tickers:
        return pd.DataFrame()

    bench = get_weekly_close(screener.loader, benchmark_for(codes[0]))
    res = screener.scan(strategy, tickers, market_label="INTL", benchmark=bench)
    if not res.empty:
        res["Market"] = res["Ticker"].map(market_map).fillna("?")
        res["Sector"] = res["Ticker"].map(sector_map).fillna("")
    return res


def run(cfg: RunConfig) -> pd.DataFrame:
    load_strategies("strategies")
    strategy = cfg.build_strategy()
    log.info("Strategy: %s | params: %s", strategy.name, strategy.params)

    loader = DataLoader(max_age_hours=cfg.cache_max_age_hours)
    screener = Screener(loader, start_date=cfg.data_start, strict=cfg.strict)

    result = _scan(cfg, screener, strategy)

    # Ticker (diagnostic) mode: always produce charts for the requested names,
    # even if no signal fired — the whole point is to SEE the detected base and
    # understand why it did or didn't match.
    diagnostic = getattr(cfg, "us_source", "") == "ticker"
    if result.empty and not diagnostic:
        log.info("No setups matched today.")
        return result

    meta = strategy.meta
    if diagnostic and cfg.make_charts:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        bench = None
        try:
            from src.benchmarks import benchmark_for, get_weekly_close
            bench = get_weekly_close(loader, benchmark_for("US"))
        except Exception:  # noqa: BLE001
            pass
        if bench is not None and hasattr(strategy, "set_benchmark"):
            strategy.set_benchmark(bench)
        for ticker in cfg.us_tickers:
            df = loader.get_data(ticker, start_date=cfg.chart_start)
            if df is None or df.empty:
                log.warning("No data for %s.", ticker)
                continue
            signals = strategy.generate_signals(df)
            sig = int(signals[meta.signal_column].iloc[-1]) if meta.signal_column in signals else 0
            log.info("%s: Signal=%d", ticker, sig)
            chart_path = str(out_dir / f"chart_{ticker}.png")
            if not strategy.plot(ticker, signals, chart_path):
                plot_generic(ticker, signals, chart_path, meta.signal_column)
        log.info("Diagnostic charts written to %s/", out_dir)
        # Fall through: if some tickers matched, also show them in the table.
        if result.empty:
            return result

    display_cols = ["Market", "Ticker", "Price", *meta.display_columns]
    display_cols = [c for c in dict.fromkeys(display_cols) if c in result.columns]

    sort_by = [c for c in meta.sort_by if c in result.columns]
    if sort_by:
        asc = list(meta.sort_ascending[:len(sort_by)]) or [True] * len(sort_by)
        result = result.sort_values(by=sort_by, ascending=asc)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + result[display_cols].to_string(index=False) + "\n")

    csv_path = out_dir / f"{meta.key}_setups.csv"
    result.to_csv(csv_path, index=False)
    log.info("CSV written: %s", csv_path)

    if cfg.make_charts:
        for _, row in result.iterrows():
            ticker = row["Ticker"]
            df = loader.get_data(ticker, start_date=cfg.chart_start)
            if df is None or df.empty:
                continue
            signals = strategy.generate_signals(df)
            chart_path = str(out_dir / f"chart_{ticker}.png")
            # Strategy may plot itself; otherwise fall back to the generic chart.
            if not strategy.plot(ticker, signals, chart_path):
                plot_generic(ticker, signals, chart_path, meta.signal_column)
        log.info("Charts written to %s/", out_dir)

    log.info("Done. %d setup(s).", len(result))
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="VEKTOR — setup screener (detection only).")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--strategy", help="Override strategy key from config.")
    p.add_argument("--no-charts", dest="make_charts", action="store_false", default=None)
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
        "make_charts": args.make_charts,
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
