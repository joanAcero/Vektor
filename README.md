# VEKTOR

A **detection-only** stock-setup screener with **pluggable strategies**. VEKTOR
scans a universe of tickers, flags charts matching a strategy's criteria, writes
a CSV, and renders charts. It places no orders and gives no advice — it finds
setups for *you* to evaluate.

The flagship strategy implements Stan Weinstein's Stage-1→Stage-2 pre-breakout
base. Adding your own strategy is a single file (see below).

> **Not financial advice.** VEKTOR is a research tool. Signals are pattern
> matches on historical data, not recommendations.

## Quick start

```bash
pip install -e ".[dev]"          # or: pip install -r requirements.txt
python run.py --list             # show discovered strategies
python run.py --config config/default.yaml
```

Common overrides:

```bash
python run.py --strategy sma_cross          # pick a strategy
python run.py --no-charts                    # skip chart rendering
python run.py --strict -v                    # re-raise strategy errors, verbose
```

## How it works

```
run.py                CLI + orchestration (strategy-agnostic)
config/default.yaml   the single source of truth for a run
src/
  registry.py         explicit registry; strategies/__init__.py is the manifest
  strategy.py         the plugin contract (Strategy ABC + StrategyMeta)
  config.py           YAML -> validated RunConfig
  data_loader.py      yfinance access with a real, time-boxed cache
  screener.py         runs a strategy over tickers (errors are logged, not hidden)
  plotter.py          generic fallback chart
strategies/
  weinstein_setup.py  flagship strategy
  __init__.py         explicit manifest: imports each active strategy
  sma_cross.py        minimal example strategy
```

The orchestration layer never references strategy-specific columns. It asks each
strategy's `meta` what to display, how to sort, and which column carries the
signal. That is what makes strategies truly pluggable.

## Adding a strategy

Create `strategies/my_strategy.py`:

```python
from src.strategy import ParamSpec, Strategy, StrategyMeta
from src.registry import register

@register
class MyStrategy(Strategy):
    meta = StrategyMeta(
        key="my_strategy",
        display_name="My Strategy",
        signal_column="Signal",            # int column; a hit_value means "found"
        hit_values=(1,),
        param_schema=(
            ParamSpec("window", 20, int, "Lookback window"),
        ),
        display_columns=("MyMetric",),     # shown in results if present
        sort_by=("Market", "Ticker"),
        sort_ascending=(True, True),
    )

    def generate_signals(self, df):
        # df: Open/High/Low/Close/Volume indexed by date
        # return df + a `Signal` column (1 == setup present)
        ...
```

Then register it by adding one line to `strategies/__init__.py`:

```python
from . import my_strategy   # noqa: F401
```

and run `python run.py --strategy my_strategy`. No other file changes.

Notes:
- Discovery is **explicit**: a strategy is active only if `strategies/__init__.py`
  imports it. WIP or helper files simply aren't listed until you add them.
- Override `Strategy.plot()` for a custom chart; otherwise the generic plotter
  is used automatically.

## Data and caching

Prices come from yfinance with `auto_adjust=True` (split/dividend-adjusted),
which is correct for trend analysis. Because adjusted history is rewritten on
every corporate action, the on-disk cache is **time-boxed** (`cache_max_age_hours`)
rather than permanent.

## Testing

```bash
pytest -q
```

## License

MIT. See `LICENSE`.
