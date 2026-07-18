"""
config.py
---------
Loads run configuration from YAML and merges CLI overrides. Replaces the
original interactive input() flow, which could not be tested, scripted or
scheduled. Interactive use is still possible via the CLI, but the source of
truth is now a declarative config file.

Strategy params declared in the file are coerced and validated against the
chosen strategy's ParamSpec schema, so a typo in the YAML fails loudly at load
time rather than silently using a default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.registry import get_strategy

log = logging.getLogger(__name__)


@dataclass
class RunConfig:
    strategy_key: str
    strategy_params: dict[str, Any]
    market_mode: str                 # "us" | "international"
    intl_codes: list[str] = field(default_factory=list)
    us_source: str = "industries"   # "industries" | "sectors" | "ticker"
    us_tickers: list[str] = field(default_factory=list)  # used when us_source=="ticker"
    us_top_n_industries: int = 0
    us_perf_col: str = "Perf Week"
    data_start: str = "2020-01-01"
    chart_start: str = "2021-01-01"
    output_dir: str = "results"
    make_charts: bool = True
    strict: bool = False
    cache_max_age_hours: float = 12.0

    def build_strategy(self):
        cls = get_strategy(self.strategy_key)
        return cls(**self.strategy_params)


def _coerce_params(strategy_key: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce YAML param values using the strategy's declared schema."""
    cls = get_strategy(strategy_key)
    schema = {p.name: p for p in cls.meta.param_schema}
    unknown = set(raw) - set(schema)
    if unknown:
        raise ValueError(
            f"Config sets unknown params for {strategy_key!r}: {sorted(unknown)}. "
            f"Allowed: {sorted(schema)}"
        )
    out: dict[str, Any] = {}
    for name, value in raw.items():
        spec = schema[name]
        # spec.type may be a real type (int/float) or a parser callable
        # (e.g. a lambda for bool-from-string). Only short-circuit on
        # isinstance when it is actually a type; otherwise always call it.
        try:
            if isinstance(spec.type, type) and isinstance(value, spec.type):
                out[name] = value
            else:
                out[name] = spec.type(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Param {name!r}={value!r} not coercible via {spec.type}: {e}")
    return out


def _validate_against_schema(strategy_key: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Return raw unchanged if all keys are valid for the strategy; else raise."""
    cls = get_strategy(strategy_key)
    allowed = {p.name for p in cls.meta.param_schema}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown params for {strategy_key!r}: {sorted(unknown)}")
    return raw


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    overrides = overrides or {}
    data.update({k: v for k, v in overrides.items() if v is not None})

    strategy_key = data.get("strategy")
    if not strategy_key:
        raise ValueError("Config must specify `strategy:`.")

    # Params are strategy-specific. The config holds them namespaced under each
    # strategy key in `strategy_params:`, and we read ONLY the block for the
    # selected strategy. This lets you switch strategies (e.g. via --strategy)
    # without editing params, and a missing block falls back to the strategy's
    # declared defaults.
    per_strategy = data.get("strategy_params", {}) or {}
    if not isinstance(per_strategy, dict):
        raise ValueError("`strategy_params:` must be a mapping of strategy_key -> params.")

    raw_params = per_strategy.get(strategy_key, {}) or {}

    # Backward-compatible shim: a flat top-level `params:` block is honoured only
    # if it belongs to the selected strategy (validates cleanly against its
    # schema). If it doesn't, it's ignored rather than crashing the run — most
    # often it's leftover params for a different strategy.
    flat = data.get("params")
    if flat and not raw_params:
        try:
            raw_params = _validate_against_schema(strategy_key, flat)
        except ValueError:
            log.warning(
                "Ignoring top-level `params:` — they are not valid for strategy "
                "%r. Move per-strategy params under `strategy_params:`.",
                strategy_key,
            )
            raw_params = {}

    params = _coerce_params(strategy_key, raw_params)

    market = data.get("market", {}) or {}
    mode = market.get("mode", "us").lower()
    if mode not in ("us", "international"):
        raise ValueError(f"market.mode must be 'us' or 'international', got {mode!r}")

    return RunConfig(
        strategy_key=strategy_key,
        strategy_params=params,
        market_mode=mode,
        intl_codes=list(market.get("intl_codes", []) or []),
        us_top_n_industries=int(market.get("top_n_industries", 0)),
        us_perf_col=market.get("perf_col", "Perf Week"),
        data_start=data.get("data_start", "2020-01-01"),
        chart_start=data.get("chart_start", "2021-01-01"),
        output_dir=data.get("output_dir", "results"),
        make_charts=bool(data.get("make_charts", True)),
        strict=bool(data.get("strict", False)),
        cache_max_age_hours=float(data.get("cache_max_age_hours", 12.0)),
    )
