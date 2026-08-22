"""
config.py
---------
Loads run configuration from YAML and merges CLI overrides. The source of truth
is a declarative config file; the CLI only overrides.

Strategy params declared in the file are coerced and validated against the
chosen strategy's ParamSpec schema, so a typo in the YAML fails loudly at load
time rather than silently using a default.

CONFIGURED VALUES GET NO CODE-SIDE DEFAULTS
===========================================
Where a value belongs to the config file, this module does not also carry a
fallback for it. A dataclass default plus a `.get(key, default)` is two sources
of truth for one decision, and the code copy wins silently whenever the YAML
key is missing or misspelled. `market.rotation_quadrants` is read with no
fallback and validated against the vocabulary in src/rotation.py, so a missing
or mistyped value stops the process at load time instead of producing an empty
scan universe that looks exactly like "no sector qualified".

TWO VOCABULARIES, TWO TREATMENTS
================================
`market.indices` IS validated against src/indices.py::INDICES, because that
registry is ours: a typo there is a bug we can name. `market.group` is NOT
validated against a list, because Finviz owns the sector and industry names,
they change, and there are ~150 industries -- a hardcoded copy would be a
second source of truth that goes stale silently. Where the vocabulary is ours,
check it; where it belongs to someone else, pass it through and let the query
report an empty result.

NAMING: `market` IN THE FILE, "TARGET" IN THE UI
================================================
The web UI calls this section the Target. The YAML key and the RunConfig fields
are still `market*`. Renaming would break every existing config and
daily_report.py; if wanted, it should be one deliberate commit.

REMOVED AND RENAMED KEYS ARE ERRORS, NOT NO-OPS
===============================================
`make_charts` and `chart_start` raise if present. `source: sp500` still works
but is translated, with a note in the log -- it is now one option within
`source: index`, and silently accepting it while ignoring `indices` would be
the worst of both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.registry import get_strategy

log = logging.getLogger(__name__)

# Ticker-sourcing modes understood by run.py::_scan.
#
#   industries/sectors  Finviz groups -- top-N by perf_col, or one named group
#                       when market.group is set
#   rotation            sectors selected by the RRG monitor
#   index               union of the market.indices constituent lists
#   quality             iShares MSCI USA Quality Factor ETF holdings
#   ticker              an explicit list, diagnostic mode
US_SOURCES = ("industries", "sectors", "rotation", "index", "quality", "ticker")

# Sources for which market.group means anything.
GROUPED_US_SOURCES = ("sectors", "industries")

# Old name for `source: index` with `indices: [sp500]`.
LEGACY_US_SOURCES = {"sp500": ("index", ["sp500"])}

#   indices             Wikipedia constituents of the selected markets
#   quality             iShares Edge MSCI Europe Quality Factor UCITS ETF
#                       holdings. Pan-European by construction, so
#                       market.intl_codes is not consulted for the universe --
#                       only for which benchmark the regime gate uses.
INTL_SOURCES = ("indices", "quality")

REMOVED_KEYS = {
    "make_charts": ("Charts are always drawn now; there is no switch. "
                    "Delete the key."),
    "chart_start": ("The chart window is fixed at run.py::CHART_YEARS (2 years) "
                    "and applied as a display trim after the signals are "
                    "computed on `data_start` history. Delete the key; widen "
                    "`data_start` if you want the STRATEGY to see more."),
}


def valid_quadrants() -> tuple[str, ...]:
    """The canonical RRG quadrant names, derived from src/rotation.py rather
    than restated here. Imported lazily so importing config does not pull in
    the rotation stack."""
    from src.rotation import QUADRANT_COLOR
    return tuple(QUADRANT_COLOR)


def valid_indices() -> tuple[str, ...]:
    """The index keys, derived from src/indices.py for the same reason."""
    from src.indices import INDICES
    return tuple(INDICES)


def validate_us_source(source: str) -> tuple[str, list[str] | None]:
    """Normalise `market.source`.

    Returns (source, implied_indices) — the second element is non-None only
    when a legacy alias was translated, so the caller can apply it without
    this function needing to know where indices are stored.
    """
    source = str(source).lower().strip()
    if source in LEGACY_US_SOURCES:
        new, implied = LEGACY_US_SOURCES[source]
        log.info("market.source=%r is now %r with indices=%s. Update the config "
                 "when convenient; it is translated for you meanwhile.",
                 source, new, implied)
        return new, list(implied)
    if source not in US_SOURCES:
        raise ValueError(
            f"market.source must be one of {list(US_SOURCES)}, got {source!r}.")
    return source, None


def validate_intl_source(source: str) -> str:
    source = str(source).lower().strip()
    if source not in INTL_SOURCES:
        raise ValueError(
            f"market.intl_source must be one of {list(INTL_SOURCES)}, "
            f"got {source!r}.")
    return source


def validate_indices(keys: Any, *, required: bool) -> list[str]:
    """Check index keys against src/indices.py.

    `required` is True only when us_source == "index". As with the rotation
    quadrants, an empty required value raises rather than defaulting to the
    S&P 500: picking a universe on your behalf is exactly the silent
    divergence this module exists to prevent.
    """
    items = [str(k).lower().strip() for k in (keys or []) if str(k).strip()]
    if not items:
        if required:
            raise ValueError(
                "market.source is 'index' but market.indices is missing or "
                "empty. There is deliberately no code default for it. Valid "
                f"values: {list(valid_indices())}.")
        return []
    bad = sorted(set(items) - set(valid_indices()))
    if bad:
        raise ValueError(f"Unknown index/indices {bad} in market.indices; valid "
                         f"values are {list(valid_indices())}.")
    return list(dict.fromkeys(items))


def validate_rotation_quadrants(quadrants: Any, *, required: bool) -> list[str]:
    """Check `quadrants` against the canonical set."""
    items = [str(q).lower().strip() for q in (quadrants or [])]
    if not items:
        if required:
            raise ValueError(
                "market.source is 'rotation' but market.rotation_quadrants is "
                "missing or empty in the config. There is deliberately no code "
                f"default for it. Valid values: {list(valid_quadrants())}.")
        return []
    bad = sorted(set(items) - set(valid_quadrants()))
    if bad:
        raise ValueError(
            f"Unknown quadrant(s) {bad} in market.rotation_quadrants; valid "
            f"values are {list(valid_quadrants())}.")
    return list(dict.fromkeys(items))  # dedupe, keep the order given


@dataclass
class RunConfig:
    strategy_key: str
    strategy_params: dict[str, Any]
    market_mode: str                 # "us" | "international"
    intl_codes: list[str] = field(default_factory=list)
    intl_source: str = "indices"     # one of INTL_SOURCES
    us_source: str = "industries"    # one of US_SOURCES
    us_tickers: list[str] = field(default_factory=list)  # us_source=="ticker"
    # Which index constituent lists to union, when us_source == "index".
    # Empty means nobody configured it, which validate() turns into an error
    # when it matters.
    us_indices: list[str] = field(default_factory=list)
    us_top_n_industries: int = 0
    us_perf_col: str = "Perf Week"
    # One named Finviz sector/industry instead of the top-N by perf_col. Empty
    # means "rank them" -- the original behaviour.
    us_group: str = ""
    us_rotation_quadrants: list[str] = field(default_factory=list)
    # History given to the strategy AND to the chart pass. Charts then show
    # only the last run.py::CHART_YEARS of it, so widening this affects the
    # analysis (MA warm-up, prior-decline and cluster windows), never the
    # visible window.
    data_start: str = "2020-01-01"
    output_dir: str = "results"
    strict: bool = False
    cache_max_age_hours: float = 12.0

    def build_strategy(self):
        cls = get_strategy(self.strategy_key)
        return cls(**self.strategy_params)

    def validate(self) -> "RunConfig":
        """Re-run the cross-field checks load_config() performs.

        For callers that mutate a loaded config (webapp.py). Mutates nothing
        except normalising the validated fields.
        """
        self.us_source, implied = validate_us_source(self.us_source)
        if implied and not self.us_indices:
            self.us_indices = implied

        self.intl_source = validate_intl_source(self.intl_source)

        self.us_group = str(self.us_group or "").strip()
        if self.us_group and self.us_source not in GROUPED_US_SOURCES:
            # Warn rather than raise: a stale value in flight from the browser
            # should not fail an otherwise valid run. Warn rather than clear
            # silently, because "I picked Healthcare and got the whole market"
            # is not something you should have to deduce from the results.
            log.warning("market.group=%r is ignored when market.source=%r "
                        "(it applies to %s only).",
                        self.us_group, self.us_source, list(GROUPED_US_SOURCES))
            self.us_group = ""

        self.us_indices = validate_indices(
            self.us_indices,
            required=(self.market_mode == "us" and self.us_source == "index"))
        self.us_rotation_quadrants = validate_rotation_quadrants(
            self.us_rotation_quadrants,
            required=(self.market_mode == "us" and self.us_source == "rotation"))
        return self


def coerce_params(strategy_key: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce param values using the strategy's declared schema.

    Public because webapp.py coerces browser values through it too. Two
    coercion paths for one schema would eventually disagree about what an int
    or a bool is; there is one, and it uses the schema's declared `type`.
    """
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
        # spec.type may be a real type (int/float) or a parser callable. Only
        # short-circuit on isinstance when it is actually a type.
        try:
            if isinstance(spec.type, type) and isinstance(value, spec.type):
                out[name] = value
            else:
                out[name] = spec.type(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Param {name!r}={value!r} not coercible via {spec.type}: {e}")
    return out


def _validate_against_schema(strategy_key: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Membership only, no coercion -- the flat-`params:` shim below needs to
    ask "do these belong to this strategy?" without a coercion failure being
    mistaken for the answer "no"."""
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

    for key, why in REMOVED_KEYS.items():
        if key in data:
            raise ValueError(f"`{key}:` is no longer a config option. {why}")

    strategy_key = data.get("strategy")
    if not strategy_key:
        raise ValueError("Config must specify `strategy:`.")

    # Params are strategy-specific, namespaced under each strategy key in
    # `strategy_params:`. Only the block for the selected strategy is read, so
    # switching strategies needs no other edits.
    per_strategy = data.get("strategy_params", {}) or {}
    if not isinstance(per_strategy, dict):
        raise ValueError("`strategy_params:` must be a mapping of strategy_key -> params.")

    raw_params = per_strategy.get(strategy_key, {}) or {}

    # Backward-compatible shim: a flat top-level `params:` block is honoured
    # only if it belongs to the selected strategy. If it doesn't, it's ignored
    # rather than crashing the run — usually it's leftovers for another one.
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

    params = coerce_params(strategy_key, raw_params)

    market = data.get("market", {}) or {}
    mode = str(market.get("mode", "us")).lower()
    if mode not in ("us", "international"):
        raise ValueError(f"market.mode must be 'us' or 'international', got {mode!r}")

    if "rotation_states" in market:
        raise ValueError(
            "market.rotation_states has been renamed to "
            "market.rotation_quadrants, and its values are now RRG quadrant "
            f"names {list(valid_quadrants())} rather than hunt/watch/avoid. "
            "Update config/default.yaml.")

    # validate() below re-runs the source/indices/quadrant checks, so they are
    # not duplicated here: the constructor takes the raw values and the single
    # validation pass normalises them.
    return RunConfig(
        strategy_key=strategy_key,
        strategy_params=params,
        market_mode=mode,
        intl_codes=list(market.get("intl_codes", []) or []),
        intl_source=validate_intl_source(market.get("intl_source", "indices")),
        us_source=str(market.get("source", "industries")),
        us_tickers=list(market.get("tickers", []) or []),
        us_indices=list(market.get("indices", []) or []),
        us_top_n_industries=int(market.get("top_n_industries", 0)),
        us_perf_col=market.get("perf_col", "Perf Week"),
        us_group=str(market.get("group", "") or "").strip(),
        us_rotation_quadrants=list(market.get("rotation_quadrants", []) or []),
        data_start=data.get("data_start", "2020-01-01"),
        output_dir=data.get("output_dir", "results"),
        strict=bool(data.get("strict", False)),
        cache_max_age_hours=float(data.get("cache_max_age_hours", 12.0)),
    ).validate()
