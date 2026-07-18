"""
strategy.py
-----------
The plugin contract. Everything the rest of the system needs to know about a
strategy is declared *by the strategy itself*, so that Screener, the runner and
the plotter never reference strategy-specific column names.

A strategy author implements one class:

    @register
    class MyStrategy(Strategy):
        meta = StrategyMeta(
            key="my_strategy",
            display_name="My Strategy",
            param_schema={...},
            display_columns=[...],
        )
        def generate_signals(self, df): ...

and drops the file in strategies/. That is the entire integration surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter, used to build CLI / YAML and validate input.

    choices: if set, the parameter is an enumeration — the frontend renders a
    dropdown of these options instead of a free-text/number field.
    """
    name: str
    default: Any
    type: Callable[[str], Any]
    help: str = ""
    choices: tuple = ()


@dataclass(frozen=True)
class StrategyMeta:
    """
    Self-description of a strategy.

    key            : stable machine identifier (used in config & CLI).
    display_name   : human-readable name.
    signal_column  : name of the integer column that flags a hit.
    hit_values     : which values in signal_column count as a "setup found".
    param_schema   : tunable parameters, surfaced to YAML/CLI.
    display_columns : ordered columns to show in the results table *if present*.
    sort_by / sort_ascending : default ordering of the results table.
    """
    key: str
    display_name: str
    description: str = ""
    signal_column: str = "Signal"
    hit_values: tuple[int, ...] = (1,)
    param_schema: tuple[ParamSpec, ...] = field(default_factory=tuple)
    display_columns: tuple[str, ...] = field(default_factory=tuple)
    sort_by: tuple[str, ...] = field(default_factory=tuple)
    sort_ascending: tuple[bool, ...] = field(default_factory=tuple)


class Strategy(ABC):
    """
    Base class for all strategies.

    Subclasses MUST set a class-level ``meta: StrategyMeta`` and implement
    ``generate_signals``. They MAY override ``plot`` to provide a custom chart;
    if they don't, the runner falls back to the generic plotter.
    """

    meta: StrategyMeta  # required on every concrete subclass

    def __init__(self, **params: Any):
        # Validate + store params against the declared schema.
        schema = {p.name: p for p in self.meta.param_schema}
        unknown = set(params) - set(schema)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} received unknown params {sorted(unknown)}; "
                f"allowed: {sorted(schema)}"
            )
        self.params: dict[str, Any] = {
            name: params.get(name, spec.default) for name, spec in schema.items()
        }

    def __getattr__(self, item: str) -> Any:
        # Convenience: expose validated params as attributes (self.sma_weeks),
        # but only for names actually in the schema, so typos still raise.
        params = self.__dict__.get("params", {})
        if item in params:
            return params[item]
        raise AttributeError(item)

    @property
    def name(self) -> str:
        return self.meta.display_name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        INPUT : DataFrame indexed by date with columns Open/High/Low/Close/Volume.
        OUTPUT: the same data plus, at minimum, ``self.meta.signal_column``
                (int: a value in ``meta.hit_values`` means "setup present").
        Strategies are detection-only; they never place orders.
        """
        raise NotImplementedError

    def plot(self, ticker: str, signals_df: pd.DataFrame, out_path: str) -> bool:
        """
        Optional custom chart. Return True if a chart was written, False to let
        the runner use the generic fallback plotter. Default: no custom plot.
        """
        return False
