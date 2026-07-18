from src.registry import load_strategies, get_registry, get_strategy
from src.strategy import Strategy


def test_weinstein_registered():
    load_strategies("strategies")
    assert "weinstein" in get_registry()


def test_manifest_is_explicit():
    # sma_cross is registered because strategies/__init__.py imports it,
    # not because it happens to be a file on disk.
    load_strategies("strategies")
    assert "sma_cross" in get_registry()


def test_get_strategy_unknown():
    load_strategies("strategies")
    import pytest
    with pytest.raises(KeyError):
        get_strategy("nope")


def test_registered_is_strategy_subclass():
    load_strategies("strategies")
    for cls in get_registry().values():
        assert issubclass(cls, Strategy)
        assert cls.meta.key
