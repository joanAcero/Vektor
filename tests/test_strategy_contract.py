import pytest
from src.registry import load_strategies, get_strategy

def test_unknown_param_raises():
    load_strategies("strategies")
    W = get_strategy("weinstein")
    with pytest.raises(TypeError):
        W(bogus=1)

def test_defaults_applied():
    load_strategies("strategies")
    W = get_strategy("weinstein")
    s = W()
    assert s.params["sma_weeks"] == 30
    assert s.meta.signal_column == "Signal"
