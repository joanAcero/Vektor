import numpy as np, pandas as pd, pytest
from src.registry import load_strategies, get_strategy
from src.screener import Screener

class FakeLoader:
    def __init__(self, df): self.df = df
    def get_data(self, ticker, start_date, end_date=None):
        return None if ticker == "BAD" else self.df

def _series():
    n=420; idx=pd.date_range("2021-01-01",periods=n,freq="D")
    base=100+np.where(np.sin(np.linspace(0,40,n))>0,2,-2)+np.linspace(0,6,n)
    vol=np.full(n,1e6); vol[-25:]=4e6
    return pd.DataFrame({"Open":base,"High":base+1,"Low":base-1,"Close":base,"Volume":vol},index=idx)

def test_scan_returns_agnostic_columns():
    load_strategies("strategies")
    s = get_strategy("weinstein")()
    res = Screener(FakeLoader(_series())).scan(s, ["AAA","BAD"], market_label="T")
    assert "Ticker" in res.columns and "Market" in res.columns

class Boom:
    # minimal strategy-like object that raises, to test error handling
    from src.strategy import StrategyMeta
    meta = StrategyMeta(key="boom", display_name="Boom")
    name = "Boom"
    def generate_signals(self, df): raise RuntimeError("boom")

def test_strict_reraises():
    df=_series()
    with pytest.raises(RuntimeError):
        Screener(FakeLoader(df), strict=True).scan(Boom(), ["AAA"])

def test_nonstrict_swallows_but_continues():
    df=_series()
    res = Screener(FakeLoader(df), strict=False).scan(Boom(), ["AAA","BBB"])
    assert res.empty  # no crash, just no results
