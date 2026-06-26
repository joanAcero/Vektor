"""
finviz_engine.py
-----------------
Wraps finvizfinance for US-market sector/industry data.

Public interface
----------------
get_top_sectors(top_n, col_target)          → list[str]   (legacy, kept for compat)
get_top_industries(top_n, col_target)       → list[str]
get_tickers_in_sector(sector_name)          → list[str]
get_tickers_in_industry(industry_name)      → list[str]
get_ticker_details_in_industry(industry)    → pd.DataFrame  [Ticker, Sector, Industry]
"""

import pandas as pd
from finvizfinance.group.performance import Performance
from finvizfinance.screener.overview import Overview as ScreenerOverview


class FinvizEngine:

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_pct(x):
        if isinstance(x, str):
            return float(x.strip("%")) / 100
        return x

    def _get_group_top(
        self,
        group: str,
        top_n: int,
        col_target: str,
        label: str,
    ) -> list[str]:
        """
        Generic helper that fetches a Finviz Performance group table,
        sorts by col_target and returns the top-N names.
        """
        print(f"📊 Consultando {label} en Finviz (Vista Performance)...")
        try:
            df = Performance().screener_view(group=group)

            if col_target not in df.columns:
                print(
                    f"⚠️  Columna '{col_target}' no encontrada. "
                    f"Disponibles: {df.columns.tolist()}"
                )
                fallback = next(
                    (c for c in ["Perf Month", "Perf Quarter", "Perf Year"]
                     if c in df.columns),
                    None,
                )
                if fallback is None:
                    return []
                col_target = fallback
                print(f"   ↳ Usando '{col_target}' como respaldo.")

            df[col_target] = df[col_target].apply(self._clean_pct)
            df_sorted = df.sort_values(by=col_target, ascending=False)

            print(f"\n🏆 Top {label} ({col_target}):")
            print(df_sorted[["Name", col_target]].head(top_n).to_string(index=False))

            return df_sorted["Name"].head(top_n).tolist()

        except Exception as e:
            print(f"❌ Error consultando Finviz ({label}): {e}")
            return []

    def _screener_filter(self, filters_dict: dict) -> pd.DataFrame:
        """Run a Finviz Overview screener with the given filters."""
        try:
            fs = ScreenerOverview()
            fs.set_filter(filters_dict=filters_dict)
            return fs.screener_view()
        except Exception as e:
            print(f"❌ Error en Finviz Screener: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # SECTOR API  (kept for backwards compatibility)
    # ------------------------------------------------------------------

    def get_top_sectors(self, top_n: int = 3, col_target: str = "Perf Week") -> list[str]:
        return self._get_group_top("Sector", top_n, col_target, "Sectores")

    def get_tickers_in_sector(self, sector_name: str) -> list[str]:
        print(f"   📥 Tickers para sector: {sector_name}...")
        df = self._screener_filter({
            "Sector"        : sector_name,
            "Market Cap."   : "+Small (over $300mln)",
            "Average Volume": "Over 300K",
        })
        return df["Ticker"].tolist() if not df.empty else []

    # ------------------------------------------------------------------
    # INDUSTRY API  (new)
    # ------------------------------------------------------------------

    def get_top_industries(self, top_n: int = 5, col_target: str = "Perf Week") -> list[str]:
        """Return the top-N performing Finviz industries for col_target."""
        return self._get_group_top("Industry", top_n, col_target, "Industrias")

    def get_tickers_in_industry(self, industry_name: str) -> list[str]:
        """Return tickers that belong to a specific Finviz industry."""
        print(f"   📥 Tickers para industria: {industry_name}...")
        df = self._screener_filter({
            "Industry"      : industry_name,
            "Market Cap."   : "+Small (over $300mln)",
            "Average Volume": "Over 300K",
        })
        return df["Ticker"].tolist() if not df.empty else []

    def get_ticker_details_in_industry(self, industry_name: str) -> pd.DataFrame:
        """
        Return a DataFrame with columns [Ticker, Sector, Industry] for all
        stocks in *industry_name* that pass the standard liquidity filters.

        This lets the caller enrich scan results with both Sector and
        Industry metadata in a single Finviz round-trip.
        """
        print(f"   📥 Detalles de tickers para industria: {industry_name}...")
        df = self._screener_filter({
            "Industry"      : industry_name,
            "Market Cap."   : "+Small (over $300mln)",
            "Average Volume": "Over 300K",
        })
        if df.empty:
            return pd.DataFrame(columns=["Ticker", "Sector", "Industry"])

        # The Overview screener always returns Sector and Industry columns.
        cols_needed = [c for c in ["Ticker", "Sector", "Industry"] if c in df.columns]
        result = df[cols_needed].copy()
        result["Industry"] = industry_name   # ensure consistent name
        return result
