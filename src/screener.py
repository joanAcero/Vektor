import pandas as pd


class Screener:
    def __init__(self, data_loader):
        self.loader = data_loader

    def scan(self, strategy, tickers_list, market_label: str = "US"):
        """
        Scan a list of tickers with the given strategy.

        Parameters
        ----------
        tickers_list : list[str]
        market_label : str
            Human-readable label added as 'Market' column in results
            (e.g. "US", "ES", "DE").
        """
        print(f"\n🔍 [{market_label}] Escaneando con: {strategy.name}  "
              f"({len(tickers_list)} candidatos)...")
        results = []

        for ticker in tickers_list:
            df = self.loader.get_data(ticker, start_date="2020-01-01")
            if df is None or df.empty:
                continue

            try:
                data_with_signals = strategy.generate_signals(df)
            except Exception:
                continue

            if data_with_signals.empty or len(data_with_signals) < 2:
                continue

            last_row = data_with_signals.iloc[-1]

            if last_row["Signal"] == 1:
                row_result = {
                    "Market": market_label,       # ← new column
                    "Ticker": ticker,
                    "Price" : last_row["Close"],
                    "Signal_Value": last_row["Signal"],
                }

                for col in [
                    "Resistance", "Res_Zone_Bot", "Res_Zone_Top", "Res_Touches",
                    "Support",    "Sup_Zone_Bot", "Sup_Zone_Top", "Sup_Touches",
                    "Range_Width_Pct", "Distance_to_Breakout",
                    "Vol_Spike_2x", "Vol_4W_Expansion", "Vol_vs_Avg",
                ]:
                    if col in last_row.index:
                        row_result[col] = last_row[col]

                results.append(row_result)

        return pd.DataFrame(results)
