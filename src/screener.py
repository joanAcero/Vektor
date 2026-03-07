import pandas as pd


class Screener:
    def __init__(self, data_loader):
        self.loader = data_loader

    def scan(self, strategy, tickers_list):
        print(f"\n🔍 Escaneando mercado con: {strategy.name}...")
        results = []

        for ticker in tickers_list:
            df = self.loader.get_data(ticker, start_date="2020-01-01")

            if df is None or df.empty: continue

            try:
                data_with_signals = strategy.generate_signals(df)
            except Exception:
                continue

            if data_with_signals.empty or len(data_with_signals) < 2:
                continue

            last_row = data_with_signals.iloc[-1]

            if last_row['Signal'] == 1:
                row_result = {
                    "Ticker": ticker,
                    "Price": last_row['Close'],
                    "Signal_Value": last_row['Signal']
                }

                # --- AÑADIMOS LAS NUEVAS COLUMNAS AQUÍ ---
                cols_of_interest = [
                    'Major_High',
                    'Distance_to_Breakout',
                    'Touch_Count',
                    'Vol_Spike_2x',  # <--- NUEVO
                    'Vol_4W_Expansion',  # <--- NUEVO
                    'Vol_vs_Avg'
                ]

                for col in cols_of_interest:
                    if col in last_row.index:
                        row_result[col] = last_row[col]

                results.append(row_result)

        return pd.DataFrame(results)