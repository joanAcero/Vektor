import yfinance as yf
import pandas as pd
import os


class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def get_data(self, ticker, start_date, end_date=None):
        """
        Descarga datos o los carga del CSV si ya existen y son recientes.
        """
        # Nombre de archivo seguro
        clean_ticker = ticker.replace("^", "")
        file_path = f"{self.data_dir}/{clean_ticker}.csv"

        # Lógica simple: Siempre intentamos descargar para tener lo último
        try:
            #print(f"Descargando {ticker}...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

            if df.empty:
                print(f"⚠No hay datos para {ticker}")
                return None

            # Limpieza básica
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.to_csv(file_path)
            return df

        except Exception as e:
            print(f"❌ Error descargando {ticker}: {e}")
            return None