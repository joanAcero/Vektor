from src.strategy import Strategy
import pandas as pd
import numpy as np


class WeinsteinSetup(Strategy):
    def __init__(self, sma_weeks=30, lookback_weeks=30, proximity_pct=0.05, min_touches=2, touch_tolerance=0.02):
        super().__init__(f"Weinstein Setup (Near Highs < {proximity_pct * 100}%)")
        self.sma_period = sma_weeks
        self.lookback = lookback_weeks
        self.proximity = proximity_pct
        self.min_touches = min_touches
        self.tolerance = touch_tolerance

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Resample a Semanal
        logic = {
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        w_df = df.resample('W-FRI').agg(logic).dropna()

        # --- CÁLCULOS BÁSICOS ---
        w_df['Major_High'] = w_df['High'].rolling(window=self.lookback).max()
        w_df['SMA_30W'] = w_df['Close'].rolling(window=self.sma_period).mean()

        # --- LÓGICA DE TOQUES  ---
        def count_touches_in_window(x):
            window_max = np.max(x)
            touch_limit = window_max * (1 - self.tolerance)
            return np.sum(x >= touch_limit)

        w_df['Touch_Count'] = w_df['High'].rolling(window=self.lookback).apply(count_touches_in_window, raw=True)

        # ==============================================================================
        # NUEVOS FILTROS DE VOLUMEN (WEINSTEIN RULES)
        # ==============================================================================

        # 1. PICO DE VOLUMEN (Semana actual vs Mes anterior)
        # Calculamos el promedio de las 4 semanas PREVIAS (shift 1)
        w_df['Vol_Avg_Prev_4W'] = w_df['Volume'].shift(1).rolling(window=4).mean()

        # Condición: Volumen actual >= 2 * Promedio anterior
        w_df['Vol_Spike_2x'] = np.where(w_df['Volume'] >= (2 * w_df['Vol_Avg_Prev_4W']), 1, 0)

        # 2. EXPANSIÓN DE VOLUMEN (Últimas 4 semanas vs Base anterior)
        # Promedio de las últimas 4 semanas (incluyendo actual)
        w_df['Vol_Avg_Curr_4W'] = w_df['Volume'].rolling(window=4).mean()
        # Línea base: Promedio de las 12 semanas ANTERIORES a ese bloque de 4
        # (Shift 4 para saltar el mes actual, y miramos 12 semanas atrás como referencia)
        w_df['Vol_Baseline'] = w_df['Volume'].shift(4).rolling(window=12).mean()
        # El bloque actual es el doble de intenso que la base
        w_df['Vol_4W_Expansion'] = np.where(w_df['Vol_Avg_Curr_4W'] >= (2 * w_df['Vol_Baseline']), 1, 0)


        w_df['SMA_Slope'] = w_df['SMA_30W'].diff(4)  # Change over 4 weeks
        # Price was below SMA at some point in the lookback, then recovered
        w_df['Was_Below_SMA'] = (w_df['Close'] < w_df['SMA_30W']).rolling(window=self.lookback).max()

        # Guardamos el ratio para ver la magnitud si quisieras
        w_df['Vol_vs_Avg'] = (w_df['Volume'] / w_df['Vol_Avg_Prev_4W']).round(2)
        # ==============================================================================

        # --- FILTROS DE SETUP ---
        limit_lower = w_df['Major_High'] * (1 - self.proximity)
        cond_near_high = (w_df['Close'] < w_df['Major_High']) & \
                         (w_df['Close'] > limit_lower)

        cond_trend = w_df['Close'] > w_df['SMA_30W']
        cond_touches = w_df['Touch_Count'] >= self.min_touches
        cond_volume = (w_df['Vol_Spike_2x'] == 1) | (w_df['Vol_4W_Expansion'] == 1)
        cond_sma_rising = w_df['SMA_Slope'] > 0
        cond_prior_weakness = w_df['Was_Below_SMA'] == 1


        w_df['Signal'] = np.where(
            cond_trend & cond_near_high & cond_touches & cond_volume & cond_sma_rising,
            1, 0
        )

        w_df['Distance_to_Breakout'] = ((w_df['Major_High'] - w_df['Close']) / w_df['Close']) * 100

        return w_df