from src.strategy import Strategy
import pandas as pd
import numpy as np


class WeinsteinSetup(Strategy):
    """
    Weinstein Stage 1 → Stage 2 Pre-Breakout Setup.

    Detects stocks forming a proper trading range (base) with:
      - A clustered resistance ZONE  (dense ceiling of swing highs)
      - A clustered support ZONE     (dense floor of swing lows)
      - Range width validation       (not too tight, not too loose)
      - Price below resistance zone, above rising 30W SMA
      - Minimum confirmed touches on BOTH resistance and support
      - Volume expansion
    """

    def __init__(
            self,
            sma_weeks: int = 30,
            lookback_weeks: int = 36,
            min_touches: int = 2,
            min_sup_touches: int = 2,
            touch_tolerance: float = 0.01,
            min_range_width: float = 0.08,
            max_range_width: float = 0.45,
    ):
        super().__init__("Weinstein Setup (Pre-Breakout)")
        self.sma_period = sma_weeks
        self.lookback = lookback_weeks
        self.min_touches = min_touches
        self.min_sup_touches = min_sup_touches
        self.tolerance = touch_tolerance
        self.min_range_width = min_range_width
        self.max_range_width = max_range_width

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _swing_extremes(self, arr: np.ndarray, mode: str) -> np.ndarray:
        """
        Return VALUES of all local peaks ('peaks') or troughs ('troughs').
        A peak is >= both neighbours; a trough is <= both neighbours.
        """
        extremes = []
        for i in range(1, len(arr) - 1):
            if mode == "peaks" and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
                extremes.append(arr[i])
            elif mode == "troughs" and arr[i] <= arr[i - 1] and arr[i] <= arr[i + 1]:
                extremes.append(arr[i])
        return np.array(extremes) if extremes else np.array([])

    def _swing_extreme_indices(self, arr: np.ndarray, mode: str) -> list[int]:
        """
        Return LOCAL INDICES (within arr) of swing peaks or troughs.
        Mirror of _swing_extremes but keeps position information so
        we can map touches back to specific bars in the DataFrame.
        """
        indices = []
        for i in range(1, len(arr) - 1):
            if mode == "peaks" and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
                indices.append(i)
            elif mode == "troughs" and arr[i] <= arr[i - 1] and arr[i] <= arr[i + 1]:
                indices.append(i)
        return indices

    def _dominant_cluster(self, extremes: np.ndarray) -> tuple[float, int]:
        """
        Find the densest price cluster among swing extremes.

        For each extreme as anchor, count how many others fall within
        ±tolerance. The largest cluster wins; returns its mean and count.
        """
        extremes = extremes[np.isfinite(extremes) & (extremes > 0)]

        if len(extremes) == 0:
            return np.nan, 0

        best_level = extremes[0]
        best_count = 1

        for anchor in extremes:
            if anchor <= 0 or not np.isfinite(anchor):
                continue
            mask = np.abs(extremes - anchor) / anchor <= self.tolerance
            count = int(np.sum(mask))
            if count > best_count:
                best_count = count
                best_level = float(np.mean(extremes[mask]))

        return best_level, best_count

    def _analyse_window(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> dict:
        """
        Analyse one rolling window and return all trading-range metrics.
        """
        # --- RESISTANCE ---
        swing_highs = self._swing_extremes(highs, "peaks")
        if len(swing_highs) >= 2:
            res_level, res_touches = self._dominant_cluster(swing_highs)
        else:
            res_level = float(np.nanmax(highs)) if len(highs) > 0 else np.nan
            res_touches = 1

        # --- SUPPORT ---
        swing_lows = self._swing_extremes(lows, "troughs")
        if len(swing_lows) >= 2:
            sup_level, sup_touches = self._dominant_cluster(swing_lows)
        else:
            sup_level = float(np.nanmin(lows)) if len(lows) > 0 else np.nan
            sup_touches = 1

        # --- STRICT RANGE VALIDATION ---
        # If the price closes outside the boundaries AFTER the base starts forming,
        # it means the base was broken. Invalidate it so we don't count it twice.

        if np.isfinite(res_level) and res_level > 0:
            strict_res_ceiling = res_level * (1 + self.tolerance)
            # Find the index of the first touch of this resistance cluster
            res_touch_indices = np.where(np.abs(highs - res_level) / res_level <= self.tolerance)[0]
            if len(res_touch_indices) > 0:
                first_res_idx = res_touch_indices[0]
                # If ANY weekly close exceeded the strict ceiling after the first touch, invalidate
                if np.any(closes[first_res_idx:] > strict_res_ceiling):
                    res_touches = 0
                    res_level = np.nan  # Destroys the zone

        if np.isfinite(sup_level) and sup_level > 0:
            strict_sup_floor = sup_level * (1 - self.tolerance)
            # Find the index of the first touch of this support cluster
            sup_touch_indices = np.where(np.abs(lows - sup_level) / sup_level <= self.tolerance)[0]
            if len(sup_touch_indices) > 0:
                first_sup_idx = sup_touch_indices[0]
                # If ANY weekly close fell below the strict floor after the first touch, invalidate
                if np.any(closes[first_sup_idx:] < strict_sup_floor):
                    sup_touches = 0
                    sup_level = np.nan  # Destroys the zone

        # --- ZONE BANDS (±half tolerance around each level) ---
        half_tol = self.tolerance / 2
        res_zone_top = res_level * (1 + half_tol)
        res_zone_bot = res_level * (1 - half_tol)
        sup_zone_top = sup_level * (1 + half_tol)
        sup_zone_bot = sup_level * (1 - half_tol)

        # --- RANGE WIDTH ---
        if sup_level and np.isfinite(sup_level) and sup_level > 0 and np.isfinite(res_level):
            range_width = (res_level - sup_level) / sup_level
        else:
            range_width = np.nan

        return {
            "resistance": res_level,
            "res_zone_top": res_zone_top,
            "res_zone_bot": res_zone_bot,
            "res_touches": res_touches,
            "support": sup_level,
            "sup_zone_top": sup_zone_top,
            "sup_zone_bot": sup_zone_bot,
            "sup_touches": sup_touches,
            "range_width": range_width,
        }

    def _mark_touch_bars(
            self,
            w_df: pd.DataFrame,
            highs: np.ndarray,
            lows: np.ndarray,
            n: int,
    ) -> None:
        """
        Tag the exact bars inside the final lookback window whose swing
        high/low belongs to the dominant resistance/support cluster.
        """
        w_df["Is_Res_Touch"] = False
        w_df["Is_Sup_Touch"] = False

        if n <= self.lookback:
            return

        last = w_df.iloc[-1]
        res_level = last["Resistance"]
        sup_level = last["Support"]
        win_start = n - self.lookback

        w_highs = highs[win_start:n]
        w_lows = lows[win_start:n]

        # Resistance: swing peaks in cluster
        if np.isfinite(res_level) and res_level > 0:
            for local_i in self._swing_extreme_indices(w_highs, "peaks"):
                h = w_highs[local_i]
                if abs(h - res_level) / res_level <= self.tolerance:
                    global_i = win_start + local_i
                    w_df.iat[global_i, w_df.columns.get_loc("Is_Res_Touch")] = True

        # Support: swing troughs in cluster
        if np.isfinite(sup_level) and sup_level > 0:
            for local_i in self._swing_extreme_indices(w_lows, "troughs"):
                l = w_lows[local_i]
                if abs(l - sup_level) / sup_level <= self.tolerance:
                    global_i = win_start + local_i
                    w_df.iat[global_i, w_df.columns.get_loc("Is_Sup_Touch")] = True

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:

        # ── 1. RESAMPLE TO WEEKLY ──────────────────────────────────────
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        logic = {"Open": "first", "High": "max", "Low": "min",
                 "Close": "last", "Volume": "sum"}
        w_df = df.resample("W-FRI").agg(logic).dropna()

        n = len(w_df)
        highs = w_df["High"].values
        lows = w_df["Low"].values
        closes = w_df["Close"].values

        # ── 2. ROLLING TRADING RANGE ───────────────────────────────────
        cols = ["Resistance", "Res_Zone_Top", "Res_Zone_Bot", "Res_Touches",
                "Support", "Sup_Zone_Top", "Sup_Zone_Bot", "Sup_Touches",
                "Range_Width_Pct"]
        arrays = {c: np.full(n, np.nan) for c in cols}

        for i in range(self.lookback, n):
            w_high = highs[i - self.lookback: i]
            w_low = lows[i - self.lookback: i]
            w_close = closes[i - self.lookback: i]

            m = self._analyse_window(w_high, w_low, w_close)

            arrays["Resistance"][i] = m["resistance"]
            arrays["Res_Zone_Top"][i] = m["res_zone_top"]
            arrays["Res_Zone_Bot"][i] = m["res_zone_bot"]
            arrays["Res_Touches"][i] = m["res_touches"]
            arrays["Support"][i] = m["support"]
            arrays["Sup_Zone_Top"][i] = m["sup_zone_top"]
            arrays["Sup_Zone_Bot"][i] = m["sup_zone_bot"]
            arrays["Sup_Touches"][i] = m["sup_touches"]
            arrays["Range_Width_Pct"][i] = m["range_width"] * 100

        for col, arr in arrays.items():
            w_df[col] = arr

        # ── 3. MARK CLUSTER TOUCH BARS (for the plotter) ──────────────
        self._mark_touch_bars(w_df, highs, lows, n)

        # ── 4. TREND INDICATOR (30W SMA + SLOPE) ──────────────────────
        w_df["SMA_30W"] = w_df["Close"].rolling(window=self.sma_period).mean()
        w_df["SMA_Slope"] = w_df["SMA_30W"].diff(4)

        # ── 5. VOLUME FILTERS ──────────────────────────────────────────
        w_df["Vol_Avg_Prev_4W"] = w_df["Volume"].shift(1).rolling(window=4).mean()
        w_df["Vol_Spike_2x"] = np.where(
            w_df["Volume"] >= 2 * w_df["Vol_Avg_Prev_4W"], 1, 0)

        w_df["Vol_Avg_Curr_4W"] = w_df["Volume"].rolling(window=4).mean()
        w_df["Vol_Baseline"] = w_df["Volume"].shift(4).rolling(window=12).mean()
        w_df["Vol_4W_Expansion"] = np.where(
            w_df["Vol_Avg_Curr_4W"] >= 2 * w_df["Vol_Baseline"], 1, 0)

        w_df["Vol_vs_Avg"] = (w_df["Volume"] / w_df["Vol_Avg_Prev_4W"]).round(2)

        # ── 6. SIGNAL CONDITIONS ───────────────────────────────────────

        # 30W SMA rising
        cond_sma_rising = w_df["SMA_Slope"] > 0

        # Price above 30W SMA
        cond_above_sma = w_df["Close"] > w_df["SMA_30W"]

        # Enough confirmed resistance touches (Will be False if range was invalidated)
        cond_res_touches = w_df["Res_Touches"] >= self.min_touches

        # Enough confirmed support touches
        cond_sup_touches = w_df["Sup_Touches"] >= self.min_sup_touches

        # Volume expanding
        cond_volume = (w_df["Vol_Spike_2x"] == 1) | (w_df["Vol_4W_Expansion"] == 1)

        w_df["Signal"] = np.where(
            cond_sma_rising &
            cond_above_sma &
            cond_res_touches &
            cond_sup_touches ,#&
            #cond_volume,
            1, 0
        )

        # ── 7. DISTANCE METRIC ─────────────────────────────────────────
        w_df["Distance_to_Breakout"] = (
                (w_df["Res_Zone_Bot"] - w_df["Close"]) / w_df["Close"] * 100
        )

        return w_df