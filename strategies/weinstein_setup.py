"""
weinstein_setup.py
------------------
Weinstein Stage 1 -> Stage 2 pre-breakout setup, as a self-describing plugin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import ParamSpec, Strategy, StrategyMeta
from src.registry import register


@register
class WeinsteinSetup(Strategy):

    meta = StrategyMeta(
        key="weinstein",
        display_name="Weinstein Stage Setup",
        description=(
            "Detects Weinstein pre-breakout Stage 1 setups: stocks that came "
            "from a decline, built a base, with the 30W MA flattening and price "
            "coiled below resistance — about to break out but NOT yet. The "
            "'base_length' preset controls how long a base must be (short catches "
            "forming bottoms early; long requires mature bases). Designed to "
            "catch stocks before the breakout and avoid extended names."
        ),
        signal_column="Signal",
        hit_values=(1,),
        param_schema=(
            ParamSpec("base_length", "short",
                      lambda s: str(s).strip().lower(),
                      "Base length preset: 'short' (~8w, catches forming bottoms early), "
                      "'medium' (~14w), 'long' (~24w, mature bases), or 'custom' to use "
                      "lookback_weeks/min_base_weeks directly.",
                      choices=("short", "medium", "long", "custom")),
            ParamSpec("sma_weeks", 30, int, "Weeks for the 30W moving average"),
            ParamSpec("lookback_weeks", 36, int,
                      "Analysis window (weeks). Only used when base_length='custom'."),
            ParamSpec("min_base_weeks", 10, int,
                      "Min weeks in the base. Only used when base_length='custom'."),
            ParamSpec("min_touches", 2, int, "Minimum touches to RESISTANCE"),
            ParamSpec("min_sup_touches", 2, int, "Minimum touches to SUPPORT"),
            ParamSpec("touch_tolerance", 0.015, float, "Touch tolerance (fraction, 0.015 = 1.5%)"),
            ParamSpec("max_base_breach_frac", 0.10, float,
                      "Max fraction of base weeks whose CLOSE may sit outside the "
                      "support/resistance zone. Low = the base must respect its levels "
                      "(a real trading range). Higher = tolerate a sloppier range."),
            ParamSpec("max_range_vs_decline", 1.2, float,
                      "Max base width as a fraction of the prior decline. A healthy base "
                      "consolidates within a fraction of what it fell; rejects wide "
                      "oscillation while accepting tight bases at any price level."),
            ParamSpec("max_dist_to_breakout", 30.0, float,
                      "pre_breakout: max %% below resistance (price still in/near the base)"),
            ParamSpec("min_prior_decline", 0.15, float,
                      "Min drop from the pre-base peak (fraction). A real Stage 1 follows "
                      "a considerable decline; the key filter against extended names."),
            ParamSpec("max_ma_decline_pct", 8.0, float,
                      "Max the 30W MA may be FALLING (5-week slope, %%). Generous: we "
                      "want to catch bottoms early while the MA is still declining."),
            ParamSpec("max_ma_rise_pct", 1.5, float,
                      "Max the 30W MA may be RISING (5-week slope, %%). Strict: a Stage-1 "
                      "base has a flat/falling MA. A rising MA means the stock already "
                      "trended up (Stage 2) or is consolidating within an uptrend — NOT a "
                      "Stage-1 bottom. Rejects names that just rallied into a range."),
            ParamSpec("require_volume", False,
                      lambda s: str(s).strip().lower() in ("1", "true", "yes", "y"),
                      "Require volume expansion (off by default; a confirming filter)"),
            ParamSpec("require_rs_positive", False,
                      lambda s: str(s).strip().lower() in ("1", "true", "yes", "y"),
                      "Require Mansfield RS > 0 (off by default; filter by eye afterwards)"),
            ParamSpec("rs_period", 52, int, "Mansfield RS lookback (weeks)"),
        ),
        display_columns=(
            "Stage", "Resistance", "Support", "Range_Width_Pct", "Base_Weeks",
            "Distance_to_Breakout", "Prior_Decline_Pct", "Mansfield_RS",
            "Res_Touches", "Sup_Touches", "Vol_Spike_2x", "Vol_4W_Expansion",
            "Sector", "Industry",
        ),
        sort_by=("Market", "Distance_to_Breakout"),
        sort_ascending=(True, True),
    )

    # Preset table: base_length -> (lookback_weeks, min_base_weeks, min_sup_touches).
    # A forming bottom ('short') has often bounced off support only ONCE, so we
    # relax the support-touch requirement there; mature bases keep 2.
    _BASE_PRESETS = {
        "short":  (20, 8, 1),
        "medium": (30, 14, 2),
        "long":   (44, 24, 2),
    }

    # ---- convenient aliases onto validated params -------------------------
    @property
    def sma_period(self) -> int:
        return self.params["sma_weeks"]

    @property
    def _preset(self):
        return self._BASE_PRESETS.get(self.params["base_length"])

    @property
    def lookback(self) -> int:
        preset = self._preset
        return preset[0] if preset else self.params["lookback_weeks"]

    @property
    def min_base(self) -> int:
        preset = self._preset
        return preset[1] if preset else self.params["min_base_weeks"]

    @property
    def min_sup_touches(self) -> int:
        preset = self._preset
        return preset[2] if preset else self.params["min_sup_touches"]

    @property
    def tolerance(self) -> float:
        return self.params["touch_tolerance"]

    # ---- benchmark injection (for Mansfield relative strength) ------------
    def __init__(self, **params):
        super().__init__(**params)
        self._benchmark_weekly = None  # set per-market by the screener

    def set_benchmark(self, weekly_close):
        """Receive the market's benchmark weekly close series (for Mansfield RS)."""
        self._benchmark_weekly = weekly_close

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------
    def _swing_extremes(self, arr: np.ndarray, mode: str) -> np.ndarray:
        extremes = []
        for i in range(1, len(arr) - 1):
            if mode == "peaks" and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
                extremes.append(arr[i])
            elif mode == "troughs" and arr[i] <= arr[i - 1] and arr[i] <= arr[i + 1]:
                extremes.append(arr[i])
        return np.array(extremes) if extremes else np.array([])

    def _swing_extreme_indices(self, arr: np.ndarray, mode: str) -> list[int]:
        indices = []
        for i in range(1, len(arr) - 1):
            if mode == "peaks" and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
                indices.append(i)
            elif mode == "troughs" and arr[i] <= arr[i - 1] and arr[i] <= arr[i + 1]:
                indices.append(i)
        return indices

    def _dominant_cluster(self, extremes: np.ndarray) -> tuple[float, int]:
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

    def _analyse_window(self, highs, lows, closes) -> dict:
        swing_highs = self._swing_extremes(highs, "peaks")
        if len(swing_highs) >= 2:
            res_level, res_touches = self._dominant_cluster(swing_highs)
        else:
            res_level = float(np.nanmax(highs)) if len(highs) > 0 else np.nan
            res_touches = 1

        swing_lows = self._swing_extremes(lows, "troughs")
        if len(swing_lows) >= 2:
            sup_level, sup_touches = self._dominant_cluster(swing_lows)
        else:
            sup_level = float(np.nanmin(lows)) if len(lows) > 0 else np.nan
            sup_touches = 1

        if np.isfinite(res_level) and res_level > 0:
            strict_res_ceiling = res_level * (1 + self.tolerance)
            res_touch_indices = np.where(np.abs(highs - res_level) / res_level <= self.tolerance)[0]
            if len(res_touch_indices) > 0:
                first_res_idx = res_touch_indices[0]
                breaches = int(np.sum(closes[first_res_idx:] > strict_res_ceiling))
                if breaches >= 2:
                    res_touches = 0
                    res_level = np.nan

        if np.isfinite(sup_level) and sup_level > 0:
            strict_sup_floor = sup_level * (1 - self.tolerance)
            sup_touch_indices = np.where(np.abs(lows - sup_level) / sup_level <= self.tolerance)[0]
            if len(sup_touch_indices) > 0:
                first_sup_idx = sup_touch_indices[0]
                breaches = int(np.sum(closes[first_sup_idx:] < strict_sup_floor))
                if breaches >= 2:
                    sup_touches = 0
                    sup_level = np.nan

        half_tol = self.tolerance / 2
        res_zone_top = res_level * (1 + half_tol)
        res_zone_bot = res_level * (1 - half_tol)
        sup_zone_top = sup_level * (1 + half_tol)
        sup_zone_bot = sup_level * (1 - half_tol)

        if sup_level and np.isfinite(sup_level) and sup_level > 0 and np.isfinite(res_level):
            range_width = (res_level - sup_level) / sup_level
        else:
            range_width = np.nan

        return {
            "resistance": res_level, "res_zone_top": res_zone_top,
            "res_zone_bot": res_zone_bot, "res_touches": res_touches,
            "support": sup_level, "sup_zone_top": sup_zone_top,
            "sup_zone_bot": sup_zone_bot, "sup_touches": sup_touches,
            "range_width": range_width,
        }

    def _mark_touch_bars(self, w_df, highs, lows, n) -> None:
        w_df["Is_Res_Touch"] = False
        w_df["Is_Sup_Touch"] = False
        if n <= self.lookback:
            return
        last = w_df.iloc[-1]
        res_level = last["Resistance"]
        sup_level = last["Support"]
        win_start = (n - 1) - self.lookback
        win_end = n - 1
        if win_start < 0:
            return
        w_highs = highs[win_start:win_end]
        w_lows = lows[win_start:win_end]
        if np.isfinite(res_level) and res_level > 0:
            for local_i in self._swing_extreme_indices(w_highs, "peaks"):
                h = w_highs[local_i]
                if abs(h - res_level) / res_level <= self.tolerance:
                    w_df.iat[win_start + local_i, w_df.columns.get_loc("Is_Res_Touch")] = True
        if np.isfinite(sup_level) and sup_level > 0:
            for local_i in self._swing_extreme_indices(w_lows, "troughs"):
                l = w_lows[local_i]
                if abs(l - sup_level) / sup_level <= self.tolerance:
                    w_df.iat[win_start + local_i, w_df.columns.get_loc("Is_Sup_Touch")] = True

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        logic = {"Open": "first", "High": "max", "Low": "min",
                 "Close": "last", "Volume": "sum"}
        w_df = df.resample("W-FRI").agg(logic).dropna()

        n = len(w_df)
        highs = w_df["High"].values
        lows = w_df["Low"].values
        closes = w_df["Close"].values

        cols = ["Resistance", "Res_Zone_Top", "Res_Zone_Bot", "Res_Touches",
                "Support", "Sup_Zone_Top", "Sup_Zone_Bot", "Sup_Touches",
                "Range_Width_Pct"]
        arrays = {c: np.full(n, np.nan) for c in cols}

        for i in range(self.lookback, n):
            w_high = highs[i - self.lookback:i]
            w_low = lows[i - self.lookback:i]
            w_close = closes[i - self.lookback:i]
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

        self._mark_touch_bars(w_df, highs, lows, n)

        w_df["SMA_30W"] = w_df["Close"].rolling(window=self.sma_period).mean()
        w_df["SMA_Slope"] = w_df["SMA_30W"].diff(5)
        w_df["SMA_Slope_Pct"] = (w_df["SMA_Slope"] / w_df["SMA_30W"]) * 100

        w_df["Vol_Avg_Prev_4W"] = w_df["Volume"].shift(1).rolling(window=4).mean()
        w_df["Vol_Spike_2x"] = np.where(w_df["Volume"] >= 2 * w_df["Vol_Avg_Prev_4W"], 1, 0)
        w_df["Vol_Avg_Curr_4W"] = w_df["Volume"].rolling(window=4).mean()
        w_df["Vol_Baseline"] = w_df["Volume"].shift(4).rolling(window=12).mean()
        w_df["Vol_4W_Expansion"] = np.where(w_df["Vol_Avg_Curr_4W"] >= 2 * w_df["Vol_Baseline"], 1, 0)
        w_df["Vol_vs_Avg"] = (w_df["Volume"] / w_df["Vol_Avg_Prev_4W"]).round(2)

        w_df["Distance_to_Breakout"] = (
            (w_df["Res_Zone_Bot"] - w_df["Close"]) / w_df["Close"] * 100)

        w_df["Base_Weeks"] = self._base_duration(w_df)
        w_df["Prior_Decline_Pct"] = self._prior_decline(w_df) * 100
        w_df["Mansfield_RS"] = self._mansfield(w_df)
        w_df["Weeks_Since_Breakout"] = self._weeks_since_breakout(w_df)

        # ============================ SIGNAL ============================
        p = self.params

        cond_res_touches = w_df["Res_Touches"] >= p["min_touches"]
        cond_sup_touches = w_df["Sup_Touches"] >= self.min_sup_touches
        with np.errstate(divide="ignore", invalid="ignore"):
            width_vs_decline = w_df["Range_Width_Pct"] / w_df["Prior_Decline_Pct"]
        cond_width = width_vs_decline <= p["max_range_vs_decline"]
        cond_base_len = w_df["Base_Weeks"] >= self.min_base
        cond_prior_decline = w_df["Prior_Decline_Pct"] >= p["min_prior_decline"] * 100
        cond_volume = (w_df["Vol_Spike_2x"] == 1) | (w_df["Vol_4W_Expansion"] == 1)

        rs_ok = w_df["Mansfield_RS"] > 0
        if self._benchmark_weekly is None:
            rs_ok = pd.Series(True, index=w_df.index)

        base_common = cond_res_touches & cond_sup_touches & cond_width & cond_base_len

        # Pre-breakout: coiled under resistance, base built after a decline, 30W
        # MA flat (not necessarily rising yet), price NOT yet broken out, near
        # the breakout level. We catch the stock BEFORE it breaks.
        cond_coiled = (w_df["Distance_to_Breakout"] > 0) & \
                      (w_df["Distance_to_Breakout"] <= p["max_dist_to_breakout"])
        # "Not broken": no breakout on record, or the last one is older than the
        # base itself (i.e. not a fresh breakout we'd be chasing).
        cond_not_broken = w_df["Weeks_Since_Breakout"].isna() | \
                          (w_df["Weeks_Since_Breakout"] > self.min_base)
        # MA must be flat or falling — NOT rising. Asymmetric band: tolerate a
        # steep decline (catch early bottoms) but reject a rising MA (stock that
        # already turned up / is consolidating in an uptrend, like ABUS).
        cond_ma_flat = (w_df["SMA_Slope_Pct"] >= -p["max_ma_decline_pct"]) & \
                       (w_df["SMA_Slope_Pct"] <= p["max_ma_rise_pct"])
        cond_revisited = pd.Series(self._touched_support_recently(w_df) >= 1,
                                   index=w_df.index)
        signal = (base_common & cond_coiled & cond_not_broken &
                  cond_ma_flat & cond_prior_decline & cond_revisited)
        if p["require_volume"]:
            signal = signal & cond_volume
        if p["require_rs_positive"]:
            signal = signal & rs_ok

        self._cond_cols = {
            "res_touches": cond_res_touches,
            "sup_touches": cond_sup_touches,
            "width_vs_decline": cond_width,
            "base_len": cond_base_len,
            "prior_decline": cond_prior_decline,
            "coiled": cond_coiled,
            "not_broken": cond_not_broken,
            "ma_flat": cond_ma_flat,
            "revisited_support": cond_revisited,
            "volume": cond_volume if p["require_volume"] else pd.Series(True, index=w_df.index),
            "rs_positive": rs_ok if p["require_rs_positive"] else pd.Series(True, index=w_df.index),
        }
        for cname, cseries in self._cond_cols.items():
            w_df[f"Cond_{cname}"] = cseries.fillna(False).astype(bool)

        w_df["Stage"] = "pre_breakout"
        w_df["Signal"] = np.where(signal, 1, 0)
        return w_df

    # ------------------------------------------------------------------
    # METRIC HELPERS
    # ------------------------------------------------------------------
    def _base_duration(self, w_df: pd.DataFrame) -> np.ndarray:
        n = len(w_df)
        out = np.zeros(n)
        closes = w_df["Close"].values
        sup = w_df["Sup_Zone_Bot"].values
        res = w_df["Res_Zone_Top"].values
        # Fraction of weeks allowed to close OUTSIDE the support/resistance zone
        # band during the base. The band already includes a tolerance margin, so
        # a close outside it is a real breach. Keep this small: the base must
        # actually respect its levels (a genuine trading range), not merely
        # spend "most" of its time near them.
        budget = self.params["max_base_breach_frac"]
        for i in range(n):
            if not (np.isfinite(sup[i]) and np.isfinite(res[i])):
                continue
            out_band = 0
            span = 0
            for j in range(i, -1, -1):
                inside = sup[i] <= closes[j] <= res[i]
                if not inside:
                    out_band += 1
                span += 1
                # Allow 1 isolated breach, then enforce the fraction budget.
                if out_band > 1 and out_band > budget * span:
                    span -= 1
                    break
            out[i] = span
        return out

    def _prior_decline(self, w_df: pd.DataFrame) -> np.ndarray:
        n = len(w_df)
        out = np.full(n, np.nan)
        highs = w_df["High"].values
        res = w_df["Resistance"].values
        base_weeks = w_df["Base_Weeks"].values if "Base_Weeks" in w_df.columns else None
        for i in range(n):
            if not np.isfinite(res[i]):
                continue
            base_len = int(base_weeks[i]) if base_weeks is not None and np.isfinite(base_weeks[i]) else self.lookback
            base_start = max(0, i - base_len)
            if base_start <= 0:
                continue
            pre_peak = float(np.nanmax(highs[:base_start]))
            if pre_peak <= 0:
                continue
            out[i] = (pre_peak - res[i]) / pre_peak
        return out

    def _weeks_since_breakout(self, w_df: pd.DataFrame) -> np.ndarray:
        n = len(w_df)
        out = np.full(n, np.nan)
        closes = w_df["Close"].values
        res_top = w_df["Res_Zone_Top"].values
        base_weeks = w_df["Base_Weeks"].values if "Base_Weeks" in w_df.columns else None
        min_base = max(4, self.min_base // 2)
        last_breakout = None
        for i in range(n):
            if i > 0 and np.isfinite(res_top[i]) and closes[i] > res_top[i] >= closes[i - 1]:
                established = (base_weeks is None or
                              (np.isfinite(base_weeks[i - 1]) and base_weeks[i - 1] >= min_base))
                if established:
                    last_breakout = i
            if last_breakout is not None:
                out[i] = i - last_breakout
        return out

    def _touched_support_recently(self, w_df: pd.DataFrame) -> np.ndarray:
        n = len(w_df)
        out = np.zeros(n)
        lows = w_df["Low"].values
        sup = w_df["Support"].values
        base_weeks = w_df["Base_Weeks"].values
        tol = self.tolerance
        for i in range(n):
            if not (np.isfinite(sup[i]) and sup[i] > 0):
                continue
            recent = max(2, int(base_weeks[i] // 2)) if np.isfinite(base_weeks[i]) else 4
            start = max(0, i - recent + 1)
            seg = lows[start:i + 1]
            out[i] = int(np.sum(np.abs(seg - sup[i]) / sup[i] <= tol))
        return out

    def _mansfield(self, w_df: pd.DataFrame) -> pd.Series:
        if self._benchmark_weekly is None:
            return pd.Series(np.nan, index=w_df.index)
        from src.benchmarks import mansfield_rs
        mrs = mansfield_rs(w_df["Close"], self._benchmark_weekly,
                           n=self.params["rs_period"])
        return mrs.reindex(w_df.index)
