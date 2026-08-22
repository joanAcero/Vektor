from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.strategy import Strategy, StrategyMeta
from src.registry import register
from src.stages import (MA_FAST_WEEKS, MA_SLOPE_WEEKS, MIN_DECLINE_PCT,
                        SMA_WEEKS, Stage, classify as classify_stages)

log = logging.getLogger(__name__)

# ======================================================================
# BOOK CONSTANTS -- explicitly stated by Weinstein. Do not tune.
# ======================================================================
# SMA_WEEKS (30) and MA_SLOPE_WEEKS (5) are imported from src/stages.py, which
# owns the 30-week MA and its slope for the whole framework. They are book
# constants, but they must be book constants in exactly ONE place: the MA this
# strategy gates on and the MA the stage label is derived from have to be the
# same series, or a stock can be "Stage 1" and fail the MA-shape gate at once.
RS_PERIOD_WEEKS = 52        # Mansfield RS zero line = 1-year RP average
VOL_BREAKOUT_MULT = 2.0     # breakout volume >= 2x ...
VOL_AVG_WEEKS = 4           # ... the previous 4-week average (weekly chart)
VOL_BUILDUP_WEEKS = 8       # accumulation window before a breakout

# ======================================================================
# CALIBRATION CONSTANTS -- NOT in the book. Validate vs reference charts:
# UBER, SONY (early); RACE, CRDA, ANE (mid); PFE, CPR.MI, CRON, PCRX, BIO,
# GPN, TMO (long/valid v3 catches that must keep passing).
# ======================================================================
MIN_STAB_WEEKS = 10         # validity floor for the stabilization. LOW on
                            # purpose (SONY ~18w, UBER ~25w must pass with
                            # margin). With the score gone, nothing in the
                            # system prefers a longer base -- Base_Weeks is
                            # displayed and that judgement is now manual.
MAX_STAB_WIDTH = 0.40       # close-to-close band that defines "stabilized"
STAB_OUTLIER_FRAC = 0.05    # bull traps / undershoots are ISOLATED closes
                            # outside the band; up to 1 + 5% of the span may
                            # be skipped instead of terminating the scan
                            # (same rationale as the original breach budget:
                            # two single-week excursions on opposite sides
                            # must not veto a two-year structure)
STAB_DRIFT_FRAC = 0.4       # a span is RECORDED as a base only if its LSQ
                            # drift is <= this fraction of its own range
                            # (horizontality). Drift is non-monotone in span
                            # length -- high across one oscillation leg, low
                            # across full cycles, high again once the window
                            # tunnels into the prior decline -- so the scan
                            # records the LARGEST compliant span rather than
                            # stopping at the first violation.
#                             (no hard-abort: intra-base legs can run 10+
#                             monotone weeks, so any consecutive-violation
#                             abort makes Base_Weeks depend on the phase of
#                             the last oscillation and the signal flickers.
#                             Decline-contaminated windows simply never meet
#                             the recording criterion, which is the actual
#                             anti-tunnel mechanism.)
MAX_LEVEL_WIDTH = 0.50      # high/low width cap on the displayed range
                            # (wicks legitimately run wider than the 40%
                            # close-based band; PFE-type bases hit ~47%)
MIN_PRIOR_DECLINE = MIN_DECLINE_PCT / 100.0
                            # "a considerable decline" precedes a Stage 1.
                            # Derived from src/stages.py so the GATE and the
                            # stage LABEL cannot drift apart -- they are the
                            # same test asked by two different callers.
DECLINE_WINDOW_WEEKS = 260  # peak search window before the base (5y)
MA_FLAT_BAND = (-1.0, 1.5)  # 5w %-slope band = "flat" (mature-base branch)
MAX_MA_RISE_PCT = 1.5       # clearly rising MA = already trended, reject
MAX_MA_DECLINE_PCT = 4.0    # freefall floor for the early-base branch
MA_DECEL_LOOKBACK = 8       # deceleration reference: slope vs 8w ago
MA_DECEL_MARGIN = 0.25      # slope must have improved by at least this
MAX_BELOW_MA = 0.10         # early branch: price within 10% below the MA.
                            # This distance is what separates "base with the
                            # MA catching down onto it" (UBER 7%, SONY 8%)
                            # from "pause far beneath a Stage-4 MA" (typical
                            # mid-decline pause sits 15-25% under its MA).
MAX_CLUSTER_WEEKS = 156     # cap on the Tier-2 cluster window. The window
                            # IS the stabilization span (clusters describe
                            # THIS base, so old structure above it cannot
                            # out-vote the base's own resistance), capped
                            # for cost on multi-year bases.
TOUCH_TOL_MIN = 0.015       # touch tolerance floor
TOUCH_TOL_MAX = 0.04        # ...and cap
TOUCH_TOL_VOL_MULT = 0.6    # tolerance = this x median weekly range
MIN_TOUCH_SEP_WEEKS = 3     # touches closer than this are one test
BREAKOUT_CONSEC = 3         # sustained break = 3 consecutive weekly closes
RS_SLOPE_WEEKS = 10         # RS trend column (display only)
# STAGE_TREND_PCT and STAGE_FALLBACK_WEEKS used to live here and were
# duplicated in src/rotation.py with a different fallback window. Both now
# live in src/stages.py as TREND_SLOPE_PCT and FALLBACK_WEEKS.


def _max_consecutive(mask: np.ndarray) -> int:
    """Length of the longest run of True values in a boolean array."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        if run > best:
            best = run
    return best


def _rolling_lsq_slope(s: pd.Series, window: int) -> pd.Series:
    """Least-squares slope per rolling window (consistent with
    src/rotation.py's regression-slope choice)."""
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denom = float((x * x).sum())

    def _slope(vals: np.ndarray) -> float:
        if np.any(~np.isfinite(vals)):
            return np.nan
        return float((x * (vals - vals.mean())).sum() / denom)

    return s.rolling(window).apply(_slope, raw=True)


@register
class WeinsteinSetup(Strategy):

    meta = StrategyMeta(
        key="weinstein",
        display_name="Weinstein Stage-1 Base (pre-breakout)",
        description=(
            "Fixed Weinstein system, no tunables, recall-first: detects "
            "price STABILIZATIONS (>= 10 weeks of closes holding a <= 40% "
            "band) sitting >= 15% below their pre-base peak, with a 30W MA "
            "that is either flat or falling-but-decelerating with price "
            "converged onto it -- so it catches early bases while the MA is "
            "still catching down (UBER/SONY-type), mid bases (Ferrari-type) "
            "and long mature bases (PFE-type). Swing-cluster S/R levels, "
            "touch counts, RS and volume character refine the displayed "
            "levels and the chart but never gate the signal. Listed closest "
            "to its breakout first; the buy decision stays manual at the "
            "breakout."
        ),
        signal_column="Signal",
        hit_values=(1,),
        param_schema=(),  # fixed system: one configuration, run consistently
        display_columns=(
            "Stage", "Resistance", "Support",
            "Range_Width_Pct", "Base_Weeks", "Distance_to_Breakout",
            "Prior_Decline_Pct", "Mansfield_RS", "RS_Slope_10W",
            "Res_Touches", "Sup_Touches", "Vol_Dryup", "Vol_Buildup_8W",
            "Sector", "Industry",
        ),
        # One measured quantity, ascending: nearest to triggering first.
        # Deliberately NOT a composite -- see the module docstring.
        sort_by=("Market", "Distance_to_Breakout"),
        sort_ascending=(True, True),
    )

    # ---- benchmark injection (for Mansfield relative strength) ------------
    def __init__(self, **params):
        super().__init__(**params)
        self._benchmark_weekly = None   # injected per-market by the Screener
        self._warned_no_benchmark = False

    def set_benchmark(self, weekly_close) -> None:
        """Receive the market's benchmark weekly close series (Mansfield RS)."""
        self._benchmark_weekly = weekly_close

    # ------------------------------------------------------------------
    # TIER 1 -- STABILIZATION (the primary Stage-1 evidence)
    # ------------------------------------------------------------------
    @staticmethod
    def _stabilization(closes: np.ndarray) -> np.ndarray:
        """For each bar, the LARGEST trailing span of weekly closes that is
        (a) BOUNDED: total range within MAX_STAB_WIDTH, and
        (b) HORIZONTAL: absolute LSQ drift across the span at most
            STAB_DRIFT_FRAC of the span's own range.
        (a) alone is insufficient: any decline shallower than the band "fits"
        and the walk-back tunnels through it into the prior plateau. (b) is
        what separates a base (oscillation) from a trend segment (drift) --
        but drift is non-monotone in span length (see constants), so the scan
        RECORDS the largest compliant span and only aborts on a sustained
        hard violation (STAB_HARD_RUN consecutive extensions with drift >
        STAB_DRIFT_HARD of range), i.e. once it is demonstrably inside a
        trend. Regression sums are maintained incrementally: O(1) per
        extension."""
        n = len(closes)
        out = np.zeros(n)
        for i in range(n):
            ci = closes[i]
            if not np.isfinite(ci) or ci <= 0:
                continue
            c_hi = c_lo = ci
            s1 = sx = sxy = sxx = 0.0
            m = 0
            best = 0
            skipped = 0
            for j in range(i, -1, -1):
                cj = closes[j]
                if not np.isfinite(cj) or cj <= 0:
                    break
                nh = max(c_hi, cj)
                nl = min(c_lo, cj)
                if (nh - nl) / nl > MAX_STAB_WIDTH:
                    # isolated outlier close (bull trap / undershoot): skip it
                    # against the budget instead of terminating; it neither
                    # widens the band nor enters the regression, but it DOES
                    # count toward the base's chronological length.
                    if skipped < 1 + int(STAB_OUTLIER_FRAC * (m + skipped)):
                        skipped += 1
                        continue
                    break
                m_t = m + 1
                s1_t, sx_t = s1 + cj, sx + j
                sxy_t, sxx_t = sxy + j * cj, sxx + j * j
                total = m_t + skipped
                if m_t >= 3 and nh > nl:
                    denom = m_t * sxx_t - sx_t * sx_t
                    slope = (m_t * sxy_t - sx_t * s1_t) / denom if denom > 0 else 0.0
                    rel = abs(slope) * (m_t - 1) / (nh - nl)
                    if rel <= STAB_DRIFT_FRAC:
                        best = total
                else:
                    best = total
                c_hi, c_lo = nh, nl
                s1, sx, sxy, sxx, m = s1_t, sx_t, sxy_t, sxx_t, m_t
            out[i] = best
        return out

    @staticmethod
    def _range_bounds(highs: np.ndarray, lows: np.ndarray,
                      spans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fallback range levels: highest high / lowest low over each bar's
        stabilization span. Always defined when a stabilization exists, so
        every candidate has levels for charts and distance metrics."""
        n = len(spans)
        top = np.full(n, np.nan)
        bot = np.full(n, np.nan)
        for i in range(n):
            k = int(spans[i])
            if k < 2:
                continue
            top[i] = float(np.nanmax(highs[i - k + 1:i + 1]))
            bot[i] = float(np.nanmin(lows[i - k + 1:i + 1]))
        return top, bot

    # ------------------------------------------------------------------
    # TIER 2 -- SWING-CLUSTER LEVELS (display refinement only)
    # ------------------------------------------------------------------
    @staticmethod
    def _effective_tol(highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray) -> float:
        with np.errstate(invalid="ignore", divide="ignore"):
            rng = (highs - lows) / closes
        rng = rng[np.isfinite(rng) & (rng > 0)]
        if len(rng) == 0:
            return TOUCH_TOL_MIN
        return float(np.clip(TOUCH_TOL_VOL_MULT * np.median(rng),
                             TOUCH_TOL_MIN, TOUCH_TOL_MAX))

    @staticmethod
    def _swing_points(arr: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """(indices, values) of swing extremes, plateaus collapsed."""
        idxs: list[int] = []
        vals: list[float] = []
        n = len(arr)
        i = 1
        while i < n - 1:
            v = arr[i]
            if mode == "peaks":
                is_ext = v >= arr[i - 1] and v >= arr[i + 1]
            else:
                is_ext = v <= arr[i - 1] and v <= arr[i + 1]
            if is_ext:
                j = i
                while j + 1 < n - 1 and arr[j + 1] == v:
                    j += 1
                idxs.append(i)
                vals.append(float(v))
                i = j + 1
            else:
                i += 1
        return np.asarray(idxs, dtype=int), np.asarray(vals, dtype=float)

    @staticmethod
    def _separated(indices: np.ndarray) -> list[int]:
        if len(indices) == 0:
            return []
        kept = [int(indices[0])]
        for k in indices[1:]:
            if k - kept[-1] >= MIN_TOUCH_SEP_WEEKS:
                kept.append(int(k))
        return kept

    def _dominant_cluster(self, idxs: np.ndarray, vals: np.ndarray,
                          tol: float) -> tuple[float, int, list[int]]:
        finite = np.isfinite(vals) & (vals > 0)
        idxs, vals = idxs[finite], vals[finite]
        if len(vals) == 0:
            return np.nan, 0, []
        best_level, best_kept = np.nan, []
        for anchor in vals:
            mask = np.abs(vals - anchor) / anchor <= tol
            kept = self._separated(np.sort(idxs[mask]))
            if len(kept) > len(best_kept):
                best_kept = kept
                best_level = float(np.mean(vals[mask]))
        return best_level, len(best_kept), best_kept

    def _analyse_window(self, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray) -> dict:
        tol = self._effective_tol(highs, lows, closes)

        pk_i, pk_v = self._swing_points(highs, "peaks")
        if len(pk_v) >= 2:
            res_level, res_touches, res_kept = self._dominant_cluster(pk_i, pk_v, tol)
        else:
            res_level, res_touches, res_kept = np.nan, 0, []

        tr_i, tr_v = self._swing_points(lows, "troughs")
        if len(tr_v) >= 2:
            sup_level, sup_touches, sup_kept = self._dominant_cluster(tr_i, tr_v, tol)
        else:
            sup_level, sup_touches, sup_kept = np.nan, 0, []

        # Invalidation: only a SUSTAINED decisive break (3+ consecutive
        # closes past 2*tol) after the level was established (2nd touch),
        # never revisited afterwards, kills a level. Bull traps survive.
        if np.isfinite(res_level) and res_level > 0 and len(res_kept) >= 2:
            anchor = res_kept[1]
            ceiling = res_level * (1 + 2 * tol)
            if _max_consecutive(closes[anchor:] > ceiling) >= BREAKOUT_CONSEC \
                    and res_kept[-1] <= anchor:
                res_level, res_touches = np.nan, 0
        if np.isfinite(sup_level) and sup_level > 0 and len(sup_kept) >= 2:
            anchor = sup_kept[1]
            floor_ = sup_level * (1 - 2 * tol)
            if _max_consecutive(closes[anchor:] < floor_) >= BREAKOUT_CONSEC \
                    and sup_kept[-1] <= anchor:
                sup_level, sup_touches = np.nan, 0

        return {"res": res_level, "res_touches": res_touches,
                "sup": sup_level, "sup_touches": sup_touches, "tol": tol}

    def _mark_touch_bars(self, w_df: pd.DataFrame, highs: np.ndarray,
                         lows: np.ndarray, n: int) -> None:
        """Annotate touch bars for the FINAL window (chart overlay)."""
        w_df["Is_Res_Touch"] = False
        w_df["Is_Sup_Touch"] = False
        k = min(int(w_df["Base_Weeks"].iloc[-1]), MAX_CLUSTER_WEEKS)
        if k < 6 or n - 1 - k < 0:
            return
        last = w_df.iloc[-1]
        res_level, sup_level = last["Resistance"], last["Support"]
        win_start = (n - 1) - k
        w_highs = highs[win_start:n - 1]
        w_lows = lows[win_start:n - 1]
        tol = self._effective_tol(w_highs, w_lows,
                                  w_df["Close"].values[win_start:n - 1])
        res_col = w_df.columns.get_loc("Is_Res_Touch")
        sup_col = w_df.columns.get_loc("Is_Sup_Touch")
        if np.isfinite(res_level) and res_level > 0:
            pk_i, pk_v = self._swing_points(w_highs, "peaks")
            for local_i, v in zip(pk_i, pk_v):
                if abs(v - res_level) / res_level <= tol:
                    w_df.iat[win_start + int(local_i), res_col] = True
        if np.isfinite(sup_level) and sup_level > 0:
            tr_i, tr_v = self._swing_points(w_lows, "troughs")
            for local_i, v in zip(tr_i, tr_v):
                if abs(v - sup_level) / sup_level <= tol:
                    w_df.iat[win_start + int(local_i), sup_col] = True

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

        # ---- TIER 1: stabilization + fallback range -----------------------
        spans = self._stabilization(closes)
        fb_top, fb_bot = self._range_bounds(highs, lows, spans)
        w_df["Base_Weeks"] = spans  # name kept: plotter/report read Base_Weeks

        # ---- TIER 2: cluster levels (display refinement) ------------------
        cl = {c: np.full(n, np.nan) for c in
              ("c_res", "c_sup", "c_res_t", "c_sup_t", "tol")}
        for i in range(n):
            k = min(int(spans[i]), MAX_CLUSTER_WEEKS)
            if k < 6 or i - k < 0:
                continue
            m = self._analyse_window(highs[i - k:i],
                                     lows[i - k:i],
                                     closes[i - k:i])
            cl["c_res"][i] = m["res"]
            cl["c_sup"][i] = m["sup"]
            cl["c_res_t"][i] = m["res_touches"]
            cl["c_sup_t"][i] = m["sup_touches"]
            cl["tol"][i] = m["tol"]

        # Effective levels: the cluster level refines the fallback when it is
        # coherent with the current price and range; otherwise the range
        # bounds stand in. Every stabilized bar therefore HAS levels.
        tol_arr = np.where(np.isfinite(cl["tol"]), cl["tol"], TOUCH_TOL_MIN)
        use_res = (np.isfinite(cl["c_res"]) & (cl["c_res"] >= closes) &
                   np.isfinite(fb_top) & (cl["c_res"] <= fb_top * 1.10))
        use_sup = (np.isfinite(cl["c_sup"]) & (cl["c_sup"] <= closes) &
                   np.isfinite(fb_bot) & (cl["c_sup"] >= fb_bot * 0.90))
        res_eff = np.where(use_res, cl["c_res"], fb_top)
        sup_eff = np.where(use_sup, cl["c_sup"], fb_bot)

        half = tol_arr / 2
        w_df["Touch_Tol_Pct"] = tol_arr * 100
        w_df["Resistance"] = res_eff
        w_df["Support"] = sup_eff
        w_df["Res_Zone_Top"] = res_eff * (1 + half)
        w_df["Res_Zone_Bot"] = res_eff * (1 - half)
        w_df["Sup_Zone_Top"] = sup_eff * (1 + half)
        w_df["Sup_Zone_Bot"] = sup_eff * (1 - half)
        w_df["Res_Touches"] = np.where(np.isfinite(cl["c_res_t"]), cl["c_res_t"], 0)
        w_df["Sup_Touches"] = np.where(np.isfinite(cl["c_sup_t"]), cl["c_sup_t"], 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            w_df["Range_Width_Pct"] = (res_eff - sup_eff) / sup_eff * 100
        w_df["Distance_to_Breakout"] = (res_eff - closes) / closes * 100

        self._mark_touch_bars(w_df, highs, lows, n)

        # ---- prior decline (fail-closed when unobservable) ----------------
        # Computed BEFORE the stage classification, which consumes it: this is
        # the evidence that separates a Stage-1 base from a Stage-3 top, and
        # this strategy is the only module in the framework that can measure
        # it (it needs the stabilization span and the OHLC highs).
        w_df["Prior_Decline_Pct"] = self._prior_decline(highs, res_eff, spans) * 100

        # ---- 30W MA, slope and stage -- ALL from src/stages.py -------------
        # One call produces the MA the gates use AND the stage label. There is
        # no second definition of either anywhere in the framework.
        st = classify_stages(w_df["Close"],
                             prior_decline_pct=w_df["Prior_Decline_Pct"])
        w_df["SMA_30W"] = st["ma"]
        # The 10W (= 50-day) MA. Not a gate here -- Stage-1 detection must not
        # use a fast average, because price crosses it constantly inside a
        # base. It exists so the plotter draws the same warning line the stage
        # transitions are judged on.
        w_df["SMA_10W"] = st["ma_fast"]
        w_df["SMA_Slope"] = w_df["SMA_30W"].diff(MA_SLOPE_WEEKS)  # display only
        w_df["SMA_Slope_Pct"] = st["slope_pct"]
        # Near zero => the Stage 1 / Stage 3 label rests on noise. Surfaced so
        # the chart can say so instead of asserting a coin flip.
        w_df["Stage_Transition_Margin_Pct"] = st["transition_margin_pct"]

        # ---- volume (columns only; never a gate) --------------------------
        w_df["Vol_Avg_Prev_4W"] = w_df["Volume"].shift(1).rolling(VOL_AVG_WEEKS).mean()
        w_df["Vol_Spike_2x"] = np.where(
            w_df["Volume"] >= VOL_BREAKOUT_MULT * w_df["Vol_Avg_Prev_4W"], 1, 0)
        w_df["Vol_Avg_Curr_4W"] = w_df["Volume"].rolling(VOL_AVG_WEEKS).mean()
        w_df["Vol_Baseline"] = w_df["Volume"].shift(VOL_AVG_WEEKS).rolling(12).mean()
        w_df["Vol_4W_Expansion"] = np.where(
            w_df["Vol_Avg_Curr_4W"] >= VOL_BREAKOUT_MULT * w_df["Vol_Baseline"], 1, 0)
        w_df["Vol_vs_Avg"] = (w_df["Volume"] / w_df["Vol_Avg_Prev_4W"]).round(2)
        w_df["Vol_Base_Avg"] = w_df["Volume"].rolling(26).mean()
        dry = w_df["Vol_Avg_Curr_4W"] < w_df["Vol_Base_Avg"]
        w_df["Vol_Dryup"] = np.where(
            w_df["Vol_Avg_Curr_4W"].notna() & w_df["Vol_Base_Avg"].notna() & dry, 1, 0)
        vol_slope = _rolling_lsq_slope(w_df["Volume"].astype(float), VOL_BUILDUP_WEEKS)
        w_df["Vol_Buildup_8W"] = np.where(vol_slope > 0, 1, 0)

        # ---- breakout tracking (sustained breaks only) --------------------
        w_df["Weeks_Since_Breakout"] = self._weeks_since_breakout(
            closes, w_df["Res_Zone_Top"].values, spans)

        # ---- relative strength (display only) -----------------------------
        w_df["Mansfield_RS"] = self._mansfield(w_df)
        w_df["RS_Slope_10W"] = _rolling_lsq_slope(w_df["Mansfield_RS"], RS_SLOPE_WEEKS)
        if self._benchmark_weekly is None and not self._warned_no_benchmark:
            log.warning(
                "WeinsteinSetup: no benchmark injected -> Mansfield RS and "
                "RS_Slope_10W are NaN in the output table. Signals are "
                "unaffected (RS never gated anything). Call set_benchmark() "
                "(the Screener does this per market).")
            self._warned_no_benchmark = True

        # ============================ SIGNAL ============================
        ma = w_df["SMA_30W"]
        slope = w_df["SMA_Slope_Pct"]
        close_s = w_df["Close"]

        # 1) Stabilized long enough, in a sane, resolvable range.
        cond_base_len = w_df["Base_Weeks"] >= MIN_STAB_WEEKS
        cond_width = (w_df["Range_Width_Pct"] <= MAX_LEVEL_WIDTH * 100) & \
                     (w_df["Range_Width_Pct"] >= 3.0 * w_df["Touch_Tol_Pct"])

        # 2) A considerable decline demonstrably preceded the base.
        cond_prior_decline = w_df["Prior_Decline_Pct"] >= MIN_PRIOR_DECLINE * 100

        # 3) Price inside the range, no sustained breakout fresher than the
        #    base itself.
        cond_in_base = (close_s >= w_df["Sup_Zone_Bot"]) & \
                       (close_s <= w_df["Res_Zone_Top"])
        cond_not_broken = w_df["Weeks_Since_Breakout"].isna() | \
                          (w_df["Weeks_Since_Breakout"] > MIN_STAB_WEEKS)

        # 4) MA shape: flat (mature base) OR falling-but-decelerating with
        #    price converged onto the MA (early base) -- never clearly rising.
        flat = slope.between(MA_FLAT_BAND[0], MA_FLAT_BAND[1])
        decel = slope >= slope.shift(MA_DECEL_LOOKBACK) + MA_DECEL_MARGIN
        near_ma = close_s >= ma * (1 - MAX_BELOW_MA)
        early = (slope >= -MAX_MA_DECLINE_PCT) & decel
        cond_ma_shape = near_ma & (slope <= MAX_MA_RISE_PCT) & (flat | early)

        signal = (cond_base_len & cond_width & cond_prior_decline &
                  cond_in_base & cond_not_broken & cond_ma_shape)

        self._cond_cols = {
            # levels_found: DIAGNOSTIC ONLY -- did the swing clusters resolve?
            # It no longer gates anything; when False, Resistance/Support are
            # the stabilization-range bounds instead of cluster levels.
            "levels_found": pd.Series(np.isfinite(cl["c_res"]) &
                                      np.isfinite(cl["c_sup"]),
                                      index=w_df.index),
            "base_len": cond_base_len,
            "width": cond_width,
            "prior_decline": cond_prior_decline,
            "in_base": cond_in_base,
            "not_broken": cond_not_broken,
            "ma_shape": cond_ma_shape,
        }
        for cname, cseries in self._cond_cols.items():
            w_df[f"Cond_{cname}"] = cseries.fillna(False).astype(bool)

        signal_arr = signal.fillna(False).values
        w_df["Signal"] = np.where(signal_arr, 1, 0)

        # ---- stage label ---------------------------------------------------
        # The classification itself came from src/stages.py above. The ONLY
        # thing applied here is strategy policy: a signalling bar IS the
        # Stage-1 certification (the gates ARE the Stage-1 test), so the label
        # must agree -- this covers the late-Stage-4 look of early bases and
        # the fact that Stage 2 cannot precede its own breakout. That override
        # is deliberately NOT in stages.py: it is this strategy's opinion, not
        # a property of the stage definition, and the sector monitor must not
        # inherit it.
        stage_num = np.where(signal_arr, int(Stage.ONE), st["stage"].to_numpy())
        w_df["Stage_Num"] = stage_num
        w_df["Stage"] = [Stage(int(v)).short for v in stage_num]

        return w_df

    # ------------------------------------------------------------------
    # METRIC HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _prior_decline(highs: np.ndarray, res_eff: np.ndarray,
                       spans: np.ndarray) -> np.ndarray:
        """Fractional drop from the pre-base peak (within
        DECLINE_WINDOW_WEEKS before the stabilization start) to the range
        top. NaN (fail-closed) when the decline is not observable."""
        n = len(res_eff)
        out = np.full(n, np.nan)
        for i in range(n):
            if not (np.isfinite(res_eff[i]) and res_eff[i] > 0):
                continue
            base_start = i - int(spans[i])
            if base_start <= 0:
                continue
            w0 = max(0, base_start - DECLINE_WINDOW_WEEKS)
            seg = highs[w0:base_start]
            if len(seg) == 0 or not np.any(np.isfinite(seg)):
                continue
            pre_peak = float(np.nanmax(seg))
            if pre_peak <= 0:
                continue
            out[i] = (pre_peak - res_eff[i]) / pre_peak
        return out

    @staticmethod
    def _weeks_since_breakout(closes: np.ndarray, res_top: np.ndarray,
                              spans: np.ndarray) -> np.ndarray:
        """Weeks since the last SUSTAINED break (BREAKOUT_CONSEC consecutive
        closes above the zone top) of an established base. Anchored at the
        first bar of the run, and only when the bar before the run was still
        below its own level -- so the event is dated once, at the actual
        transition, instead of re-dating forward every week of a rally."""
        n = len(closes)
        out = np.full(n, np.nan)
        min_established = max(4, MIN_STAB_WEEKS // 2)
        last_breakout = None
        k = BREAKOUT_CONSEC
        for i in range(k, n):
            run_ok = all(np.isfinite(res_top[i - j]) and
                         closes[i - j] > res_top[i - j] for j in range(k))
            before = i - k
            was_below = (np.isfinite(res_top[before]) and
                         closes[before] <= res_top[before])
            if run_ok and was_below and spans[before] >= min_established:
                last_breakout = i - (k - 1)
            if last_breakout is not None:
                out[i] = i - last_breakout
        return out

    def _mansfield(self, w_df: pd.DataFrame) -> pd.Series:
        if self._benchmark_weekly is None:
            return pd.Series(np.nan, index=w_df.index)
        from src.benchmarks import mansfield_rs
        mrs = mansfield_rs(w_df["Close"], self._benchmark_weekly,
                           n=RS_PERIOD_WEEKS)
        return mrs.reindex(w_df.index)
