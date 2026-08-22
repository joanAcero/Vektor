"""
stages.py
---------
THE Weinstein stage classifier. One definition, one implementation, one set
of constants. Every consumer in VEKTOR imports from here:

    strategies/weinstein_setup.py   Stage column + the MA/slope its gates use
    src/rotation.py                 sector_stage for each SPDR ETF
    notifications/stage_chart.py    the coloured bands on every chart
    web/index.html                  badge colours (mirrors STAGE_COLOR)

There used to be two implementations. They agreed on Stage 2 and Stage 4 and
disagreed on the Stage 1 / Stage 3 split -- the hard part -- so the same chart
could be labelled differently depending on which module asked. Do not add a
third: if a stage rule needs to change, it changes here and propagates.

DEPENDENCIES: numpy and pandas only. No matplotlib, no strategy imports, no
data loader. This module is a primitive; everything else depends on it and it
depends on nothing in the project. Keep it that way -- the moment it imports a
strategy, the strategy can no longer import it.

THE RULES
=========
Two moving averages on WEEKLY closes, both of them Weinstein's:

    MA30  the 30-week average. The DECISION line. Its slope over 5 weeks,
          as a percentage of its own level, is the only slope used anywhere.
    MA10  the 10-week average -- which is the 50-day average; Weinstein's
          own equivalence, so no daily data and no mixed timeframes. The
          WARNING line: a slice below it is the first sign an advance is over.

Every stage change is a transition with its own trigger. Each fires on the
bar its trigger is true -- no confirmation window, no lag:

    from  to   trigger
    ---------------------------------------------------------------------
      1    2   close > MA30 and MA30 slope > +0.5%
      1    4   close < MA30 and MA30 slope < -0.5%
      2    4   close < MA30 and MA30 slope < -0.5%   (checked before 2->3)
      2    3   close < MA10
      3    4   close < MA30
      3    2   close > MA10 and MA30 slope > +0.5%
      4    2   close > MA30 and MA30 slope > +0.5%   (checked before 4->1)
      4    1   close > MA10 and MA30 slope > -0.5%

Read the shape rather than the rows: THE FAST LINE GOVERNS THE WARNINGS
(2->3, 4->1) AND THE SLOW LINE GOVERNS THE COMMITMENTS (3->4, 1->2, 2->4).
Entering a trend additionally needs a slope condition that leaving one does
not -- exit fast, enter slow, which is Weinstein's disposition throughout.

Two asymmetries worth defending, because they look like inconsistencies:

  * 3->4 has no slope guard, but 1->4 and 2->4 do. Out of Stage 3 the MA30
    sits above or beside a price that has already turned, so breaking it means
    something on its own. Out of Stage 1 price oscillates ACROSS the MA30 --
    that is what a base is -- and out of Stage 2 a break of a still-RISING
    MA30 is a pullback, not a decline. Without that guard an ordinary pullback
    in an advance was labelled Stage 4, and the recovery afterwards was then
    labelled Stage 1, because Stage 1 was the only exit from Stage 4.
  * 4->2 exists so a violent recovery is not forced through Stage 1. It uses
    the same trigger as 1->2 and is checked first: reclaiming a RISING 30-week
    MA is an advance, whatever the machine was calling the chart last week.

KNOWN LIMITATION -- READ THIS BEFORE TRUSTING A LABEL
=====================================================
Every trigger is a bare MA crossing, and in a lateral market price sits ON the
moving averages and crosses them every few weeks. On a sideways chart the
labels therefore change constantly, and each individual change is correct by
the rules while the sequence as a whole is not informative. Measured on
synthetic lateral series: ~64 stage changes per 500 weeks, against ~20 for a
trending series.

Guards do not fix this; the fix is a threshold, not a rule. Requiring a break
to clear the MA by a volatility-scaled band (about 2.5x the median absolute
weekly return, clipped) roughly halves the churn AND improves the labels in
trending markets, because it changes what counts as a crossing rather than
delaying when a crossing is acted on. That is not implemented here.

Because price must break MA10 before it breaks MA30, the ordinary sequence
2 -> 3 -> 4 falls out of the MA hierarchy on its own. An earlier version of
this module enforced cycle legality with an explicit state variable; that
machinery is gone. The guarantee now comes from the model rather than from a
rule bolted on beside it, which is why base -> top is still unreachable:
Stage 1 has no transition to Stage 3 and Stage 3 has none to Stage 1.

NO CONFIRMATION WINDOW
======================
A transition fires immediately. This is a decision, not an oversight: the
break of the 30-week MA is Weinstein's exit signal, and a label that arrives
three weeks after the exit is worse than useless. On weekly bars a single
close is a week of evidence.

The cost is churn. In a healthy Stage 2, one week's close under the 10-week
MA flips the label to Stage 3 and the next week can flip it back, so 2 <-> 3
oscillation is normal and expected. Note what this makes Stage 3 mean: not
"distribution top" but "the advance is not confirmed right now". That is
Weinstein's own sell trigger, and it is a deliberate change of meaning from
the book's usage. Anything downstream that treats Stage 3 as a durable
condition -- ranking, alerting, position sizing -- has to tolerate a label
that changes week to week.

SEEDING
=======
The first classifiable bar has no predecessor to transition from. It is
seeded: a confirmed trend if one is visible, otherwise Stage 1 or Stage 3
from `prior_decline_pct` (the book's test, when the caller can measure it)
or from `ma[t] < ma[t-FALLBACK_WEEKS]`. The seed self-corrects at the first
real transition, so it matters far less than it did when the 1-vs-3 split was
re-derived on every bar.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd

__all__ = [
    "Stage", "STAGE_COLOR", "hunt_rank",
    "SMA_WEEKS", "MA_FAST_WEEKS", "MA_SLOPE_WEEKS", "TREND_SLOPE_PCT",
    "FALLBACK_WEEKS", "MIN_DECLINE_PCT",
    "weekly_ma", "weekly_ma_fast", "ma_slope_pct", "classify",
    "classify_last", "stage_spans", "weeks_in_stage",
]


# ======================================================================
# CONSTANTS
# ======================================================================
# BOOK: Weinstein states these.
SMA_WEEKS = 30          # "the 30-week MA is ideal for investors"
MA_FAST_WEEKS = 10      # the 50-day MA. Weinstein's own equivalence: 10 weeks
                        # IS 50 days, so this needs no daily data and mixes no
                        # timeframes. He treats a slice below it as the first
                        # warning that an advance is over.
MA_SLOPE_WEEKS = 5      # slope over ~1 month of weekly bars

# CALIBRATION: not in the book. Changing these changes every stage label in
# the framework, which is the point.
TREND_SLOPE_PCT = 0.5   # |slope| beyond this + the right price side => 2 or 4.
                        # 0.5% over 5 weeks is ~5%/yr: a narrow "flat" band.
FALLBACK_WEEKS = 52     # MA-context lookback, used ONLY to seed the label at
                        # the very start of a series, before any transition has
                        # fired. It used to decide 1-vs-3 on every bar, which is
                        # what produced illegal base->top jumps.
# There is deliberately NO confirmation window. A transition fires on the bar
# its trigger is true. An earlier version required three consecutive weeks,
# which dated every turn three weeks late -- unacceptable on the exit
# transitions, where the break of the 30-week MA IS the signal. On weekly bars
# a single close is already a week of evidence, not a tick.
MIN_DECLINE_PCT = 15.0  # "a considerable decline" precedes a Stage 1.
                        # strategies/weinstein_setup.py derives its own
                        # MIN_PRIOR_DECLINE from this so the gate and the
                        # label can never disagree.


class Stage(IntEnum):
    """Weinstein's four stages, plus UNKNOWN for insufficient history.

    Integer-valued so it can live in a numpy array and a DataFrame column
    without object dtype. Each consumer formats via the properties below
    rather than hard-coding its own strings.
    """
    UNKNOWN = 0
    ONE = 1     # base / accumulation
    TWO = 2     # advance
    THREE = 3   # top / distribution
    FOUR = 4    # decline

    @property
    def slug(self) -> str:
        """JSON / API form: 'stage1'..'stage4', 'unknown'."""
        return "unknown" if self is Stage.UNKNOWN else f"stage{int(self)}"

    @property
    def short(self) -> str:
        """Table / CSV / Telegram form: '1'..'4', '-'."""
        return "-" if self is Stage.UNKNOWN else str(int(self))

    @property
    def label(self) -> str:
        """Human form for charts and badges."""
        return {
            Stage.UNKNOWN: "unknown",
            Stage.ONE: "1 · base",
            Stage.TWO: "2 · advance",
            Stage.THREE: "3 · top",
            Stage.FOUR: "4 · decline",
        }[self]

    @property
    def color(self) -> str:
        return STAGE_COLOR[self]

    @classmethod
    def coerce(cls, value) -> "Stage":
        """Accept a Stage, an int, or a slug/short string and return a Stage.

        Every public helper funnels through this. Callers hold stages in
        whatever form their layer uses -- an int in a DataFrame column, a
        'stage3' string off the JSON API, a Stage in Python -- and none of
        them should have to know which. Raises ValueError on anything else,
        loudly, at the point of the mistake.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, (int, np.integer)):
            return cls(int(value))
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("unknown", "-", ""):
                return cls.UNKNOWN
            return cls(int(v.removeprefix("stage")))
        raise ValueError(f"Cannot coerce {value!r} to a Stage")

    @classmethod
    def from_slug(cls, slug: str) -> "Stage":
        """Deprecated alias for coerce(). Kept so existing call sites keep
        working; new code should use coerce()."""
        return cls.coerce(slug)


# The one palette. web/index.html mirrors these four hex values as CSS
# variables -- that is the single duplicate in the framework and it is
# deliberate: an endpoint to serve four colours costs more than it saves.
#
# Tuned for a WHITE chart background (see notifications/stage_chart.py). Two
# channels, not one, because hue alone is unreliable for red-green colour
# deficiency: the hues are far apart AND the luminances are spread, so the
# pair that collapses on one channel is still separable on the other.
#   decline (.09) < base (.17) < advance (.26) < top (.46)
STAGE_COLOR: dict[Stage, str] = {
    Stage.UNKNOWN: "#9aa0a6",
    Stage.ONE:     "#0072b2",   # blue         -- basing
    Stage.TWO:     "#009e73",   # green        -- advancing
    Stage.THREE:   "#f0a202",   # amber        -- topping
    Stage.FOUR:    "#9e1b32",   # dark crimson -- declining
}


# Where to hunt Stage 1->2 stock breakouts, in order of preference. Stage 2
# first (the sector tailwind is already blowing), then Stage 1 (basing, so its
# leaders may break out first), then Stage 3, then Stage 4.
#
# PRIVATE, and reached only through hunt_rank(). An earlier version exported
# this dict directly and a same-named but STRING-keyed dict left behind in
# src/rotation.py shadowed the import (a later module-level binding wins over
# an earlier `from ... import`). Stage.THREE is an IntEnum, so it hashes and
# compares as 3 and can never match a "stage3" key -- KeyError at runtime,
# long after the real mistake. A function is harder to shadow by accident,
# fails at the point of the mistake, and normalises whatever it is handed.
_HUNT_RANK: dict[Stage, int] = {
    Stage.TWO: 0, Stage.ONE: 1, Stage.THREE: 2, Stage.FOUR: 3, Stage.UNKNOWN: 4,
}


def hunt_rank(stage) -> int:
    """Hunting priority for a stage: 0 is the best place to look, 4 the worst.

    Accepts a Stage, an int, or a slug string -- see Stage.coerce.
    """
    return _HUNT_RANK[Stage.coerce(stage)]


# ======================================================================
# PRIMITIVES
# ======================================================================
def weekly_ma(weekly_close: pd.Series, window: int = SMA_WEEKS) -> pd.Series:
    """The 30-week simple moving average of weekly closes."""
    return weekly_close.rolling(window=window).mean()


def weekly_ma_fast(weekly_close: pd.Series,
                   window: int = MA_FAST_WEEKS) -> pd.Series:
    """The 10-week (= 50-day) simple moving average of weekly closes."""
    return weekly_close.rolling(window=window).mean()


def ma_slope_pct(ma: pd.Series, weeks: int = MA_SLOPE_WEEKS) -> pd.Series:
    """MA slope over `weeks` bars, as a percentage of the MA's own level.

    Percentage rather than absolute so a EUR 8 stock and a EUR 800 stock are
    judged on the same scale.
    """
    return (ma.diff(weeks) / ma) * 100.0


# ======================================================================
# CLASSIFIER
# ======================================================================
def classify(weekly_close: pd.Series, *,
             prior_decline_pct: pd.Series | None = None,
             ma: pd.Series | None = None,
             slope_pct: pd.Series | None = None,
             ma_fast: pd.Series | None = None) -> pd.DataFrame:
    """
    Classify every bar of a WEEKLY close series. See the module docstring for
    the transition table.

    Args:
        weekly_close: weekly closes, DatetimeIndex, ascending.
        prior_decline_pct: optional, same index. Percentage drop from the
            pre-base peak, NaN where unobservable. Used only to SEED the label
            before any transition has fired. Only the Weinstein strategy can
            measure it; everyone else passes None.
        ma, slope_pct, ma_fast: optional precomputed series, to avoid
            recomputing when the caller already holds them.

    Returns a DataFrame on the same index with columns:
        stage        int Stage value
        stage_raw    the memoryless reading (Stage 2/4 where the trend
                     conditions hold outright, else the seed). Kept for
                     diagnosis: where stage and stage_raw disagree is exactly
                     where the transition machine overrode the local picture.
        ma           the 30W MA, the decision line
        ma_fast      the 10W (= 50-day) MA, the warning line
        slope_pct    the 30W MA's 5-week slope, % of its own level
        transition_margin_pct  the seed diagnostic, decisive only at the start
        decline_used True where prior_decline_pct was available
    """
    close = pd.Series(weekly_close).astype(float)
    if ma is None:
        ma = weekly_ma(close)
    if slope_pct is None:
        slope_pct = ma_slope_pct(ma)
    if ma_fast is None:
        ma_fast = weekly_ma_fast(close)

    valid = (ma.notna() & slope_pct.notna() & ma_fast.notna()).to_numpy()
    above30 = (close > ma).to_numpy()
    below30 = (close < ma).to_numpy()
    above10 = (close > ma_fast).to_numpy()
    below10 = (close < ma_fast).to_numpy()
    rising = (slope_pct > TREND_SLOPE_PCT).to_numpy()
    falling = (slope_pct < -TREND_SLOPE_PCT).to_numpy()
    not_falling = (slope_pct > -TREND_SLOPE_PCT).to_numpy()

    # One boolean per trigger, evaluated on the bar itself.
    t_to2_slow = above30 & rising          # 1->2, 4->2
    t_to2_fast = above10 & rising          # 3->2
    t_to4_slow = below30 & falling         # 1->4, 2->4
    t_to4 = below30                        # 3->4
    t_to3 = below10                        # 2->3
    t_to1 = above10 & not_falling          # 4->1

    # Seed for a series whose first bars precede any transition.
    ma_ref = ma.shift(FALLBACK_WEEKS)
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = (ma / ma_ref - 1.0) * 100.0
    margin = margin.replace([np.inf, -np.inf], np.nan)
    seed_is_one = (margin < 0).to_numpy()
    if prior_decline_pct is not None:
        decline = pd.Series(prior_decline_pct, index=close.index).astype(float)
        have_decline = decline.notna().to_numpy()
        seed_is_one = np.where(have_decline,
                               (decline >= MIN_DECLINE_PCT).to_numpy(),
                               seed_is_one)
    else:
        have_decline = np.zeros(len(close), dtype=bool)

    stage_raw = np.select(
        [~valid, above30 & rising, below30 & falling, seed_is_one],
        [int(Stage.UNKNOWN), int(Stage.TWO), int(Stage.FOUR), int(Stage.ONE)],
        default=int(Stage.THREE)).astype(int)

    n = len(close)
    stage = np.full(n, int(Stage.UNKNOWN), dtype=int)
    current = int(Stage.UNKNOWN)

    for i in range(n):
        if not valid[i]:
            stage[i] = int(Stage.UNKNOWN)
            current = int(Stage.UNKNOWN)
            continue

        if current == int(Stage.UNKNOWN):
            # Nothing to transition from: seed it.
            if above30[i] and rising[i]:
                current = int(Stage.TWO)
            elif below30[i] and falling[i]:
                current = int(Stage.FOUR)
            else:
                current = int(Stage.ONE if seed_is_one[i] else Stage.THREE)
            stage[i] = current
            continue

        if current == int(Stage.ONE):
            if t_to2_slow[i]:
                current = int(Stage.TWO)
            elif t_to4_slow[i]:
                current = int(Stage.FOUR)
        elif current == int(Stage.TWO):
            # Stage 4 checked first, but it needs the MA30 to be FALLING: a
            # break of a rising 30-week MA out of an advance is a pullback.
            if t_to4_slow[i]:
                current = int(Stage.FOUR)
            elif t_to3[i]:
                current = int(Stage.THREE)
        elif current == int(Stage.THREE):
            if t_to4[i]:
                current = int(Stage.FOUR)
            elif t_to2_fast[i]:
                current = int(Stage.TWO)
        elif current == int(Stage.FOUR):
            # Stage 2 checked first: a recovery that reclaims a rising MA30 is
            # an advance and must not be routed through "base".
            if t_to2_slow[i]:
                current = int(Stage.TWO)
            elif t_to1[i]:
                current = int(Stage.ONE)
        stage[i] = current

    return pd.DataFrame(
        {
            "stage": stage,
            "stage_raw": stage_raw,
            "ma": ma,
            "ma_fast": ma_fast,
            "slope_pct": slope_pct,
            "transition_margin_pct": margin,
            "decline_used": have_decline,
        },
        index=close.index,
    )


def classify_last(weekly_close: pd.Series, **kwargs) -> Stage:
    """The stage of the most recent bar. Convenience over `classify`; there is
    no separate scalar code path, so a scalar answer can never drift from the
    series answer."""
    if weekly_close is None or len(weekly_close) == 0:
        return Stage.UNKNOWN
    out = classify(weekly_close, **kwargs)
    if out.empty:
        return Stage.UNKNOWN
    return Stage(int(out["stage"].iloc[-1]))


def weeks_in_stage(stage_nums) -> np.ndarray:
    """Consecutive weeks the stage at each bar has been unchanged, 1-based.

    Lives here rather than in a strategy so "how long has this been Stage 1"
    has one answer everywhere -- the screener, the sector monitor and the
    charts all count it the same way.
    """
    arr = np.asarray(stage_nums, dtype=int)
    out = np.zeros(len(arr), dtype=int)
    run = 0
    prev = None
    for i, v in enumerate(arr):
        run = run + 1 if v == prev else 1
        prev = v
        out[i] = run
    return out


def stage_spans(stage_nums) -> list[tuple[int, int, Stage]]:
    """Collapse a per-bar stage array into contiguous runs.

    Returns [(start_idx, end_idx_inclusive, Stage), ...]. Used by the chart
    shader so a five-year weekly chart draws ~10 rectangles instead of 260,
    and so a run's boundaries are computed once rather than per renderer.
    """
    arr = np.asarray(stage_nums, dtype=int)
    if arr.size == 0:
        return []
    edges = np.flatnonzero(np.diff(arr)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges - 1, [arr.size - 1]))
    return [(int(s), int(e), Stage(int(arr[s]))) for s, e in zip(starts, ends)]
