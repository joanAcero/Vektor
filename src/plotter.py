"""
plotter.py
----------
Generic, strategy-agnostic fallback chart: weekly OHLC candles + volume, and a
marker on the final bar when the signal fires. A strategy can override
Strategy.plot() to draw something richer (as Weinstein does); the runner uses
this only when a strategy declines to plot itself.

Kept deliberately small: any strategy gets a usable chart for free, with no
hard-coded references to Weinstein columns.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # headless: required for CI / cron with no display
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

UP, DN = "#26a641", "#e03131"


def _candles(ax, df, width=5):
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    x = mdates.date2num(df.index.to_pydatetime())
    for i in range(len(df)):
        col = UP if c[i] >= o[i] else DN
        ax.plot([x[i], x[i]], [l[i], h[i]], color=col, linewidth=0.9, zorder=3)
        ax.add_patch(mpatches.Rectangle(
            (x[i] - width / 2, min(o[i], c[i])), width, abs(c[i] - o[i]) or 1e-9,
            facecolor=col, edgecolor=col, linewidth=0.4, zorder=4))


def plot_generic(ticker: str, df: pd.DataFrame, out_path: str,
                 signal_column: str = "Signal") -> bool:
    if df is None or df.empty or not {"Open", "High", "Low", "Close"} <= set(df.columns):
        log.warning("plot_generic: insufficient columns for %s", ticker)
        return False

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    has_vol = "Volume" in df.columns
    if has_vol:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax2 = None

    fig.suptitle(f"{ticker}", fontsize=14, fontweight="bold")
    _candles(ax1, df)

    # 30-period moving average. The frames passed here are weekly (strategies
    # resample to W-FRI), so this is the 30-week MA. Reuse a column the strategy
    # already computed if present; otherwise derive it from Close.
    ma = None
    for col in ("SMA_30W", "SMA", "MA30"):
        if col in df.columns:
            ma = df[col]
            break
    if ma is None:
        ma = df["Close"].rolling(window=30).mean()
    ax1.plot(df.index, ma, color="#e07b00", linewidth=1.6, linestyle="--",
             alpha=0.9, label="MA30 (30W)", zorder=6)

    # --- Detected base overlay (diagnostic) ---------------------------------
    # Draw what the algorithm detected as the base on the LAST bar: the support
    # and resistance levels, the shaded range band, and a rectangle spanning the
    # most recent Base_Weeks. This is the key view for diagnosing why a setup
    # was or wasn't detected.
    last = df.iloc[-1]
    res = last.get("Resistance")
    sup = last.get("Support")
    if res is not None and sup is not None and pd.notna(res) and pd.notna(sup):
        # Shaded range band between support and resistance.
        ax1.axhspan(sup, res, color="#3b82f6", alpha=0.08, zorder=1)
        ax1.axhline(res, color="#ef4444", linewidth=1.4, linestyle="-",
                    alpha=0.85, zorder=5, label=f"Resistance {res:.2f}")
        ax1.axhline(sup, color="#22c55e", linewidth=1.4, linestyle="-",
                    alpha=0.85, zorder=5, label=f"Support {sup:.2f}")

        # Rectangle marking the base window (last Base_Weeks bars).
        bw = last.get("Base_Weeks")
        if bw is not None and pd.notna(bw) and bw >= 1:
            bw = int(bw)
            start_idx = max(0, len(df) - bw)
            x0 = mdates.date2num(df.index[start_idx].to_pydatetime())
            x1 = mdates.date2num(df.index[-1].to_pydatetime())
            ax1.add_patch(mpatches.Rectangle(
                (x0, sup), x1 - x0, res - sup,
                fill=False, edgecolor="#3b82f6", linewidth=1.6,
                linestyle="--", zorder=6))
            ax1.text(x0, res, f" base: {bw}w", color="#3b82f6", fontsize=9,
                     va="bottom", ha="left", zorder=7, fontweight="bold")

        # Annotate the key diagnostic metrics in a corner box.
        bits = []
        for label, key, fmt in [
            ("Base", "Base_Weeks", "{:.0f}w"),
            ("Width/Decline", None, None),
            ("Dist2BO", "Distance_to_Breakout", "{:.1f}%"),
            ("PriorDecline", "Prior_Decline_Pct", "{:.0f}%"),
            ("ResTouch", "Res_Touches", "{:.0f}"),
            ("SupTouch", "Sup_Touches", "{:.0f}"),
        ]:
            if key is None:  # computed field
                rw = last.get("Range_Width_Pct"); pd_ = last.get("Prior_Decline_Pct")
                if pd.notna(rw) and pd.notna(pd_) and pd_:
                    bits.append(f"Width/Decline: {rw/pd_:.2f}")
                continue
            v = last.get(key)
            if v is not None and pd.notna(v):
                bits.append(f"{label}: {fmt.format(v)}")
        if bits:
            ax1.text(0.985, 0.97, "\n".join(bits), transform=ax1.transAxes,
                     ha="right", va="top", fontsize=8.5, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                               edgecolor="#cccccc", alpha=0.9), zorder=8)

    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

    if signal_column in df.columns and int(df[signal_column].iloc[-1]) != 0:
        x = mdates.date2num(df.index[-1].to_pydatetime())
        y = df["High"].iloc[-1] * 1.02
        ax1.annotate("", xy=(x, y * 1.03), xytext=(x, y),
                     arrowprops=dict(arrowstyle="-|>", color="#16a34a", lw=2.4))

    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle=":", alpha=0.5)

    if ax2 is not None:
        opens = df["Open"] if "Open" in df.columns else df["Close"].shift(1)
        colors = np.where(df["Close"] >= opens, "#86efac", "#fca5a5")
        ax2.bar(df.index, df["Volume"], color=colors, width=5)
        ax2.set_ylabel("Volume")
        ax2.grid(True, linestyle=":", alpha=0.5)
        target = ax2
    else:
        target = ax1

    target.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    target.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True
