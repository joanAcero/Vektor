import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import numpy as np


# ── PALETTE (light theme) ─────────────────────────────────────────────────────
BG_FIG    = "#ffffff"
BG_AX     = "#f8f9fa"
GRID_COL  = "#e0e0e0"
TICK_COL  = "#333333"
TEXT_COL  = "#222222"
SPINE_COL = "#cccccc"

CANDLE_UP   = "#26a641"
CANDLE_DN   = "#e03131"

SMA_COL     = "#e07b00"
RES_COL     = "#e03131"
SUP_COL     = "#26a641"
RES_TCH_COL = "#7c3aed"
SUP_TCH_COL = "#0891b2"
BASE_RECT   = "#1d4ed8"
BREAK_COL   = "#16a34a"
VOL_UP      = "#86efac"
VOL_DN      = "#fca5a5"
VOL_AVG_COL = "#e07b00"


def _draw_candlesticks(ax, df, width_days=5):
    """Draw weekly OHLC candlesticks. width_days controls body width."""
    col_o = df["Open"].values
    col_h = df["High"].values
    col_l = df["Low"].values
    col_c = df["Close"].values
    dates = mdates.date2num(df.index.to_pydatetime())
    half  = width_days / 2.0

    for i in range(len(df)):
        o, h, l, c = col_o[i], col_h[i], col_l[i], col_c[i]
        x        = dates[i]
        bullish  = c >= o
        col      = CANDLE_UP if bullish else CANDLE_DN

        # Wick
        ax.plot([x, x], [l, h], color=col, linewidth=0.9, zorder=3,
                solid_capstyle="round")

        # Body
        body_bot = min(o, c)
        body_h   = abs(c - o)
        rect = mpatches.FancyBboxPatch(
            (x - half, body_bot), width_days, body_h,
            boxstyle="square,pad=0",
            linewidth=0.4,
            edgecolor=col,
            facecolor=col,
            zorder=4,
        )
        ax.add_patch(rect)


def _draw_base_rectangle(ax, df, lookback_weeks, sup_bot, res_top):
    """Dashed blue rectangle framing the entire base period."""
    if len(df) <= lookback_weeks:
        return

    x0 = mdates.date2num(df.index[-lookback_weeks].to_pydatetime())
    x1 = mdates.date2num(df.index[-1].to_pydatetime())

    # Filled background
    rect = mpatches.FancyBboxPatch(
        (x0, sup_bot), x1 - x0, res_top - sup_bot,
        boxstyle="square,pad=0",
        linewidth=0,
        facecolor=BASE_RECT,
        alpha=0.05,
        zorder=2,
    )
    ax.add_patch(rect)

    # Dashed border (drawn as four line segments so linestyle works cleanly)
    for xs, ys in [
        ([x0, x1], [sup_bot, sup_bot]),
        ([x0, x1], [res_top, res_top]),
        ([x0, x0], [sup_bot, res_top]),
        ([x1, x1], [sup_bot, res_top]),
    ]:
        ax.plot(xs, ys, color=BASE_RECT, linewidth=1.8,
                linestyle="--", alpha=0.65, zorder=5)


def _draw_breakout_arrow(ax, df, res_zone_top):
    """Bold green upward arrow above the resistance zone when Signal == 1."""
    if "Signal" not in df.columns or int(df["Signal"].iloc[-1]) != 1:
        return

    x       = mdates.date2num(df.index[-1].to_pydatetime())
    y_base  = res_zone_top
    y_tip   = res_zone_top * 1.038

    ax.annotate(
        "",
        xy=(x, y_tip), xytext=(x, y_base),
        arrowprops=dict(
            arrowstyle="-|>",
            color=BREAK_COL,
            lw=2.6,
            mutation_scale=24,
        ),
        zorder=9,
    )
    ax.text(x, y_tip * 1.004, "  BREAKOUT",
            color=BREAK_COL, fontsize=8.5, fontweight="bold",
            va="bottom", zorder=9)


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────

def plot_weinstein_setup(ticker, w_df, lookback_weeks, filename):
    """
    Weinstein Setup chart — candlestick / light-theme edition.

    Panel 1 – Price
      · Weekly OHLC candlesticks
      · 30W SMA (orange dashed)
      · Resistance ZONE (red band + dashed midline)
      · Support ZONE    (green band + dashed midline)
      · Base rectangle  (blue dashed frame over the lookback window)
      · Breakout arrow  (bold green ↑, only when Signal = 1)
      · Resistance touch markers (purple ▼ above High)
      · Support touch markers    (teal ▲ below Low)
      · Info box

    Panel 2 – Volume
      · Coloured bars + 4-week average line
    """

    df = w_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # ── FIGURE ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor(BG_FIG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG_AX)
        ax.tick_params(colors=TICK_COL, labelsize=8)
        ax.yaxis.label.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COL)

    fig.suptitle(
        f"{ticker}  –  Weinstein Setup  (Base: {lookback_weeks}w)",
        fontsize=15, fontweight="bold", color=TEXT_COL, y=0.98,
    )

    # ── PRICE PANEL ───────────────────────────────────────────────────

    last    = df.iloc[-1]
    has_res = ("Res_Zone_Bot" in df.columns and pd.notna(last.get("Res_Zone_Bot")))
    has_sup = ("Sup_Zone_Bot" in df.columns and pd.notna(last.get("Sup_Zone_Bot")))

    res_bot_val = float(last["Res_Zone_Bot"]) if has_res else None
    res_top_val = float(last["Res_Zone_Top"]) if has_res else None
    res_mid_val = float(last["Resistance"])   if has_res else None
    sup_bot_val = float(last["Sup_Zone_Bot"]) if has_sup else None
    sup_top_val = float(last["Sup_Zone_Top"]) if has_sup else None
    sup_mid_val = float(last["Support"])      if has_sup else None

    # (A) Base rectangle — behind everything
    if has_res and has_sup:
        _draw_base_rectangle(ax1, df, lookback_weeks, sup_bot_val, res_top_val)

    # (B) Candlesticks
    _draw_candlesticks(ax1, df, width_days=5)

    # (C) 30W SMA
    if "SMA_30W" in df.columns:
        ax1.plot(df.index, df["SMA_30W"],
                 color=SMA_COL, linewidth=1.7, linestyle="--",
                 alpha=0.90, label="30W SMA", zorder=6)

    # (D) Resistance zone
    if has_res:
        ax1.axhspan(res_bot_val, res_top_val,
                    color=RES_COL, alpha=0.10, zorder=2, label="Resistance Zone")
        ax1.axhline(res_mid_val, color=RES_COL, linewidth=1.2,
                    linestyle="--", alpha=0.70, zorder=6)

    # (E) Support zone
    if has_sup:
        ax1.axhspan(sup_bot_val, sup_top_val,
                    color=SUP_COL, alpha=0.10, zorder=2, label="Support Zone")
        ax1.axhline(sup_mid_val, color=SUP_COL, linewidth=1.2,
                    linestyle="--", alpha=0.70, zorder=6)

    # (F) Touch markers
    if "Is_Res_Touch" in df.columns:
        rtp = df[df["Is_Res_Touch"] == True]
        if not rtp.empty:
            ax1.scatter(rtp.index, rtp["High"] * 1.013,
                        color=RES_TCH_COL, marker="v", s=110, zorder=7,
                        label=f"Res Touch ({len(rtp)})")

    if "Is_Sup_Touch" in df.columns:
        stp = df[df["Is_Sup_Touch"] == True]
        if not stp.empty:
            ax1.scatter(stp.index, stp["Low"] * 0.987,
                        color=SUP_TCH_COL, marker="^", s=110, zorder=7,
                        label=f"Sup Touch ({len(stp)})")

    # (G) Breakout arrow
    if has_res:
        _draw_breakout_arrow(ax1, df, res_top_val)

    # (H) Info box
    info = (
        f"Price      : {last['Close']:.2f}\n"
        f"Resistance : {float(last.get('Resistance', float('nan'))):.2f}"
        f"  ({int(last.get('Res_Touches', 0))} touches)\n"
        f"Support    : {float(last.get('Support', float('nan'))):.2f}"
        f"  ({int(last.get('Sup_Touches', 0))} touches)\n"
        f"Range Width: {float(last.get('Range_Width_Pct', float('nan'))):.1f}%\n"
        f"Dist→Break : {float(last.get('Distance_to_Breakout', float('nan'))):.1f}%"
    )
    ax1.text(0.01, 0.97, info, transform=ax1.transAxes,
             va="top", fontsize=8.5, color=TEXT_COL,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="#cccccc", alpha=0.92),
             family="monospace", zorder=10)

    ax1.set_ylabel("Price", color=TEXT_COL)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.90,
               facecolor="white", edgecolor="#cccccc", labelcolor=TEXT_COL)
    ax1.grid(True, linestyle=":", alpha=0.6, color=GRID_COL, zorder=1)

    # ── VOLUME PANEL ──────────────────────────────────────────────────
    opens  = df["Open"] if "Open" in df.columns else df["Close"].shift(1)
    colors = np.where(df["Close"] >= opens, VOL_UP, VOL_DN)
    ax2.bar(df.index, df["Volume"], color=colors, alpha=0.90, width=5)

    if "Vol_Avg_Curr_4W" in df.columns:
        ax2.plot(df.index, df["Vol_Avg_Curr_4W"],
                 color=VOL_AVG_COL, linewidth=1.4, label="4W Avg Vol", zorder=5)
        ax2.legend(fontsize=8, framealpha=0.90,
                   facecolor="white", edgecolor="#cccccc", labelcolor=TEXT_COL)

    ax2.set_ylabel("Volume", color=TEXT_COL)
    ax2.grid(True, linestyle=":", alpha=0.6, color=GRID_COL, zorder=1)

    # ── X-AXIS ────────────────────────────────────────────────────────
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, color=TICK_COL, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, dpi=120, facecolor=BG_FIG, bbox_inches="tight")
    plt.close(fig)
