"""
rotation_chart.py
-----------------
Renders the sector monitor as a Relative Rotation Graph, in the visual
language StockCharts uses: JdK RS-Ratio on x, JdK RS-Momentum on y, axes
crossing at 100, four coloured quadrants, and a tail per symbol whose colour
is taken from the quadrant it ENDS in.

Two documented styling rules from the ChartSchool article are reproduced:
  * "The color of each line is determined by the color of the quadrant where
    the line ends."
  * "The thickness of each stock's line is determined by the ending dot's
    distance from the center of the plot."

Signature is unchanged from the previous implementation, so daily_report.py
needs no edit to call it.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # headless: cron / systemd have no display
import matplotlib.pyplot as plt
import numpy as np

from src.rotation import (QUADRANT_COLOR, QUADRANT_RANK, QUAD_IMPROVING,
                          QUAD_LAGGING, QUAD_LEADING, QUAD_WEAKENING,
                          RRG_ORIGIN)

log = logging.getLogger(__name__)

# Faint quadrant washes; the dots and tails must stay the loudest thing here.
_QUAD_ALPHA = 0.055
_BG = "#0d1117"
_FG = "#c9d1d9"
_GRID = "#30363d"


def _axis_limits(sectors: list[dict], pad_frac: float = 0.18) -> tuple[float, float]:
    """Symmetric half-width around 100 covering every current point AND every
    tail point, so no trail is clipped. Symmetry matters: an off-centre RRG
    misleads the eye about which side of the benchmark a sector is on."""
    devs = [0.75]  # floor, so a quiet week doesn't zoom to absurd magnification
    for s in sectors:
        devs.append(abs(float(s["rs_ratio"]) - RRG_ORIGIN))
        devs.append(abs(float(s["rs_momentum"]) - RRG_ORIGIN))
        for p in s.get("tail", []):
            devs.append(abs(float(p["rs_ratio"]) - RRG_ORIGIN))
            devs.append(abs(float(p["rs_momentum"]) - RRG_ORIGIN))
    half = max(devs) * (1.0 + pad_frac)
    return RRG_ORIGIN - half, RRG_ORIGIN + half


def plot_sector_rotation(sectors: list[dict], out_path: str, *,
                         title: str = "Sector rotation (RRG)",
                         benchmark_label: str = "SPY") -> str | None:
    """
    Draw the RRG and write it to `out_path`. Returns the path, or None when
    there is nothing to draw (callers treat a falsy return as "no chart").
    """
    if not sectors:
        log.warning("plot_sector_rotation: no sectors to plot.")
        return None

    lo, hi = _axis_limits(sectors)

    fig, (ax, ax_tbl) = plt.subplots(
        1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [3, 1.15]})
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax_tbl.set_facecolor(_BG)
    ax_tbl.axis("off")

    # ---- quadrant washes + labels -------------------------------------
    ax.add_patch(plt.Rectangle((RRG_ORIGIN, RRG_ORIGIN), hi - RRG_ORIGIN,
                               hi - RRG_ORIGIN, color=QUADRANT_COLOR[QUAD_LEADING],
                               alpha=_QUAD_ALPHA, zorder=0))
    ax.add_patch(plt.Rectangle((RRG_ORIGIN, lo), hi - RRG_ORIGIN,
                               RRG_ORIGIN - lo, color=QUADRANT_COLOR[QUAD_WEAKENING],
                               alpha=_QUAD_ALPHA, zorder=0))
    ax.add_patch(plt.Rectangle((lo, lo), RRG_ORIGIN - lo, RRG_ORIGIN - lo,
                               color=QUADRANT_COLOR[QUAD_LAGGING],
                               alpha=_QUAD_ALPHA, zorder=0))
    ax.add_patch(plt.Rectangle((lo, RRG_ORIGIN), RRG_ORIGIN - lo,
                               hi - RRG_ORIGIN, color=QUADRANT_COLOR[QUAD_IMPROVING],
                               alpha=_QUAD_ALPHA, zorder=0))

    pad = (hi - lo) * 0.015
    for txt, (x, y), (ha, va), quad in (
            ("LEADING", (hi - pad, hi - pad), ("right", "top"), QUAD_LEADING),
            ("WEAKENING", (hi - pad, lo + pad), ("right", "bottom"), QUAD_WEAKENING),
            ("LAGGING", (lo + pad, lo + pad), ("left", "bottom"), QUAD_LAGGING),
            ("IMPROVING", (lo + pad, hi - pad), ("left", "top"), QUAD_IMPROVING)):
        ax.text(x, y, txt, color=QUADRANT_COLOR[quad], fontsize=11,
                fontweight="bold", ha=ha, va=va, alpha=0.75, zorder=1)

    # ---- crosshair at the benchmark -----------------------------------
    ax.axvline(RRG_ORIGIN, color=_GRID, lw=1.2, ls="--", zorder=1)
    ax.axhline(RRG_ORIGIN, color=_GRID, lw=1.2, ls="--", zorder=1)

    # ---- tails, newest segment thickest --------------------------------
    # Draw laggards first so leaders end up on top of them.
    for s in sorted(sectors, key=lambda r: -QUADRANT_RANK.get(r["quadrant"], 9)):
        color = QUADRANT_COLOR.get(s["quadrant"], _FG)
        tail = s.get("tail", [])
        xs = [float(p["rs_ratio"]) for p in tail]
        ys = [float(p["rs_momentum"]) for p in tail]

        # Per the docs, line width encodes distance from the crosshair.
        base_lw = float(np.clip(0.8 + s["distance"] * 0.9, 0.8, 4.0))
        for i in range(1, len(xs)):
            t = i / max(1, len(xs) - 1)
            ax.plot(xs[i - 1:i + 1], ys[i - 1:i + 1], color=color,
                    lw=base_lw * (0.35 + 0.65 * t),
                    alpha=0.25 + 0.55 * t, solid_capstyle="round", zorder=3)
        if xs:
            ax.scatter(xs[:-1], ys[:-1], s=9, color=color, alpha=0.45, zorder=4)

        x, y = float(s["rs_ratio"]), float(s["rs_momentum"])
        ax.scatter([x], [y], s=190, color=color, edgecolors=_BG,
                   linewidths=1.8, zorder=6)
        ax.annotate(s["etf"], (x, y), color=color, fontsize=10.5,
                    fontweight="bold", xytext=(11, 6),
                    textcoords="offset points", zorder=7)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("JdK RS-Ratio  (relative trend  \u2192)", color=_FG, fontsize=11)
    ax.set_ylabel("JdK RS-Momentum  (relative momentum  \u2192)", color=_FG,
                  fontsize=11)
    ax.set_title(f"{title}\nbenchmark: {benchmark_label}   \u00b7   "
                 f"tails: {len(sectors[0].get('tail', []))} weeks   \u00b7   "
                 f"rotation is clockwise",
                 color=_FG, fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors=_FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(True, color=_GRID, alpha=0.28, lw=0.6)

    # ---- ranked symbol table (StockCharts orders it exactly this way) ---
    ax_tbl.text(0.0, 0.985, "RANKED BY QUADRANT, THEN DISTANCE",
                color=_FG, fontsize=9.5, fontweight="bold",
                va="top", family="monospace")
    ax_tbl.text(0.0, 0.955, f"{'ETF':<6}{'Ratio':>7}{'Mom':>7}{'Dist':>6}  wks",
                color="#8b949e", fontsize=8.6, va="top", family="monospace")
    y = 0.925
    current_quad = None
    for s in sectors:
        if s["quadrant"] != current_quad:
            current_quad = s["quadrant"]
            y -= 0.012
            ax_tbl.text(0.0, y, current_quad.upper(),
                        color=QUADRANT_COLOR.get(current_quad, _FG),
                        fontsize=8.8, fontweight="bold", va="top",
                        family="monospace")
            y -= 0.031
        wks = s.get("weeks_in_quadrant")
        wks_txt = "--" if wks is None else f"{wks}w"
        flag = " <" if s.get("quadrant_changed") else ""
        ax_tbl.text(0.0, y,
                    f"{s['etf']:<6}{s['rs_ratio']:>7.2f}{s['rs_momentum']:>7.2f}"
                    f"{s['distance']:>6.2f}  {wks_txt}{flag}",
                    color=QUADRANT_COLOR.get(s["quadrant"], _FG),
                    fontsize=8.6, va="top", family="monospace")
        y -= 0.029
    ax_tbl.text(0.0, max(y - 0.02, 0.01),
                "'<' = entered this quadrant this week",
                color="#6e7681", fontsize=7.6, va="top", family="monospace")

    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=130, facecolor=_BG)
    except OSError as e:
        log.error("Could not write rotation chart to %s: %s", out_path, e)
        plt.close(fig)
        return None
    plt.close(fig)
    return out_path
