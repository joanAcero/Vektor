"""
daily_report.py
----------------
Daily notification job for VEKTOR. Cleans stale charts/CSVs from a PREVIOUS
day out of results/ once, up front -- then runs the sector-rotation monitor,
then the configured strategy list, and pushes a summary to Telegram: a short
HTML text message, one Relative Rotation Graph photo for sector rotation,
then one chart photo per matched setup (with detail metrics in the caption).

Deliberately thin: it does NOT reimplement any scan logic. It consumes
run.py::run() and src/rotation.py::sector_rotation() exactly like webapp.py
does -- CLI, web UI and this daily job are three consumers of the same core,
so a change to a strategy or to rotation.py is picked up everywhere without
touching this file.

The results/ cleanup lives HERE, not inside run.py::run(), deliberately:
run() is shared by the CLI and webapp.py, both of which expect charts from a
run to still be there afterwards (webapp.py serves them to the browser).
Wiping the directory inside run() would delete a previous strategy's charts
out from under a still-open browser tab, AND would delete
results/.daily_seen.json every call -- silently resetting the "since <date>"
tracking below to "everything is new" forever. Cleaning once, here, before
anything this run writes, avoids both.

Usage:
    python daily_report.py

Meant to be invoked by scripts/run_daily.sh (cron/systemd), which pulls the
latest committed code before running this -- see that file for why.
"""

from __future__ import annotations

import html
import json
import logging
import shutil
import time
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import load_config
from src.data_loader import DataLoader
from src.benchmarks import benchmark_for
from src.registry import load_strategies, get_strategy
from src.rotation import QUADRANT_RANK, sector_rotation
from run import run

load_dotenv()  # reads .env if present; no-op if the vars are already exported

log = logging.getLogger("vektor.daily")

# ---------------------------------------------------------------------------
# Which strategies this job scans. Explicit list, same philosophy as
# strategies/__init__.py being the manifest of active strategies: the set of
# things that page you is a readable line in version control, not implicit.
#
# To include momentum_leaders again later: STRATEGY_KEYS = ["weinstein", "momentum_leaders"]
# Nothing else in this file changes.
# ---------------------------------------------------------------------------
STRATEGY_KEYS = ["weinstein"]

CONFIG_PATH = "config/default.yaml"
RESULTS_DIR = Path("results")
STATE_PATH = RESULTS_DIR / ".daily_seen.json"  # which tickers we've already flagged, and since when

# Photo attachments are UNCAPPED: every matched setup gets its chart sent.
# The previous MAX_ATTACHMENTS=10 silently truncated the tail of the list,
# which is worse than noisy -- with the recall-first Weinstein detector the
# match count regularly exceeds 10, and the dropped names were the
# lowest-Readiness_Score ones only by accident of sort order, never by an
# explicit decision.
#
# The tradeoff that replaces it: Telegram rate-limits bots to roughly 20
# messages per minute to a single chat, and sending ~40 photos back-to-back
# reliably earns HTTP 429s and dropped images. So sends are throttled rather
# than capped -- slower, but nothing is silently lost. A 60-setup day takes
# about three minutes to deliver, which is irrelevant for a cron job.
# If you would rather cap again, filter `attachments` on Readiness_Score in
# _format_setups (a ranking floor) instead of truncating a sorted list.
PHOTO_INTERVAL_SECONDS = 3.5

# Entries in results/ that survive the daily cleanup -- everything else
# (charts, CSVs) is leftover from a previous day and gets cleared.
_PRESERVE_IN_RESULTS = {STATE_PATH.name, "logs"}


def _clean_results_dir() -> None:
    """
    Remove yesterday's charts/CSVs from results/ so it doesn't grow forever,
    while keeping the seen-state file and the log directory. Runs ONCE, before
    anything in this job writes a new file -- so nothing generated this run
    (rotation chart, setup charts) is at risk of getting deleted mid-job.
    """
    if RESULTS_DIR.exists():
        for item in RESULTS_DIR.iterdir():
            if item.name in _PRESERVE_IN_RESULTS:
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _esc(value) -> str:
    """
    Escape a value for Telegram's HTML parse mode before interpolating it into
    the TEXT message (not photo captions -- those are sent without a
    parse_mode and don't need this). Applies to tickers, sector/industry
    names, stage values, exception text -- anything that isn't a literal
    <tag> we wrote ourselves.
    """
    return html.escape(str(value), quote=False)


def _load_seen() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Could not parse %s; starting with empty state.", STATE_PATH)
    return {}


def _save_seen(seen: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(seen, indent=2))


def _run_strategy(strategy_key: str) -> pd.DataFrame:
    # Explicit override, independent of whatever `strategy:` currently sits in
    # default.yaml (e.g. while you're interactively testing another strategy
    # via webapp.py) -- the daily job's scope is decided here, not by a file
    # someone else might be mid-edit on.
    cfg = load_config(CONFIG_PATH, overrides={"strategy": strategy_key})
    return run(cfg)


def _format_rotation_summary(sectors: list[dict]) -> str:
    """
    HTML summary of the Relative Rotation Graph for the Telegram text message.

    Reports the quadrant census, then calls out only what CHANGED this week,
    since the standing state is already legible on the attached chart:
      - sectors that entered a new quadrant this week,
      - sectors whose RS-Ratio crossed 100 (per the docs, a cross from the
        left half to the right half signals a new relative uptrend).

    Entries into `improving` are listed first and deliberately: that is the
    quadrant where relative momentum has turned up while the relative trend
    is still below par, which is exactly where Stage-1 bases sit. By the time
    a sector reaches `leading`, its best bases have usually already broken.
    """
    if not sectors:
        return "<b>Sector rotation</b>: no data."

    census = {q: 0 for q in ("leading", "improving", "weakening", "lagging")}
    for s in sectors:
        census[s["quadrant"]] = census.get(s["quadrant"], 0) + 1

    lines = ["<b>Sector rotation (RRG)</b>: "
             + ", ".join(f"{n} {q}" for q, n in census.items() if n)
             + " (chart attached)"]

    # Leaders first, using the documented ranking already applied by
    # sector_rotation(): quadrant order, then distance from the crosshair.
    top = [s for s in sectors if s["quadrant"] == "leading"][:4]
    if top:
        lines.append("\U0001F7E2 Leading: "
                     + _esc(", ".join(f"{s['etf']} ({s['rs_ratio']:.1f})"
                                      for s in top)))

    entered_improving = [s for s in sectors
                         if s["quadrant"] == "improving" and s["quadrant_changed"]]
    entered_leading = [s for s in sectors
                       if s["quadrant"] == "leading" and s["quadrant_changed"]]
    crossed_up = [s for s in sectors if s["ratio_crossed_up"]]
    crossed_down = [s for s in sectors if s["ratio_crossed_down"]]

    if entered_improving:
        lines.append("\U0001F535 Just turned improving (money starting in): "
                     + _esc(", ".join(s["etf"] for s in entered_improving)))
    if entered_leading:
        lines.append("\U0001F195 Just turned leading: "
                     + _esc(", ".join(s["etf"] for s in entered_leading)))
    if crossed_up:
        lines.append("\u26A1 RS-Ratio crossed above 100 (new relative uptrend): "
                     + _esc(", ".join(s["etf"] for s in crossed_up)))
    if crossed_down:
        lines.append("\U0001F53B RS-Ratio crossed below 100: "
                     + _esc(", ".join(s["etf"] for s in crossed_down)))

    return "\n".join(lines)


def _format_setups(strategy_key: str, df: pd.DataFrame,
                   seen: dict) -> tuple[str, list[tuple[str, str]]]:
    """
    Returns (short HTML summary line, [(chart_path, plain-text caption), ...]).
    A chart is attached for EVERY current setup, not just newly-appeared ones
    -- "first seen" only controls the (tag) shown, not whether you get to see
    the picture.
    """
    meta = get_strategy(strategy_key).meta
    today = date.today().isoformat()
    key_seen = seen.setdefault(strategy_key, {})

    if df.empty:
        return f"<b>{_esc(meta.display_name)}</b>: no setups today.", []

    extra_cols = [c for c in meta.display_columns if c in df.columns]
    ticker_tags: list[str] = []
    attachments: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        ticker = str(row["Ticker"])
        is_new = ticker not in key_seen
        if is_new:
            key_seen[ticker] = today
        first_seen = key_seen[ticker]

        ticker_tags.append(f"{_esc(ticker)} ({'NEW' if is_new else 'since ' + first_seen})")

        price = row.get("Price")
        price_str = f"{price:.2f}" if isinstance(price, (int, float)) else "?"
        caption_lines = [
            f"{ticker} -- ${price_str}",
            "NEW today" if is_new else f"tracking since {first_seen}",
            "",
        ]
        for c in extra_cols:
            val = row[c]
            val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
            caption_lines.append(f"{c}: {val_str}")

        chart_path = RESULTS_DIR / f"chart_{ticker}.png"
        if chart_path.exists():
            attachments.append((str(chart_path), "\n".join(caption_lines)))
        else:
            log.warning("No chart found for %s at %s -- was make_charts enabled?", ticker, chart_path)

    summary = f"<b>{_esc(meta.display_name)}</b> -- {len(df)} setup(s): " + ", ".join(ticker_tags)
    return summary, attachments


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    from notifications.telegram_notifier import send_message, send_photo
    from notifications.rotation_chart import plot_sector_rotation

    try:
        _clean_results_dir()  # once, up front -- see module docstring for why not in run()
        load_strategies("strategies")
        seen = _load_seen()
        loader = DataLoader()

        blocks = [f"<b>VEKTOR daily -- {date.today().isoformat()}</b>", ""]
        attachments: list[tuple[str, str]] = []
        failures: list[str] = []

        # --- Sector rotation FIRST: market context before individual setups ---
        try:
            sectors = sector_rotation(loader)
            rotation_error: Exception | None = None
        except Exception as e:  # noqa: BLE001 -- report, don't crash the whole job over this
            log.exception("Rotation monitor failed")
            sectors = []
            rotation_error = e

        if sectors:
            blocks.append(_format_rotation_summary(sectors))
            benchmark_symbol = benchmark_for("US")
            chart_path = plot_sector_rotation(
                sectors, str(RESULTS_DIR / "rotation_chart.png"),
                title=f"Sector rotation (RRG) -- {date.today().isoformat()}",
                benchmark_label=benchmark_symbol,
            )
            if chart_path:
                # Rank the caption the way the RRG symbol table is ranked, so
                # the text and the picture agree at a glance.
                ranked = sorted(sectors,
                                key=lambda s: (QUADRANT_RANK.get(s["quadrant"], 9),
                                               -s["distance"]))
                caption = "\n".join(
                    [f"RRG vs {benchmark_symbol} -- JdK RS-Ratio / RS-Momentum",
                     "rotation is clockwise: leading > weakening > lagging > improving",
                     ""]
                    + [f"{s['etf']:<5} {s['quadrant']:<10} "
                       f"ratio {s['rs_ratio']:6.2f}  mom {s['rs_momentum']:6.2f}"
                       for s in ranked])
                attachments.append((chart_path, caption))
        elif rotation_error is not None:
            blocks.append(f"<i>Sector rotation monitor failed: {_esc(rotation_error)}</i>")
        else:
            blocks.append("No sector rotation data returned.")

        blocks.append("")

        # --- Then the strategy setups ---
        for key in STRATEGY_KEYS:
            try:
                df = _run_strategy(key)
            except Exception as e:  # noqa: BLE001
                log.exception("Strategy %s failed", key)
                failures.append(f"<b>{_esc(key)}</b> FAILED: {_esc(e)}")
                continue
            summary, atts = _format_setups(key, df, seen)
            blocks.append(summary)
            attachments.extend(atts)

        if failures:
            blocks.append("")
            blocks.extend(failures)

        _save_seen(seen)
        message = "\n".join(blocks)

    except Exception as e:  # noqa: BLE001 -- last-resort net so a crash still pages you
        log.exception("daily_report.py crashed outside the per-strategy try blocks")
        try:
            send_message(f"<b>VEKTOR daily job crashed:</b>\n{_esc(e)}")
        except Exception:  # noqa: BLE001
            log.exception("Could not even send the crash notification")
        return 1

    send_message(message)

    # Every attachment is sent -- no cap. Throttled so Telegram's per-chat
    # rate limit doesn't drop the tail of a long list; a failed send is
    # logged and the loop continues rather than aborting the remaining photos.
    total = len(attachments)
    log.info("Sending %d photo attachment(s), ~%.0fs apart.",
             total, PHOTO_INTERVAL_SECONDS)
    for i, (path, caption) in enumerate(attachments, start=1):
        try:
            send_photo(path, caption=caption)
        except Exception:  # noqa: BLE001
            log.exception("Failed to send photo %s (%d/%d)", path, i, total)
        if i < total:
            time.sleep(PHOTO_INTERVAL_SECONDS)

    return 1 if failures and len(failures) == len(STRATEGY_KEYS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
