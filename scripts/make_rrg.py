"""
make_rrg.py
-----------
Standalone Relative Rotation Graph generator.

Deliberately does NOT run any strategy scan: the RRG needs 12 weekly closes
for 12 symbols, not 1,600 tickers of OHLCV. That makes it fast enough and
cheap enough to run on a schedule (cron, GitHub Actions) and gives it far
fewer failure modes than daily_report.py -- a broken strategy can never stop
the rotation picture from being produced.

Outputs:
    results/rotation_chart.png   the RRG
    results/rotation.json        the ranked table, for any frontend to render

Usage:
    python scripts/make_rrg.py                      # chart + json
    python scripts/make_rrg.py --telegram           # ...and push to Telegram
    python scripts/make_rrg.py --out docs/rrg.png --json docs/rrg.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make the repo root importable no matter how this script is launched.
# `python scripts/make_rrg.py` puts scripts/ on sys.path[0], NOT the repo
# root -- so `import src...` fails even when the working directory is
# correct. Prepending the parent of this file's directory fixes it for the
# direct invocation, for cron, and for GitHub Actions alike, without
# requiring PYTHONPATH to be set by the caller.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("vektor.rrg")

# Yahoo throttles aggressively from cloud IPs (see the GitHub Actions notes in
# .github/workflows/rrg.yml). Retry with backoff rather than failing the run
# on the first refusal.
FETCH_ATTEMPTS = 4
FETCH_BACKOFF_SECONDS = 20


def _load_env() -> None:
    """Read .env, exactly as daily_report.py does.

    Matters specifically for --telegram under cron/systemd: those start with
    a near-empty environment, so TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID exist
    only if something reads .env. An interactive shell that already exported
    them hides this, which is why it works by hand and fails on a timer.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return  # python-dotenv optional; exported env vars still work
    load_dotenv(_REPO_ROOT / ".env")


def _resilient_loader():
    """DataLoader that falls back to a STALE cache when a download fails.

    Stock DataLoader.get_data() reads the cache only while it is fresh; if the
    refresh download then fails it returns None, and the sector silently
    vanishes from the RRG. That is the wrong trade for this job: Yahoo
    routinely refuses cloud-runner IPs, and a chart built on last week's
    closes is far better than no chart. A week-old weekly bar is, in any case,
    usually the same bar.

    Implemented as a subclass local to this script rather than a change to
    src/data_loader.py, deliberately: the screener may legitimately prefer to
    fail loudly on missing data, and this keeps the blast radius at one job.
    """
    import pandas as pd
    from src.data_loader import DataLoader

    class ResilientLoader(DataLoader):
        def get_data(self, ticker, start_date, end_date=None):
            df = super().get_data(ticker, start_date, end_date)
            if df is not None and not df.empty:
                return df
            path = self._cache_path(ticker)
            if not path.exists():
                log.error("%s: download failed and no cached copy exists.", ticker)
                return None
            try:
                cached = pd.read_csv(path, index_col=0, parse_dates=True)
            except Exception:  # noqa: BLE001
                log.error("%s: download failed and cache is unreadable.", ticker)
                return None
            if cached.empty:
                return None
            age_h = (time.time() - path.stat().st_mtime) / 3600.0
            log.warning("%s: download failed; using cache from %.1fh ago "
                        "(last bar %s).", ticker, age_h, cached.index[-1].date())
            return cached

    return ResilientLoader()


def build(out_png: Path, out_json: Path, tail_weeks: int) -> list[dict]:
    from src.benchmarks import benchmark_for
    from src.rotation import sector_rotation
    from notifications.rotation_chart import plot_sector_rotation

    loader = _resilient_loader()
    sectors: list[dict] = []
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            sectors = sector_rotation(loader, tail_weeks=tail_weeks)
        except Exception:  # noqa: BLE001
            log.exception("sector_rotation failed (attempt %d/%d)",
                          attempt, FETCH_ATTEMPTS)
            sectors = []
        if sectors:
            break
        if attempt < FETCH_ATTEMPTS:
            wait = FETCH_BACKOFF_SECONDS * attempt
            log.warning("No sector data on attempt %d/%d; retrying in %ds. "
                        "(Usually Yahoo rate-limiting a cloud IP.)",
                        attempt, FETCH_ATTEMPTS, wait)
            time.sleep(wait)

    if not sectors:
        log.error("No sector data after %d attempts; nothing written.",
                  FETCH_ATTEMPTS)
        return []

    benchmark = benchmark_for("US")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plot_sector_rotation(sectors, str(out_png),
                         title=f"Sector rotation (RRG) -- {stamp}",
                         benchmark_label=benchmark)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": benchmark,
        "tail_weeks": tail_weeks,
        "sectors": sectors,
    }, indent=2))

    log.info("Wrote %s and %s (%d sectors).", out_png, out_json, len(sectors))
    return sectors


def push_telegram(png: Path, sectors: list[dict]) -> None:
    from notifications.telegram_notifier import send_photo
    from src.rotation import QUADRANT_RANK
    ranked = sorted(sectors, key=lambda s: (QUADRANT_RANK.get(s["quadrant"], 9),
                                            -s["distance"]))
    caption = "\n".join(
        ["RRG -- JdK RS-Ratio / RS-Momentum",
         "clockwise: leading > weakening > lagging > improving", ""]
        + [f"{s['etf']:<5} {s['quadrant']:<10} "
           f"ratio {s['rs_ratio']:6.2f}  mom {s['rs_momentum']:6.2f}"
           for s in ranked])
    send_photo(str(png), caption=caption)
    log.info("Pushed RRG to Telegram.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the sector RRG.")
    ap.add_argument("--out", default="results/rotation_chart.png")
    ap.add_argument("--json", dest="json_out", default="results/rotation.json")
    ap.add_argument("--tail-weeks", type=int, default=12)
    ap.add_argument("--telegram", action="store_true",
                    help="also push the chart to Telegram")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    _load_env()

    sectors = build(Path(args.out), Path(args.json_out), args.tail_weeks)
    if not sectors:
        return 1
    if args.telegram:
        try:
            push_telegram(Path(args.out), sectors)
        except Exception:  # noqa: BLE001
            log.exception("Telegram push failed (chart was still written).")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())