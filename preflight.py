"""
preflight.py
------------
Answers "why did the RRG job fail?" in one screen, before the real work runs.

Exists because a failed GitHub Actions run reports only `exit code 1`, which
is consistent with at least four different causes -- missing secrets, missing
dependencies, Yahoo refusing the runner, and an empty data cache. Guessing
between them from the annotations wastes a week per iteration when the job
runs weekly.

Exit codes:
    0  everything needed is present
    1  a hard blocker (missing dependency, or no data AND no cache)
    2  degraded but survivable (Yahoo unreachable but a usable cache exists)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RRG_SYMBOLS = ["SPY", "XLB", "XLC", "XLE", "XLF", "XLI",
               "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]


def _hdr(t: str) -> None:
    print(f"\n{'=' * 62}\n{t}\n{'=' * 62}")


def check_deps() -> list[str]:
    _hdr("1. DEPENDENCIES")
    missing = []
    for mod, why in (("pandas", "core"), ("numpy", "core"),
                     ("matplotlib", "chart rendering"), ("yfinance", "price data"),
                     ("requests", "telegram"), ("dotenv", "reads .env")):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"   ok      {mod:<12} {ver}")
        except ImportError:
            print(f"   MISSING {mod:<12} ({why})")
            missing.append(mod)
    try:
        __import__("curl_cffi")
        print("   ok      curl_cffi    (browser TLS impersonation available)")
    except ImportError:
        print("   absent  curl_cffi    (optional; may raise the block rate)")
    return missing


def check_secrets() -> list[str]:
    _hdr("2. TELEGRAM CREDENTIALS")
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env")
    except ImportError:
        pass
    missing = []
    for var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        val = os.environ.get(var, "")
        if val:
            # Never print a token. Length + last 4 is enough to spot a typo
            # or a value that got pasted with surrounding quotes.
            print(f"   ok      {var:<20} set (len {len(val)}, ends ...{val[-4:]})")
        else:
            print(f"   MISSING {var:<20} not set")
            missing.append(var)
    return missing


def check_cache() -> tuple[int, int]:
    _hdr("3. LOCAL DATA CACHE  (the fallback when Yahoo refuses)")
    data_dir = _REPO_ROOT / "data"
    if not data_dir.exists():
        print("   data/ does not exist -- no fallback available.")
        return 0, 0
    present = usable = 0
    for sym in RRG_SYMBOLS:
        p = data_dir / f"{sym}.csv"
        if not p.exists():
            print(f"   absent  {sym}")
            continue
        present += 1
        age_d = (time.time() - p.stat().st_mtime) / 86400.0
        try:
            import pandas as pd
            rows = len(pd.read_csv(p, index_col=0))
        except Exception:  # noqa: BLE001
            print(f"   BAD     {sym:<6} unreadable")
            continue
        usable += 1
        print(f"   ok      {sym:<6} {rows:>5} rows, {age_d:5.1f} days old")
    print(f"\n   {usable}/{len(RRG_SYMBOLS)} symbols cached and readable")
    return present, usable


def check_yahoo() -> bool:
    _hdr("4. YAHOO REACHABILITY FROM THIS MACHINE")
    try:
        import yfinance as yf
    except ImportError:
        print("   yfinance not installed; skipping.")
        return False
    try:
        df = yf.download("SPY", period="1mo", progress=False,
                         auto_adjust=True)
    except Exception as e:  # noqa: BLE001
        print(f"   FAIL    download raised: {type(e).__name__}: {e}")
        return False
    if df is None or df.empty:
        print("   FAIL    download returned an EMPTY frame.")
        print("           This is what a rate-limit / IP block looks like --")
        print("           yfinance swallows the HTTP error and returns nothing.")
        return False
    print(f"   ok      SPY: {len(df)} daily rows, last bar {df.index[-1].date()}")
    return True


def main() -> int:
    print("VEKTOR RRG preflight")
    print(f"repo root: {_REPO_ROOT}")
    print(f"python:    {sys.version.split()[0]}")

    missing_deps = check_deps()
    missing_secrets = check_secrets()
    _, usable_cache = check_cache()
    yahoo_ok = check_yahoo()

    _hdr("VERDICT")
    if missing_deps:
        print(f"   BLOCKED: missing dependencies {missing_deps}.")
        print("            Add them to requirements.txt.")
        return 1
    if missing_secrets:
        print(f"   BLOCKED: {missing_secrets} not set.")
        print("            GitHub: Settings > Secrets and variables > Actions.")
        print("            Local:  put them in .env at the repo root.")
        return 1
    if yahoo_ok:
        print("   OK: live data available; the RRG will use fresh closes.")
        return 0
    if usable_cache >= len(RRG_SYMBOLS):
        print("   DEGRADED: Yahoo is not serving this machine, but all 12")
        print("             symbols are cached. The RRG will build on the")
        print("             cached closes -- stale by up to a week, which for")
        print("             a weekly chart is usually the same bar.")
        return 2
    print("   BLOCKED: Yahoo is not serving this machine and the cache covers")
    print(f"            only {usable_cache}/{len(RRG_SYMBOLS)} symbols.")
    print("            Seed it once from a machine Yahoo does serve:")
    print("              python scripts/make_rrg.py")
    print("              git add -f data/SPY.csv data/XL*.csv")
    print("              git commit -m 'seed RRG data cache' && git push")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
