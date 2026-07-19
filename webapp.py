"""
webapp.py
---------
Minimal web layer for selecting a run configuration and executing it. This is a
thin wrapper over the existing runner — it does NOT reimplement any scan logic.

Endpoints:
  GET  /                -> the single-page UI
  GET  /api/strategies  -> registered strategies + their param schemas
  POST /api/run         -> build a RunConfig, call run(cfg), return results JSON
  GET  /api/rotation    -> Weinstein sector-selection monitor

The UI renders parameter fields dynamically from each strategy's declared
schema, so a newly added strategy gets correct form fields with no frontend
changes — preserving the plugin design of the rest of the repo.

Run with:  python webapp.py   (then open http://127.0.0.1:5000)
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src.config import RunConfig
from src.market_config import MARKETS
from src.registry import get_registry, load_strategies
from run import run

log = logging.getLogger("vektor.web")

# Directory the runner writes charts/CSV into; also where we serve charts from.
OUTPUT_DIR = "results"

app = Flask(__name__, static_folder="web", static_url_path="")

# Populate the registry once at import time.
load_strategies("strategies")


def _field_type(default) -> str:
    """Infer a UI field type from a param's default value (robust vs lambda types)."""
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    return "text"


def _coerce(field_type: str, value):
    if field_type == "bool":
        return bool(value)
    if field_type == "int":
        return int(value)
    if field_type == "float":
        return float(value)
    return value


@app.get("/")
def index():
    return send_from_directory("web", "index.html")


@app.get("/api/strategies")
def strategies():
    out = []
    for key, cls in sorted(get_registry().items()):
        m = cls.meta
        params = [
            {
                "name": p.name,
                "default": p.default,
                "field_type": "choice" if p.choices else _field_type(p.default),
                "choices": list(p.choices),
                "help": p.help,
            }
            for p in m.param_schema
        ]
        out.append({
            "key": key,
            "display_name": m.display_name,
            "description": m.description,
            "params": params,
        })
    intl = [{"code": c, "name": MARKETS[c]["name"]} for c in MARKETS if c != "US"]
    return jsonify({"strategies": out, "intl_markets": intl})


@app.post("/api/run")
def run_endpoint():
    body = request.get_json(force=True) or {}
    strategy_key = body.get("strategy")
    if not strategy_key or strategy_key not in get_registry():
        return jsonify({"error": f"Unknown strategy {strategy_key!r}"}), 400

    # Coerce params using the strategy's schema (field types inferred from defaults).
    cls = get_registry()[strategy_key]
    schema = {p.name: p for p in cls.meta.param_schema}
    raw_params = body.get("params", {}) or {}
    params = {}
    for name, value in raw_params.items():
        if name not in schema:
            continue  # ignore stray fields rather than erroring
        params[name] = _coerce(_field_type(schema[name].default), value)

    market = body.get("market", {}) or {}
    mode = (market.get("mode") or "us").lower()
    if mode not in ("us", "international"):
        return jsonify({"error": "market.mode must be 'us' or 'international'"}), 400

    us_source = (market.get("us_source") or "industries").lower()
    if us_source not in ("industries", "sectors", "rotation", "ticker"):
        return jsonify({"error": "us_source must be 'industries', 'sectors', 'rotation' or 'ticker'"}), 400

    # Parse comma-separated tickers if in ticker mode.
    raw_tickers = market.get("tickers", "")
    if isinstance(raw_tickers, str):
        us_tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    else:
        us_tickers = [str(t).strip().upper() for t in (raw_tickers or []) if str(t).strip()]
    if us_source == "ticker" and not us_tickers:
        return jsonify({"error": "Ticker mode selected but no tickers given."}), 400

    # In ticker mode, default charts ON — the whole point is to inspect the base.
    make_charts = bool(body.get("make_charts", us_source == "ticker"))

    cfg = RunConfig(
        strategy_key=strategy_key,
        strategy_params=params,
        market_mode=mode,
        intl_codes=list(market.get("intl_codes", []) or []),
        us_source=us_source,
        us_tickers=us_tickers,
        us_top_n_industries=int(market.get("top_n_industries", 5)),
        us_perf_col=market.get("perf_col", "Perf Week"),
        output_dir=OUTPUT_DIR,
        make_charts=make_charts,
    )

    try:
        df = run(cfg)
    except Exception as e:  # noqa: BLE001 — surface errors to the UI cleanly
        log.exception("Run failed")
        return jsonify({"error": str(e)}), 500

    # In ticker (diagnostic) mode, return the charts even if nothing matched —
    # the chart shows the detected base so you can see WHY it didn't match.
    def _ticker_charts():
        out_dir = Path(cfg.output_dir)
        ch = {}
        for ticker in cfg.us_tickers:
            fname = f"chart_{ticker}.png"
            if (out_dir / fname).exists():
                ch[ticker] = f"/charts/{fname}"
        return ch

    if df is None or df.empty:
        if us_source == "ticker":
            return jsonify({"columns": [], "rows": [], "count": 0,
                            "charts": _ticker_charts(),
                            "note": "No setup matched, but here are the detected bases."})
        return jsonify({"columns": [], "rows": [], "count": 0})

    # Order columns like the CLI does, keeping only those present.
    meta = cls.meta
    cols = ["Market", "Ticker", "Price", *meta.display_columns]
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    rows = df[cols].round(4).astype(object).where(df[cols].notna(), None).values.tolist()

    # Map each ticker to its chart URL (those that exist on disk).
    charts = {}
    if cfg.make_charts and "Ticker" in df.columns:
        out_dir = Path(cfg.output_dir)
        for ticker in df["Ticker"]:
            fname = f"chart_{ticker}.png"
            if (out_dir / fname).exists():
                charts[ticker] = f"/charts/{fname}"
    if us_source == "ticker":
        charts.update(_ticker_charts())

    return jsonify({"columns": cols, "rows": rows, "count": len(df), "charts": charts})


@app.get("/charts/<path:filename>")
def charts(filename):
    # Serve generated chart PNGs from the results directory.
    return send_from_directory(OUTPUT_DIR, filename)


@app.get("/api/rotation")
def rotation_endpoint():
    """
    Weinstein sector-selection monitor. Classifies the 11 SPDR sector ETFs
    as hunt / watch / avoid using sector stage (own weekly chart) plus
    Mansfield RS vs SPY. The frontend uses this both for the money-flow
    plot and for driving `us_source=rotation` scans.
    """
    from src.data_loader import DataLoader
    from src.rotation import sector_rotation, rotation_history

    loader = DataLoader()
    try:
        sectors = sector_rotation(loader)
    except Exception as e:  # noqa: BLE001
        log.exception("Rotation failed")
        return jsonify({"error": str(e)}), 500

    payload = {"sectors": sectors}

    # Rotation history: how each sector's state evolved over recent weeks,
    # so the user can see WHEN a rotation started (fresh = still early).
    if request.args.get("history", "1") != "0":
        try:
            weeks = int(request.args.get("weeks", 26))
        except ValueError:
            weeks = 26
        try:
            payload["history"] = rotation_history(loader, weeks=weeks)
        except Exception as e:  # noqa: BLE001
            log.warning("Rotation history failed: %s", e)
            payload["history"] = {"dates": [], "sectors": []}

    return jsonify(payload)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    app.run(host="127.0.0.1", port=5000, debug=False)
