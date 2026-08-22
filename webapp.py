"""
webapp.py
---------
Minimal web layer for selecting a run configuration and executing it. A thin
wrapper over the existing runner — it does NOT reimplement any scan logic.

Endpoints:
  GET  /                        -> the single-page UI
  GET  /api/strategies          -> strategies, param schemas, target options
  GET  /api/groups?kind=...     -> Finviz sector or industry names, for the picker
  POST /api/run                 -> build a RunConfig, call run(cfg), return JSON
  GET  /api/rotation            -> RRG snapshot of the 11 SPDR sector ETFs
  GET  /api/rotation/chart.png  -> the RRG image
  GET  /api/stage-chart/<SYM>.png -> a weekly chart with stage bands
  GET  /rrg                     -> standalone full-width RRG page

NOTHING THE BROWSER SHOWS IS A LIST THIS FILE MAINTAINS
=======================================================
Strategy fields come from each strategy's declared schema. The Quality labels
come from src/holdings.py. The index checkboxes come from src/indices.py. The
sector/industry names come from Finviz. Add an index or swap a fund and the UI
follows with no frontend edit — which is the only way a list stays correct.

CONFIG FILE FIRST, REQUEST SECOND
=================================
/api/run does NOT construct a RunConfig field by field. It loads
config/default.yaml and overrides only the fields the UI exposes. Building one
from scratch here meant every field the UI does not expose -- rotation
quadrants, data_start, cache_max_age_hours -- silently fell back to a dataclass
default, bypassing the validation in src/config.py entirely.

WHY THE PAYLOAD STILL CARRIES ROWS
==================================
The UI no longer draws a results table — the charts are the output. It still
needs `columns` and `rows`: the sort menu orders the charts by any column the
strategy produced, the sector census is counted from them, and the sector
filter selects on them. The table itself is written to CSV by run.py.

Run with:  python webapp.py   (then open http://127.0.0.1:5000)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from run import run
from src.config import RunConfig, coerce_params, load_config
from src.holdings import fund_label
from src.indices import index_options
from src.instrument import results_columns
from src.market_config import MARKETS
from src.registry import get_registry, load_strategies

log = logging.getLogger("vektor.web")

CONFIG_PATH = "config/default.yaml"
OUTPUT_DIR = "results"

app = Flask(__name__, static_folder="web", static_url_path="")

load_strategies("strategies")


# ---------------------------------------------------------------------------
# Generated images are BUILD ARTIFACTS of the code that drew them, so their
# freshness is compared against that code's mtime and not only against the
# clock. A pure time-based TTL cannot notice a deploy, which is how charts
# rendered by a previous version of a renderer survive for hours after the
# module changed — with a cache-busting query string on the client masking it.
# ---------------------------------------------------------------------------
def _newest_mtime(*modules) -> float:
    newest = 0.0
    for module in modules:
        path = getattr(module, "__file__", None)
        if not path:
            continue
        try:
            newest = max(newest, Path(path).stat().st_mtime)
        except OSError:
            pass
    return newest


def _is_stale(png: Path, max_age: float, *renderer_modules) -> bool:
    if not png.exists():
        return True
    age = time.time() - png.stat().st_mtime
    if age > max_age:
        return True
    return png.stat().st_mtime < _newest_mtime(*renderer_modules)


def _field_type(default) -> str:
    """Infer a UI FORM field type from a param's default value.

    Presentation only — which widget to draw. Value coercion is done by
    src/config.py against the schema's declared `type`, so the browser and the
    YAML cannot disagree about what an int is.
    """
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    return "text"


def _cfg_from_request(body: dict) -> RunConfig:
    """Load the config file, then apply only what the UI controls.

    Raises ValueError for anything the config module rejects; the caller turns
    that into a 400 so the user sees the reason instead of an empty result set.
    """
    cfg = load_config(CONFIG_PATH)
    cfg.output_dir = OUTPUT_DIR

    cfg.strategy_key = body["strategy"]

    # Coerce params through the strategy schema. Unknown names from the browser
    # are dropped rather than raised on: the UI rebuilds its fields when the
    # strategy changes, and a stale field in flight should not fail a run.
    cls = get_registry()[cfg.strategy_key]
    allowed = {p.name for p in cls.meta.param_schema}
    raw = {k: v for k, v in (body.get("params") or {}).items() if k in allowed}
    cfg.strategy_params = coerce_params(cfg.strategy_key, raw)

    # `market` in the payload; the UI calls this section the Target. See the
    # naming note in src/config.py before renaming either.
    market = body.get("market") or {}
    cfg.market_mode = (market.get("mode") or cfg.market_mode).lower()

    if cfg.market_mode == "us":
        cfg.us_source = (market.get("us_source") or cfg.us_source).lower()

        raw_tickers = market.get("tickers", "")
        if isinstance(raw_tickers, str):
            raw_tickers = raw_tickers.split(",")
        cfg.us_tickers = [str(t).strip().upper()
                          for t in (raw_tickers or []) if str(t).strip()]

        if "indices" in market:
            cfg.us_indices = list(market.get("indices") or [])
        if "top_n_industries" in market:
            cfg.us_top_n_industries = int(market["top_n_industries"])
        cfg.us_perf_col = market.get("perf_col", cfg.us_perf_col)
        # "" means rank the groups; a name means scan that one. Not validated
        # against a list -- Finviz owns that vocabulary. See src/config.py.
        cfg.us_group = str(market.get("group", "") or "").strip()
        # us_rotation_quadrants is deliberately NOT taken from the request.
        # It is configuration; it stays whatever config/default.yaml says.
    else:
        cfg.intl_source = (market.get("intl_source") or cfg.intl_source).lower()
        cfg.intl_codes = list(market.get("intl_codes") or [])

    # Re-validate: the request may have changed us_source AFTER load, so the
    # indices, the group and the quadrants all have to be checked against the
    # EFFECTIVE source.
    return cfg.validate()


@app.get("/")
def index():
    resp = send_from_directory("web", "index.html")
    # Dev UI: never cache the shell. A stale index.html is indistinguishable
    # from a broken deploy and costs an hour to diagnose.
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/strategies")
def strategies():
    out = []
    # Ordered by DISPLAY NAME, not by key: this list is read by a human, and
    # the key is an internal identifier the UI never shows.
    for key, cls in sorted(get_registry().items(),
                           key=lambda kv: kv[1].meta.display_name.lower()):
        m = cls.meta
        params = [
            {
                "name": p.name,
                "default": p.default,
                "field_type": "choice" if p.choices else _field_type(p.default),
                "choices": list(p.choices or ()),
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
    quality = {"us": fund_label("us_quality"), "intl": fund_label("intl_quality")}
    return jsonify({"strategies": out, "intl_markets": intl,
                    "quality_labels": quality,
                    "us_indices": index_options()})


# Finviz group names change (new industries appear, names get reworded) and
# there are ~150 industries, so the picker is populated from Finviz rather than
# from a list in this repo. Cached in-process: the picker is opened far more
# often than the taxonomy changes, and every miss is a real request.
_GROUP_CACHE: dict[str, tuple[float, list[str]]] = {}
GROUP_CACHE_SECONDS = 6 * 3600


@app.get("/api/groups")
def groups():
    kind = (request.args.get("kind") or "").strip().lower()
    if kind not in ("sector", "industry"):
        return jsonify({"error": "kind must be 'sector' or 'industry'"}), 400

    cached = _GROUP_CACHE.get(kind)
    if cached and (time.time() - cached[0]) < GROUP_CACHE_SECONDS:
        return jsonify({"groups": cached[1]})

    from src.finviz_engine import FinvizEngine
    engine = FinvizEngine()
    try:
        # Reuses the EXISTING public ranking methods with a top_n large enough
        # to return everything, rather than adding a second way of asking
        # Finviz for group names. The performance order is discarded: this list
        # is for FINDING a name, and alphabetical is the only order you can
        # search by eye.
        names = (engine.get_top_sectors(top_n=999) if kind == "sector"
                 else engine.get_top_industries(top_n=999))
    except Exception as e:  # noqa: BLE001
        log.exception("Finviz group list failed")
        return jsonify({"error": f"Could not reach Finviz: {e}"}), 502

    names = sorted({str(n).strip() for n in (names or []) if str(n).strip()})
    if not names:
        return jsonify({"error": "Finviz returned no groups."}), 502

    _GROUP_CACHE[kind] = (time.time(), names)
    return jsonify({"groups": names})


@app.post("/api/run")
def run_endpoint():
    body = request.get_json(force=True) or {}
    strategy_key = body.get("strategy")
    if not strategy_key or strategy_key not in get_registry():
        return jsonify({"error": f"Unknown strategy {strategy_key!r}"}), 400

    # mode / us_source / indices / quadrants are all validated by
    # src/config.py. Not re-checked here: a second copy of those rules is a
    # second source of truth, and the browser copy is the one that goes stale.
    try:
        cfg = _cfg_from_request(body)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        log.exception("Config file missing")
        return jsonify({"error": f"Config file not found: {CONFIG_PATH}"}), 500

    if cfg.us_source == "ticker" and not cfg.us_tickers:
        return jsonify({"error": "Ticker mode selected but no tickers given."}), 400

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
        if cfg.us_source == "ticker":
            return jsonify({"columns": [], "rows": [], "count": 0,
                            "charts": _ticker_charts(),
                            "note": "No setup matched, but here are the detected bases."})
        return jsonify({"columns": [], "rows": [], "count": 0})

    # Same column order as the CLI table and the CSV — one function decides it.
    meta = get_registry()[cfg.strategy_key].meta
    cols = results_columns(meta.display_columns, df.columns)
    rows = df[cols].round(4).astype(object).where(df[cols].notna(), None).values.tolist()

    charts = {}
    if "Ticker" in df.columns:
        out_dir = Path(cfg.output_dir)
        for ticker in df["Ticker"]:
            fname = f"chart_{ticker}.png"
            if (out_dir / fname).exists():
                charts[ticker] = f"/charts/{fname}"
    if cfg.us_source == "ticker":
        charts.update(_ticker_charts())

    return jsonify({"columns": cols, "rows": rows, "count": len(df), "charts": charts})


@app.get("/charts/<path:filename>")
def charts(filename):
    # Serve generated chart PNGs from the results directory.
    return send_from_directory(OUTPUT_DIR, filename)


@app.get("/api/rotation")
def rotation_endpoint():
    """
    RRG snapshot of the 11 SPDR sector ETFs against SPY.

    Each sector carries its quadrant, its RS-Ratio and RS-Momentum, and
    `sector_stage` — the absolute Weinstein stage of the ETF's own weekly
    chart, from src/stages.py.
    """
    from src.data_loader import DataLoader
    from src.rotation import rotation_history, sector_rotation

    loader = DataLoader()
    try:
        sectors = sector_rotation(loader)
    except Exception as e:  # noqa: BLE001
        log.exception("Rotation failed")
        return jsonify({"error": str(e)}), 500

    payload = {"sectors": sectors}

    # The UI requests ?history=0 — it shows the stage gallery instead — so this
    # branch is currently dead from the browser. Check for other callers before
    # deleting rotation_history().
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


RRG_PNG_NAME = "rotation_chart.png"
RRG_MAX_AGE_SECONDS = 6 * 3600  # regenerate at most every 6h per browser hit
STAGE_PNG_MAX_AGE = 6 * 3600


@app.get("/rrg")
def rrg_page():
    """Standalone Relative Rotation Graph page."""
    resp = send_from_directory("web", "rrg.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/rotation/chart.png")
def rotation_chart_png():
    """Serve the RRG image, regenerating it when missing, aged out, or older
    than the renderer that drew it. Pass ?force=1 to rebuild immediately."""
    import src.rotation
    import notifications.rotation_chart

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / RRG_PNG_NAME

    if (request.args.get("force") == "1"
            or _is_stale(png, RRG_MAX_AGE_SECONDS,
                         src.rotation, notifications.rotation_chart)):
        from notifications.rotation_chart import plot_sector_rotation
        from src.benchmarks import benchmark_for
        from src.data_loader import DataLoader
        from src.rotation import sector_rotation
        try:
            sectors = sector_rotation(DataLoader())
            if sectors:
                plot_sector_rotation(
                    sectors, str(png),
                    title="Sector rotation (RRG)",
                    benchmark_label=benchmark_for("US"))
        except Exception:  # noqa: BLE001
            log.exception("RRG render failed")
            # fall through: serve the stale image if we have one

    if not png.exists():
        return jsonify({"error": "RRG unavailable"}), 503
    resp = send_from_directory(str(out_dir), RRG_PNG_NAME)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/stage-chart/<symbol>.png")
def stage_chart_png(symbol):
    """Weekly chart with stage bands, for any symbol. Generic on purpose: the
    ETF gallery and any future per-stock view are the same request.

    Invalidated against src/stages.py as well as the renderer — a change to the
    classifier changes every band on every one of these images.
    """
    import src.stages
    import notifications.stage_chart

    # Path component reaching the filesystem: allow only ticker characters.
    if not re.fullmatch(r"[A-Za-z0-9._^-]{1,15}", symbol):
        return jsonify({"error": "bad symbol"}), 400

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"stage_{symbol.upper()}.png"
    png = out_dir / name

    if (request.args.get("force") == "1"
            or _is_stale(png, STAGE_PNG_MAX_AGE,
                         src.stages, notifications.stage_chart)):
        from notifications.stage_chart import plot_stage_chart
        from src.benchmarks import get_weekly_close
        from src.data_loader import DataLoader
        from src.rotation import SECTOR_ETFS
        try:
            wk = get_weekly_close(DataLoader(), symbol.upper(),
                                  start_date="2016-01-01")
            if wk is not None:
                plot_stage_chart(symbol.upper(), wk, str(png),
                                 subtitle=SECTOR_ETFS.get(symbol.upper(), ""))
        except Exception:  # noqa: BLE001
            log.exception("Stage chart render failed for %s", symbol)
            # fall through: a stale image beats a broken one

    if not png.exists():
        return jsonify({"error": "chart unavailable"}), 503
    resp = send_from_directory(str(out_dir), name)
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    app.run(host="127.0.0.1", port=5000, debug=False)
