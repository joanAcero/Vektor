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
  GET  /api/ibd-lists           -> IBD screen exports present in data/ibd/
  POST /api/ibd-upload          -> accept an IBD export and file it in data/ibd/
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
import shutil
import tempfile
import threading
import time
from datetime import date, datetime
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
INDUSTRY_LEADERS_PROGRESS_PATH = Path(OUTPUT_DIR) / "industry_leaders_progress.json"
app = Flask(__name__, static_folder="web", static_url_path="")

# In-memory progress for the screener run (single-user app, no concurrency concern).
_RUN_LOCK = threading.Lock()
RUN_PROGRESS: dict = {
    "running": False,
    "done": 0,
    "total": 0,
    "matched": 0,
    "started_at": None,   # time.monotonic() of first progress callback
    "eta_seconds": None,
}

# An IBD screen export is a few KB. The cap is here so a mis-picked file fails
# at the socket instead of being buffered into memory, and it is generous
# enough that no legitimate export ever hits it.
IBD_UPLOAD_MAX_BYTES = 2 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = IBD_UPLOAD_MAX_BYTES

load_strategies("strategies")


@app.errorhandler(413)
def _too_large(_e):
    # Flask's default 413 is an HTML page, which the fetch() caller parses as
    # JSON and reports as a syntax error instead of as "file too big".
    return jsonify({"error": f"File is larger than "
                             f"{IBD_UPLOAD_MAX_BYTES // 1024} KB."}), 413


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
        # Which IBD screen export to read. Not validated against the directory
        # here for the same reason the group is not validated against Finviz:
        # src/config.py owns that rule and src/market_us.py reports what it
        # actually found on disk.
        cfg.us_ibd_list = str(market.get("ibd_list", "") or "").strip()
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


# The IBD picker is populated from the FILESYSTEM, not from a list here, for
# the same reason the Finviz picker is populated from Finviz: the set of lists
# is whatever you have downloaded into data/ibd/, and a hardcoded copy would go
# stale the first time you export a new screen. Not cached -- it is a directory
# listing of a handful of files, and a stale picker after a fresh download is
# exactly the confusion this endpoint exists to avoid.
@app.get("/api/ibd-lists")
def ibd_lists():
    from src.ibd_import import available_ibd_lists
    try:
        lists = available_ibd_lists()
    except OSError as e:
        log.exception("Could not read the IBD export directory")
        return jsonify({"error": f"Could not read data/ibd: {e}"}), 500
    # {name: [ISO dates, newest first]} -- the UI shows the newest as the
    # as-of date so a forgotten download is visible before you run, not after.
    return jsonify({"lists": lists})


# List names become filename stems, so the slug is restricted rather than
# sanitised: an allow-list of [a-z0-9_] cannot express a path separator, a
# parent reference or a leading dot, and the target is re-checked against the
# directory afterwards anyway.
_IBD_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _ibd_slug(raw: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(raw or "").strip().lower()).strip("_")


@app.post("/api/ibd-upload")
def ibd_upload():
    """Accept an IBD screen export and file it as a dated snapshot.

    PARSED BEFORE IT IS PERSISTED. The upload goes to a temp file, through
    src/ibd_import.py, and only reaches data/ibd/ if it yielded tickers. The
    alternative -- write first, discover on the next run -- puts a file in the
    picker that produces an empty scan, which is indistinguishable from a
    market where nothing set up.

    The as-of date comes from the client because the export does not carry one.
    It defaults to today, which is right when you download and upload in the
    same session and wrong otherwise, so it is an editable field rather than a
    server-side assumption.
    """
    from src.ibd_import import IBD_DATA_DIR, SUPPORTED_SUFFIXES, read_ibd_export

    upload = request.files.get("file")
    if upload is None or not (upload.filename or "").strip():
        return jsonify({"error": "No file was uploaded."}), 400

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return jsonify({"error": f"Unsupported file type {suffix or '(none)'}. "
                                 f"Expected one of {list(SUPPORTED_SUFFIXES)}."}), 400

    slug = _ibd_slug(request.form.get("list_name") or Path(upload.filename).stem)
    if not _IBD_SLUG_RE.match(slug):
        return jsonify({"error": "The list name must start with a letter and "
                                 "contain only letters, digits and "
                                 "underscores."}), 400

    raw_date = (request.form.get("as_of") or "").strip()
    if raw_date:
        try:
            as_of = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": f"as_of must be YYYY-MM-DD, got "
                                     f"{raw_date!r}."}), 400
    else:
        as_of = date.today()
    if as_of > date.today():
        # A future date wins every point-in-time comparison in
        # load_ibd_list(), so a typo here would shadow every real snapshot.
        return jsonify({"error": f"as_of {as_of} is in the future."}), 400

    target_dir = Path(IBD_DATA_DIR).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = (target_dir / f"{slug}_{as_of.isoformat()}{suffix}").resolve()
    if target.parent != target_dir:
        return jsonify({"error": "Refusing to write outside data/ibd."}), 400

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            upload.save(tmp)
            tmp_path = Path(tmp.name)

        try:
            snapshot = read_ibd_export(tmp_path, list_name=slug, as_of=as_of)
        except (ValueError, FileNotFoundError) as e:
            return jsonify({"error": f"That file did not parse as an IBD "
                                     f"export: {e}"}), 400

        # One snapshot per list per date, whatever the extension. Without this
        # an .xlsx and a .csv for the same day both match in load_ibd_list()
        # and the tie is broken arbitrarily.
        replaced = []
        for other in SUPPORTED_SUFFIXES:
            existing = target_dir / f"{slug}_{as_of.isoformat()}{other}"
            if existing.exists():
                replaced.append(existing.name)
                existing.unlink()

        shutil.move(str(tmp_path), target)
        tmp_path = None
        log.info("IBD upload: %s -> %s (%d tickers)%s", upload.filename,
                 target.name, len(snapshot),
                 f", replacing {replaced}" if replaced else "")
    except OSError as e:
        log.exception("IBD upload failed")
        return jsonify({"error": f"Could not save the file: {e}"}), 500
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return jsonify({"list_name": slug, "as_of": as_of.isoformat(),
                    "filename": target.name, "count": len(snapshot),
                    "replaced": replaced})


@app.post("/api/run")
def run_endpoint():
    body = request.get_json(force=True) or {}
    strategy_key = body.get("strategy")
    if not strategy_key or strategy_key not in get_registry():
        return jsonify({"error": f"Unknown strategy {strategy_key!r}"}), 400

    try:
        cfg = _cfg_from_request(body)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        log.exception("Config file missing")
        return jsonify({"error": f"Config file not found: {CONFIG_PATH}"}), 500

    if cfg.us_source == "ticker" and not cfg.us_tickers:
        return jsonify({"error": "Ticker mode selected but no tickers given."}), 400

    # Wire a progress callback into the Screener so the frontend can poll it.
    def _on_progress(done: int, total: int, matched: int) -> None:
        with _RUN_LOCK:
            now = time.monotonic()
            started = RUN_PROGRESS["started_at"]
            if started is None:
                RUN_PROGRESS["started_at"] = now
                started = now
            elapsed = now - started
            eta = None
            if done > 0 and total > 0 and elapsed > 0:
                rate = done / elapsed          # tickers/sec
                remaining = (total - done) / rate
                eta = round(remaining)
            RUN_PROGRESS.update({
                "running": True,
                "done": done,
                "total": total,
                "matched": matched,
                "eta_seconds": eta,
            })

    with _RUN_LOCK:
        RUN_PROGRESS.update({
            "running": True,
            "done": 0,
            "total": 0,
            "matched": 0,
            "started_at": None,
            "eta_seconds": None,
        })

    from src.screener import Screener as _Screener  # local to avoid circular import
    # Monkey-patch the progress_cb onto a fresh Screener inside run() by
    # temporarily wrapping the Screener constructor.
    _orig_screener_init = _Screener.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_screener_init(self, *args, **kwargs)
        self.progress_cb = _on_progress

    _Screener.__init__ = _patched_init
    try:
        df = run(cfg)
    except Exception as e:  # noqa: BLE001 — surface errors to the UI cleanly
        log.exception("Run failed")
        return jsonify({"error": str(e)}), 500
    finally:
        _Screener.__init__ = _orig_screener_init
        with _RUN_LOCK:
            RUN_PROGRESS["running"] = False

    # In ticker (diagnostic) mode, return the charts even if nothing matched.
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


@app.get("/api/run-progress")
def run_progress_endpoint():
    """Poll-able progress for the active screener run."""
    with _RUN_LOCK:
        snap = dict(RUN_PROGRESS)
    snap.pop("started_at", None)
    return jsonify(snap)


@app.get("/api/market-sentiment")
def market_sentiment_endpoint():
    """
    Fetch Fear & Greed index from CNN and AAII sentiment survey.
    Returns raw values plus a regime label for Fear & Greed.
    """
    import requests
    import json as _json

    result: dict = {"fear_greed": None, "aaii": None, "errors": []}

    # Common browser-like headers to reduce bot-blocking from Cloudflare / CDN
    browser_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/116.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://www.cnn.com/",
    }

    # --- Fear & Greed (CNN) ---
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        try:
            r = requests.get(url, headers=browser_headers, timeout=8)
        except requests.exceptions.RequestException as e:
            raise Exception(f"network error: {e}")

        if r.status_code != 200:
            # Provide a clearer error message for blocked requests (418, 403, etc.)
            raise Exception(f"HTTP {r.status_code}: {r.reason}")

        # Some CDNs return an HTML page instead of JSON when blocking; detect that
        text = r.text.strip()
        if text.startswith("<"):
            raise Exception("Blocked or returned HTML instead of JSON")

        data = r.json()
        score = round(data["fear_and_greed"]["score"], 1)
        rating = data["fear_and_greed"].get("rating")
        # Map to our own regime labels
        if score > 90:
            regime = "extreme_greed"
        elif score > 75:
            regime = "greed"
        elif score > 25:
            regime = "neutral"
        elif score > 10:
            regime = "fear"
        elif score > 5:
            regime = "panic"
        else:
            regime = "extreme_panic"
        result["fear_greed"] = {"score": score, "cnn_rating": rating, "regime": regime}
    except Exception as e:  # noqa: BLE001
        # Keep error details for the client to show and retry guidance
        result["errors"].append(f"Fear&Greed: {e}")

    # --- AAII Sentiment Survey ---
    # Fetch from Nasdaq Data Link (formerly Quandl) dataset code AAII/AAII_SENTIMENT
    aaii_fetched = False
    nasdaq_urls = [
        "https://data.nasdaq.com/api/v3/datasets/AAII/AAII_SENTIMENT.json?rows=1",
        "https://www.quandl.com/api/v3/datasets/AAII/AAII_SENTIMENT.json?rows=1",
        "https://data.nasdaq.com/api/v3/datasets/AAII/AAII_SENTIMENT.csv?rows=1",
        "https://www.quandl.com/api/v3/datasets/AAII/AAII_SENTIMENT.csv?rows=1",
    ]

    aaii_req_headers = browser_headers.copy()
    aaii_req_headers.update({"Accept": "application/json, text/csv, */*"})

    for url in nasdaq_urls:
        try:
            r = requests.get(url, headers=aaii_req_headers, timeout=8)
            if r.status_code == 200:
                if ".json" in url:
                    data_obj = r.json()
                    dataset = data_obj.get("dataset") or data_obj.get("dataset_data")
                    if dataset and "column_names" in dataset and "data" in dataset:
                        cols = [str(c).lower().strip() for c in dataset["column_names"]]
                        rows = dataset["data"]
                        if rows and len(rows) > 0:
                            latest = rows[0]
                            def _val_json(col_kw):
                                for idx, col in enumerate(cols):
                                    if col_kw in col:
                                        v = latest[idx]
                                        if v is not None:
                                            try:
                                                v = float(v)
                                                return round(v * 100 if v < 1 else v, 1)
                                            except (ValueError, TypeError):
                                                pass
                                return None
                            bull = _val_json("bullish")
                            bear = _val_json("bearish")
                            neut = _val_json("neutral")
                            dt_idx = cols.index("date") if "date" in cols else 0
                            dt_val = str(latest[dt_idx]).strip() if latest and len(latest) > dt_idx else None
                            if bull is not None and bear is not None:
                                result["aaii"] = {
                                    "bullish": bull,
                                    "neutral": neut,
                                    "bearish": bear,
                                    "bull_avg": 38.0,
                                    "bear_avg": 31.5,
                                    "date": dt_val,
                                }
                                aaii_fetched = True
                                break
                elif ".csv" in url:
                    import csv, io
                    reader = list(csv.reader(io.StringIO(r.text)))
                    if len(reader) >= 2:
                        cols = [str(c).lower().strip() for c in reader[0]]
                        latest = reader[1]
                        def _val_csv(col_kw):
                            for idx, col in enumerate(cols):
                                if col_kw in col:
                                    v = latest[idx]
                                    if v is not None:
                                        try:
                                            v = float(v)
                                            return round(v * 100 if v < 1 else v, 1)
                                        except (ValueError, TypeError):
                                            pass
                            return None
                        bull = _val_csv("bullish")
                        bear = _val_csv("bearish")
                        neut = _val_csv("neutral")
                        dt_idx = cols.index("date") if "date" in cols else 0
                        dt_val = str(latest[dt_idx]).strip() if latest and len(latest) > dt_idx else None
                        if bull is not None and bear is not None:
                            result["aaii"] = {
                                "bullish": bull,
                                "neutral": neut,
                                "bearish": bear,
                                "bull_avg": 38.0,
                                "bear_avg": 31.5,
                                "date": dt_val,
                            }
                            aaii_fetched = True
                            break
        except Exception:  # noqa: BLE001
            continue

    if not aaii_fetched:
        # Fallback to legacy AAII CSV/XLS URLs if Nasdaq Data Link is unavailable
        try:
            csv_url = "https://www.aaii.com/files/surveys/sentiment.csv"
            xls_url = "https://www.aaii.com/files/surveys/sentiment.xls"
            aaii_site_headers = browser_headers.copy()
            aaii_site_headers.update({"Referer": "https://www.aaii.com/", "Accept": "text/csv, */*; q=0.1"})

            raw = None
            used_url = None
            for url in (csv_url, xls_url):
                try:
                    r = requests.get(url, headers=aaii_site_headers, timeout=8)
                except requests.exceptions.RequestException:
                    continue
                if r.status_code == 200:
                    raw = r.content
                    used_url = url
                    break

            if raw is not None and used_url:
                if used_url.lower().endswith('.csv'):
                    import csv, io
                    try:
                        rows = list(csv.reader(io.StringIO(raw.decode('utf-8'))))
                    except UnicodeDecodeError:
                        rows = list(csv.reader(io.StringIO(raw.decode('latin-1'))))
                else:
                    import io
                    try:
                        import openpyxl
                        wb = openpyxl.load_workbook(io.BytesIO(raw), data_only=True)
                        ws = wb.active
                        rows = list(ws.iter_rows(values_only=True))
                    except Exception:
                        import xlrd  # type: ignore
                        wb = xlrd.open_workbook(file_contents=raw)
                        ws = wb.sheet_by_index(0)
                        rows = [ws.row_values(i) for i in range(ws.nrows)]

                header_idx = None
                for i, row in enumerate(rows):
                    if any("bullish" in str(c).lower() for c in row):
                        header_idx = i
                        break
                if header_idx is not None:
                    headers = [str(c).lower().strip() for c in rows[header_idx]]
                    data_row = None
                    for row in reversed(rows[header_idx + 1:]):
                        if row and row[0] and str(row[0]).strip():
                            data_row = row
                            break
                    if data_row:
                        bull_idx = next((i for i, h in enumerate(headers) if "bull" in h), None)
                        bear_idx = next((i for i, h in enumerate(headers) if "bear" in h), None)
                        neutral_idx = next((i for i, h in enumerate(headers) if "neutral" in h), None)
                        def _pct(v):
                            try:
                                v = float(v)
                                return round(v * 100 if v < 1 else v, 1)
                            except (TypeError, ValueError):
                                return None
                        result["aaii"] = {
                            "bullish": _pct(data_row[bull_idx]) if bull_idx is not None else None,
                            "neutral": _pct(data_row[neutral_idx]) if neutral_idx is not None else None,
                            "bearish": _pct(data_row[bear_idx]) if bear_idx is not None else None,
                            "bull_avg": 38.0,
                            "bear_avg": 31.5,
                            "date": str(data_row[0]).strip() if data_row else None,
                        }
                        aaii_fetched = True
        except Exception:  # noqa: BLE001
            pass

    if not aaii_fetched and result["aaii"] is None:
        result["errors"].append("AAII: No se pudo obtener la encuesta de sentimiento (Nasdaq Data Link AAII/AAII_SENTIMENT ni AAII.com)")

    return jsonify(result)


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


def render_market_chart(symbol: str, out_path: str, title: str = "") -> bool:
    """
    Renders weekly market index chart with:
      - Weekly candles
      - MA(10) dashed and MA(30) solid
      - Weekly volume with Vol MA(10)
      - MACD(12, 26, 9) histogram and lines
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from src.data_loader import DataLoader
    from src.indicators import macd

    loader = DataLoader()
    df = loader.get_data(symbol.upper(), start_date="2018-01-01")
    if df is None or df.empty or len(df) < 50:
        return False

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    wk = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    if len(wk) < 35:
        return False

    wk["MA10"] = wk["Close"].rolling(10).mean()
    wk["MA30"] = wk["Close"].rolling(30).mean()
    macd_df = macd(wk["Close"])
    for col in macd_df.columns:
        wk[col] = macd_df[col]
    wk["Vol_MA10"] = wk["Volume"].rolling(10).mean()

    # Last 3 years of weekly bars
    cutoff = wk.index.max() - pd.Timedelta(weeks=156)
    view = wk.loc[wk.index >= cutoff].copy()
    if len(view) < 10:
        view = wk.copy()

    fig, (ax_price, ax_vol, ax_macd) = plt.subplots(
        3, 1, figsize=(13, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    fig.patch.set_facecolor("#ffffff")

    # 1. Price candles
    o, h, l, c = view["Open"].values, view["High"].values, view["Low"].values, view["Close"].values
    x = mdates.date2num(view.index.to_pydatetime())
    for i in range(len(view)):
        col = "#26a641" if c[i] >= o[i] else "#e03131"
        ax_price.plot([x[i], x[i]], [l[i], h[i]], color=col, linewidth=0.9, zorder=3)
        ax_price.add_patch(mpatches.Rectangle(
            (x[i] - 2.5, min(o[i], c[i])), 5, abs(c[i] - o[i]) or 1e-9,
            facecolor=col, edgecolor=col, linewidth=0.4, zorder=4,
        ))

    ax_price.plot(view.index, view["MA30"], color="#e07b00", linewidth=1.8,
                  label="MA30 semanal", zorder=5)
    ax_price.plot(view.index, view["MA10"], color="#2563eb", linewidth=1.4,
                  linestyle="--", label="MA10 semanal", zorder=5)
    ax_price.set_ylabel("Precio", fontsize=10, fontweight="bold")
    ax_price.grid(True, linestyle=":", alpha=0.6)
    ax_price.legend(loc="lower left", fontsize=9, framealpha=0.9)

    # 2. Volume
    vols = view["Volume"].values
    vcol = np.where(c >= o, "#86efac", "#fca5a5")
    ax_vol.bar(view.index, vols, color=vcol, width=5, zorder=2)
    ax_vol.plot(view.index, view["Vol_MA10"], color="#334155", linewidth=1.2,
                label="Vol MA(10)", zorder=3)
    ax_vol.set_ylabel("Volumen", fontsize=10, fontweight="bold")
    ax_vol.grid(True, linestyle=":", alpha=0.6)
    ax_vol.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # 3. MACD
    hist = view["MACD_Hist"]
    hcol = np.where(hist.fillna(0) >= 0, "#93c5fd", "#fecaca")
    ax_macd.bar(view.index, hist, color=hcol, width=5, zorder=2)
    ax_macd.plot(view.index, view["MACD"], color="#1d4ed8", linewidth=1.3,
                 label="MACD(12,26)", zorder=4)
    ax_macd.plot(view.index, view["MACD_Signal"], color="#b91c1c", linewidth=1.1,
                 label="Signal(9)", zorder=4)
    ax_macd.axhline(0, color="#94a3b8", linewidth=0.9, zorder=1)
    ax_macd.set_ylabel("MACD", fontsize=10, fontweight="bold")
    ax_macd.grid(True, linestyle=":", alpha=0.6)
    ax_macd.legend(loc="upper left", fontsize=8, framealpha=0.85, ncol=2)

    ax_macd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_macd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_macd.get_xticklabels(), rotation=45, ha="right")

    full_title = title or f"{symbol} — Gráfico semanal"
    fig.suptitle(full_title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


@app.get("/api/market-chart/<symbol>.png")
def market_chart_png(symbol):
    """
    Weekly chart for market benchmarks (RSP, SPY, QQQ, DIA, IWM) with:
    MA(10), MA(30), Volume, and MACD.
    """
    from src.benchmarks import US_MARKET_INDICES

    if not re.fullmatch(r"[A-Za-z0-9._^-]{1,15}", symbol):
        return jsonify({"error": "bad symbol"}), 400

    sym_upper = symbol.upper()
    title_match = ""
    for _code, (name, s) in US_MARKET_INDICES.items():
        if s.upper() == sym_upper:
            title_match = f"{name} ({sym_upper}) — Gráfico semanal"
            break

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"market_{sym_upper}.png"
    png = out_dir / name

    if request.args.get("force") == "1" or not png.exists() or (time.time() - png.stat().st_mtime > 3600 * 6):
        try:
            ok = render_market_chart(sym_upper, str(png), title=title_match)
            if not ok:
                return jsonify({"error": "chart generation failed"}), 500
        except Exception:  # noqa: BLE001
            log.exception("Market chart render failed for %s", sym_upper)

    if not png.exists():
        return jsonify({"error": "chart unavailable"}), 503
    resp = send_from_directory(str(out_dir), name)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/market-chart/<symbol>")
def market_chart_page(symbol):
    """HTML wrapper page displaying the market weekly chart in a clean viewer tab."""
    from src.benchmarks import US_MARKET_INDICES

    sym_upper = symbol.upper()
    title_text = f"{sym_upper} — Gráfico semanal"
    for _code, (name, s) in US_MARKET_INDICES.items():
        if s.upper() == sym_upper:
            title_text = f"{name} ({sym_upper}) — Gráfico semanal"
            break

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>{title_text} | VEKTOR</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" type="image/png" href="/config/icon.png">
  <style>
    body {{
      margin: 0; padding: 24px; background: #0f172a; color: #f8fafc;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      min-height: 100vh; box-sizing: border-box;
    }}
    .header {{
      width: 100%; max-width: 1300px; display: flex; justify-content: space-between;
      align-items: center; margin-bottom: 16px;
    }}
    h1 {{ font-size: 18px; margin: 0; font-weight: 600; }}
    .badge {{
      background: #1e293b; color: #94a3b8; padding: 4px 10px; border-radius: 6px;
      font-size: 13px; font-family: ui-monospace, Menlo, monospace;
    }}
    .img-wrap {{
      background: #ffffff; padding: 12px; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      max-width: 100%; overflow: auto;
    }}
    img {{ display: block; max-width: 100%; height: auto; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{title_text}</h1>
    <span class="badge">MA10 · MA30 · Volumen · MACD</span>
  </div>
  <div class="img-wrap">
    <img src="/api/market-chart/{sym_upper}.png" alt="{title_text}">
  </div>
</body>
</html>"""
    return html
@app.get("/api/market-state")
def market_state_endpoint():
    """
    Semáforo de estado para los índices.
    """
    from flask import request
    from src.data_loader import DataLoader
    from src.benchmarks import us_market_states, intl_market_states

    region = request.args.get("region", "us")
    loader = DataLoader()
    try:
        if region == "international":
            long_term = intl_market_states(loader, timeframe="long")
            medium_term = intl_market_states(loader, timeframe="medium")
        else:
            long_term = us_market_states(loader, timeframe="long")
            medium_term = us_market_states(loader, timeframe="medium")
    except Exception as e:  # noqa: BLE001
        log.exception("Market state failed")
        return jsonify({"error": str(e)}), 500
    return jsonify({
        "long_term": long_term,
        "medium_term": medium_term,
        "markets": long_term,
    })

@app.get("/api/sector-leaders")
def sector_leaders_endpoint():
    """
    Los 11 sectores SPDR, ordenados por Mansfield RS descendente, cada uno
    marcado como `is_leader` si cumple el filtro de Weinstein (Etapa 2 y
    RS > 0). Permite seleccionar el horizonte de cálculo de Mansfield RS
    vía query param `rs_weeks` (52 por defecto, 26 semestral, 13 trimestral).
    """
    from flask import request
    from src.data_loader import DataLoader
    from src.rotation import sector_rotation

    rs_weeks = request.args.get("rs_weeks", 52, type=int)
    if rs_weeks not in (13, 26, 52):
        rs_weeks = 52

    loader = DataLoader()
    try:
        sectors = sector_rotation(loader, rs_weeks=rs_weeks)
    except Exception as e:  # noqa: BLE001
        log.exception("Sector leaders failed")
        return jsonify({"error": str(e)}), 500

    for s in sectors:
        s["is_leader"] = (s["sector_stage"] == "stage2"
                          and (s["mansfield_rs"] or 0) > 0)

    # Un solo criterio de orden para los 11: RS descendente. is_leader no
    # entra en la clave de orden -- es un resaltado sobre el ranking, no un
    # segundo nivel de agrupación que rompería la continuidad del RS.
    sectors.sort(key=lambda s: s["mansfield_rs"] if s["mansfield_rs"] is not None
                              else float("-inf"), reverse=True)

    leader_count = sum(1 for s in sectors if s["is_leader"])
    return jsonify({
        "sectors": sectors,
        "leader_count": leader_count,
        "total_sectors": len(sectors),
        "rs_weeks": rs_weeks,
    })

@app.get("/api/sector-leaders-history")
def sector_leaders_history_endpoint():
    """
    El filtro de Weinstein recalculado en cada una de las últimas 6 semanas
    (src/rotation.py::sector_leaders_history), para ver qué sectores son
    líderes NUEVOS frente a hace unas semanas. Permite seleccionar el
    horizonte de cálculo de Mansfield RS vía query param `rs_weeks`
    (52 por defecto, 26 semestral, 13 trimestral).
    """
    from flask import request
    from src.data_loader import DataLoader
    from src.rotation import sector_leaders_history

    rs_weeks = request.args.get("rs_weeks", 52, type=int)
    if rs_weeks not in (13, 26, 52):
        rs_weeks = 52

    loader = DataLoader()
    try:
        history = sector_leaders_history(loader, weeks_back=6, rs_weeks=rs_weeks)
        history["rs_weeks"] = rs_weeks
    except Exception as e:  # noqa: BLE001
        log.exception("Sector leaders history failed")
        return jsonify({"error": str(e)}), 500
    return jsonify(history)

INDUSTRY_LEADERS_CACHE_PATH = Path(OUTPUT_DIR) / "industry_leaders_cache.json"
INDUSTRY_LEADERS_MAX_AGE = 12 * 3600  # mismo TTL que DataLoader
_industry_leaders_job = {"running": False}


@app.get("/api/industry-leaders")
def industry_leaders_endpoint():
    if _industry_leaders_job["running"]:
        progress = {"done": 0, "total": 0, "current": ""}
        if INDUSTRY_LEADERS_PROGRESS_PATH.exists():
            import json
            try:
                progress = json.loads(INDUSTRY_LEADERS_PROGRESS_PATH.read_text())
            except (OSError, json.JSONDecodeError):
                pass  # progreso es "nice to have"; un fichero a medio escribir no debe romper el polling
        return jsonify({"status": "calculating", **progress})

    if INDUSTRY_LEADERS_CACHE_PATH.exists():
        age = time.time() - INDUSTRY_LEADERS_CACHE_PATH.stat().st_mtime
        if age < INDUSTRY_LEADERS_MAX_AGE:
            import json
            data = json.loads(INDUSTRY_LEADERS_CACHE_PATH.read_text())
            data["status"] = "ready"
            data["age_seconds"] = int(age)
            return jsonify(data)

    return jsonify({"status": "not_calculated"})

@app.post("/api/industry-leaders/calculate")
def industry_leaders_calculate_endpoint():
    """
    Dispara el cálculo de forma SÍNCRONA (la petición no responde hasta que
    termina). Es deliberadamente simple -- sin cola de tareas, sin hilos --
    a costa de que el navegador debe usar un fetch sin timeout corto. El
    flag `running` existe solo para que /api/industry-leaders (polling)
    pueda decir "calculando" mientras tanto, si el usuario recarga la
    página en otra pestaña.
    """
    import json
    from src.data_loader import DataLoader
    from src.finviz_engine import FinvizEngine
    from src.rotation import sector_rotation
    from src.industry_rotation import industry_leaders

    if _industry_leaders_job["running"]:
        return jsonify({"error": "Ya hay un cálculo en curso."}), 409

    _industry_leaders_job["running"] = True
    started = time.time()

    def _write_progress(phase, done, total, current):
        # best-effort: un fallo de escritura de progreso no debe abortar el
        # cálculo real -- de ahí el except mudo.
        try:
            elapsed = time.time() - started
            per_item = elapsed / done if done else 0
            eta_seconds = int(per_item * max(0, total - done))
            INDUSTRY_LEADERS_PROGRESS_PATH.write_text(json.dumps({
                "phase": phase, "done": done, "total": total,
                "current": current, "eta_seconds": eta_seconds,
            }))
        except OSError:
            pass

    try:
        loader = DataLoader()
        sectors = sector_rotation(loader)
        for s in sectors:
            s["is_leader"] = (s["sector_stage"] == "stage2"
                              and (s["mansfield_rs"] or 0) > 0)
        result = industry_leaders(loader, FinvizEngine(), sectors,
                                  on_progress=_write_progress)
        result["computed_at"] = time.time()
        INDUSTRY_LEADERS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        INDUSTRY_LEADERS_CACHE_PATH.write_text(json.dumps(result))
        return jsonify({**result, "status": "ready", "age_seconds": 0})
    except Exception as e:  # noqa: BLE001
        log.exception("Industry leaders calculation failed")
        return jsonify({"error": str(e)}), 500
    finally:
        _industry_leaders_job["running"] = False
        INDUSTRY_LEADERS_PROGRESS_PATH.unlink(missing_ok=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    app.run(host="127.0.0.1", port=5000, debug=True)
