"""
check_versions.py — Verifica que tienes las versiones correctas de los archivos.
Colócalo en la raíz del repo y ejecuta:  python check_versions.py
"""
import os, sys

REPO = os.path.dirname(os.path.abspath(__file__))

CHECKS = [
    ("src/config.py",                "us_source",              "RunConfig.us_source (modo ticker)"),
    ("src/config.py",                "us_tickers",             "RunConfig.us_tickers"),
    ("webapp.py",                    "us_source",              "webapp pasa us_source"),
    ("webapp.py",                    "_ticker_charts",         "webapp charts de diagnóstico"),
    ("run.py",                       "diagnostic",             "run.py modo ticker diagnóstico"),
    ("src/market_us.py",             "collect_explicit_tickers","market_us escaneo por ticker"),
    ("src/market_us.py",             "collect_us_by_sector",   "market_us ranking por sector"),
    ("src/market_us.py",             "get_all_market_details", "market_us full market"),
    ("src/finviz_engine.py",         "get_top_sectors",        "finviz ranking sectores"),
    ("src/screener.py",              "progress_cb",            "screener tracking de progreso"),
    ("src/screener.py",              "Progress:",              "screener log de %"),
    ("strategies/weinstein_setup.py","base_length",            "weinstein preset base_length"),
    ("strategies/weinstein_setup.py","max_ma_decline_pct",     "weinstein banda MA asimétrica (decline)"),
    ("strategies/weinstein_setup.py","max_ma_rise_pct",        "weinstein banda MA asimétrica (rise)"),
    ("web/index.html",               "us_source",              "frontend selector de fuente"),
]

# Cadenas que NO deben estar (restos de versiones viejas)
FORBIDDEN = [
    ("strategies/weinstein_setup.py", '"stage"',            "param 'stage' eliminado (post_breakout)"),
    ("strategies/weinstein_setup.py", "max_ma_slope_pct",   "banda simétrica antigua reemplazada por asimétrica"),
    ("strategies/weinstein_setup.py", "max_weeks_since_breakout", "param post_breakout eliminado"),
]

def main():
    ok = True
    print(f"Verificando versiones en: {REPO}\n")

    caches = []
    for root, dirs, _ in os.walk(REPO):
        if "__pycache__" in dirs:
            caches.append(os.path.join(root, "__pycache__"))
    if caches:
        print(f"⚠️  Hay {len(caches)} carpeta(s) __pycache__ — bórralas para evitar versiones cacheadas:")
        print("    find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null\n")

    print("— Debe estar presente —")
    for rel, needle, desc in CHECKS:
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            print(f"❌ FALTA   {rel:32} ({desc})"); ok = False; continue
        content = open(path, encoding="utf-8").read()
        if needle in content:
            print(f"✅ OK      {rel:32} ({desc})")
        else:
            print(f"❌ ANTIGUO {rel:32} — falta '{needle}' ({desc})"); ok = False

    print("\n— NO debe estar (restos viejos) —")
    for rel, needle, desc in FORBIDDEN:
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            continue
        content = open(path, encoding="utf-8").read()
        if needle in content:
            print(f"❌ RESTO   {rel:32} — aún contiene '{needle}' ({desc})"); ok = False
        else:
            print(f"✅ LIMPIO  {rel:32} (sin '{needle}')")

    print()
    if ok:
        print("✅ TODO CORRECTO. Reinicia el servidor Flask / consola PyCharm.")
    else:
        print("❌ Hay archivos desactualizados. Reemplázalos y vuelve a ejecutar.")
        sys.exit(1)

if __name__ == "__main__":
    main()
