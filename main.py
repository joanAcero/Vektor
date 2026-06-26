"""
main.py
-------
VEKTOR — Weinstein Setup Screener (Pre-Breakout) — Global Edition

Flow
----
1.  Mode selection     : US  OR  International (mutually exclusive).
2.  Strategy config    : shared parameters (SMA, lookback, proximity, …).
3.  US scan
      · Rank Finviz *industries* by recent performance.
      · Fetch tickers + Sector/Industry metadata per top-N industry.
      · Run Weinstein screener; results tagged with Sector & Industry.
4.  International scan
      · Choose which non-US markets to include.
      · Collect every ticker from every selected market (no sector pre-filter).
      · Results tagged with Market (country code) + local Sector.
5.  Display, save CSV & PNG charts.
"""

import os
import pandas as pd

from src.data_loader import DataLoader
from src.screener import Screener
from src.finviz_engine import FinvizEngine
from src.international_engine import InternationalEngine
from src.market_config import MarketConfig, MARKETS
from src.utils import get_input_or_default, clean_data_folder
from strategies.weinstein_setup import WeinsteinSetup
from src.plotter import plot_weinstein_setup


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _run_us_scan(
    finviz: FinvizEngine,
    screener: Screener,
    strategy: WeinsteinSetup,
    top_n: int,
    col_target: str,
) -> pd.DataFrame:
    """
    US path: rank industries → fetch tickers with sector metadata → scan.
    """
    print("\n" + "═" * 70)
    print("  🇺🇸  US MARKET — Industry-level scan")
    print("═" * 70)

    top_industries = finviz.get_top_industries(top_n=top_n, col_target=col_target)
    if not top_industries:
        print("❌ No se pudieron obtener industrias de Finviz.")
        return pd.DataFrame()

    print(f"\n✅ Top industrias (Finviz): {top_industries}")

    meta_frames: list[pd.DataFrame] = []
    for industry in top_industries:
        details = finviz.get_ticker_details_in_industry(industry)
        # 🛡️ FIX: Validar que 'details' no sea None antes de usar .empty
        if details is not None and not details.empty:
            meta_frames.append(details)

    if not meta_frames:
        print("⚠️  Sin tickers para las industrias seleccionadas.")
        return pd.DataFrame()

    meta_df = pd.concat(meta_frames, ignore_index=True).drop_duplicates(subset="Ticker")
    candidates = meta_df["Ticker"].tolist()
    print(f"\n   🎯 Total candidatos US: {len(candidates)}")

    raw_results = screener.scan(strategy, candidates, market_label="US")
    # 🛡️ FIX: Validar que 'raw_results' no sea None
    if raw_results is None or raw_results.empty:
        return pd.DataFrame()

    # Enrich with Sector / Industry
    ticker_to_sector   = meta_df.set_index("Ticker")["Sector"].to_dict()
    ticker_to_industry = meta_df.set_index("Ticker")["Industry"].to_dict()

    raw_results["Sector"]   = raw_results["Ticker"].map(ticker_to_sector).fillna("")
    raw_results["Industry"] = raw_results["Ticker"].map(ticker_to_industry).fillna("")

    return raw_results


def _run_international_scan(
    selected_non_us_codes: list[str],
    screener: Screener,
    strategy: WeinsteinSetup,
) -> pd.DataFrame:
    """
    International path: sweep ALL tickers from all selected non-US markets
    """
    print("\n" + "═" * 70)
    print("  🌍  INTERNATIONAL MARKETS — Full-market sweep")
    print(f"      Markets: {selected_non_us_codes}")
    print("═" * 70)

    all_candidates: list[str] = []
    ticker_market: dict[str, str] = {}
    ticker_sector: dict[str, str] = {}

    for code in selected_non_us_codes:
        market_cfg = MarketConfig(code)
        print(f"\n   📋 {market_cfg.name}")
        for sector_name, base_tickers in market_cfg.sectors.items():
            for base in base_tickers:
                full = market_cfg.full_ticker(base)
                if full not in ticker_market:
                    ticker_market[full] = code
                    ticker_sector[full] = sector_name
                    all_candidates.append(full)

    all_candidates = list(dict.fromkeys(all_candidates))
    print(f"\n   🎯 Total candidatos internacionales: {len(all_candidates)}")

    raw_results = screener.scan(strategy, all_candidates, market_label="INTL")
    # 🛡️ FIX: Validar que 'raw_results' no sea None
    if raw_results is None or raw_results.empty:
        return pd.DataFrame()

    raw_results["Market"]   = raw_results["Ticker"].map(ticker_market).fillna("?")
    raw_results["Sector"]   = raw_results["Ticker"].map(ticker_sector).fillna("")
    raw_results["Industry"] = ""

    return raw_results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 80)
    print(" 🎯 VEKTOR: WEINSTEIN SETUP SCREENER (PRE-BREAKOUT) — GLOBAL")
    print("=" * 80)

    try:
        # ── 1. MODE: US  OR  INTERNATIONAL  (mutually exclusive) ────────────────
        print("\n🔀 ¿Qué mercado quieres analizar?")
        print("   [0] 🇺🇸  US Market  (NYSE / NASDAQ — análisis por industria)")
        print("   [1] 🌍  International Markets  (Europa — barrido completo)")

        mode_raw = input("🔹 Selección [Default: 0 — US]: ").strip()
        if mode_raw == "1":
            scan_us = False
        elif mode_raw in ("", "0"):
            scan_us = True
        else:
            print("⚠️  Entrada inválida. Usando US por defecto.")
            scan_us = True

        # ── 1b. INTERNATIONAL: which markets? ────────────────────────────────
        intl_codes: list[str] = []
        if not scan_us:
            intl_available = {code: cfg for code, cfg in MARKETS.items() if code != "US"}
            intl_list      = list(intl_available.keys())

            print("\n🌍 Mercados internacionales disponibles:")
            for i, (code, cfg) in enumerate(intl_available.items()):
                print(f"   [{i}] {code}  —  {cfg['name']}")

            print("\n💡 Introduce los números separados por coma.")
            print("   Ejemplo: '0,2' activa ES + GB   |   Enter = todos")
            raw_sel = input("🔹 Mercados a incluir [Default: todos]: ").strip()

            if not raw_sel:
                intl_codes = intl_list
            else:
                try:
                    indices    = [int(x.strip()) for x in raw_sel.split(",")]
                    intl_codes = [intl_list[i] for i in indices if 0 <= i < len(intl_list)]
                except ValueError:
                    print("⚠️  Entrada inválida. Usando todos los mercados internacionales.")
                    intl_codes = intl_list

            if not intl_codes:
                print("❌ Ningún mercado seleccionado.")
                return

            print(f"\n✅ Mercados activos: {intl_codes}  (barrido completo sin filtro sectorial)")

        else:
            print("\n✅ Modo: US Market  (análisis por industria Finviz)")

        # ── 2. PERFORMANCE PERIOD ─────────────────────────────────────────────
        perf_choice = get_input_or_default(
            "Performance: Semana(0) / Trimestre(1)", 0, int
        )
        col_target = "Perf Week" if perf_choice != 1 else "Perf Quart"

        # ── 3. US: TOP-N INDUSTRIES ───────────────────────────────────────────
        top_n_industries = 5
        if scan_us:
            top_n_industries = get_input_or_default(
                "¿Cuántas de las mejores industrias US analizar?", 5, int
            )

        # ── 4. STRATEGY SETUP ─────────────────────────────────────────────────
        print("\n⚙️  Configuración del Patrón (Setup):")
        sma_weeks       = get_input_or_default("Semanas para la media móvil", 30, int)
        lookback = get_input_or_default("Periodo de la Base — Larga:36w / Media:24w / Corta:16w (Semanas)", 36, int)
        tolerance_input = get_input_or_default("Tolerancia para 'toque' (%)", 1.0, float)
        touches_res     = get_input_or_default("Mínimo de toques a RESISTENCIA", 2, int)
        touches_sup     = get_input_or_default("Mínimo de toques a SOPORTE", 2, int)

        strategy = WeinsteinSetup(
            sma_weeks=sma_weeks,
            lookback_weeks=lookback,
            min_touches=touches_res,
            min_sup_touches=touches_sup,
            touch_tolerance=tolerance_input / 100.0,
        )

        loader   = DataLoader()
        screener = Screener(loader)
        finviz   = FinvizEngine()

        # ── 5. SCAN ───────────────────────────────────────────────────────────
        result_df = pd.DataFrame()

        if scan_us:
            result_df = _run_us_scan(finviz, screener, strategy, top_n_industries, col_target)
        else:
            result_df = _run_international_scan(intl_codes, screener, strategy)

        # ── 6. RESULTS ────────────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("  🚀 RESULTADOS — SETUPS ENCONTRADOS")
        print("=" * 80)

        if result_df.empty:
            print("Ninguna acción cumple los criterios estrictos hoy.")
            clean_data_folder()
            return

        opportunities = result_df.reset_index(drop=True)

        output_folder = "results"
        os.makedirs(output_folder, exist_ok=True)

        # Build display column list — include Sector/Industry only when present
        display_cols = [
            "Market", "Ticker", "Price",
            "Sector", "Industry",
            "Resistance", "Support", "Range_Width_Pct",
            "Distance_to_Breakout",
            "Res_Touches", "Sup_Touches",
            "Vol_Spike_2x", "Vol_4W_Expansion",
        ]
        display_cols = [c for c in display_cols if c in opportunities.columns]

        final_df = opportunities.sort_values(
            by=["Market", "Distance_to_Breakout"], ascending=[True, True]
        )[display_cols]

        print(final_df.to_string(index=False))

        print("\nℹ️  LEYENDA:")
        print("    [Market]              : Código del mercado origen (US / ES / DE / …).")
        print("    [Sector]              : Sector Finviz (US) o sector local (internacional).")
        print("    [Industry]            : Industria Finviz (sólo US).")
        print("    [Resistance/Support]  : Nivel central de la zona clusterizada.")
        print("    [Range_Width_Pct]     : Amplitud de la base (%).")
        print("    [Res/Sup_Touches]     : Toques confirmados en cada zona.")
        print("    [Vol_Spike_2x]        : Semana actual > 2× promedio mes anterior.")
        print("    [Vol_4W_Expansion]    : Bloque 4s actual > 2× línea base 12s.")

        # CSV
        csv_path = os.path.join(output_folder, "weinstein_setups.csv")
        opportunities.to_csv(csv_path, index=False)
        print(f"\n📄 CSV guardado en: '{csv_path}'")

        # ── 7. CHARTS ─────────────────────────────────────────────────────────
        print(f"\n{'─'*80}")
        print(f"  📸 GENERANDO GRÁFICOS EN '{output_folder}/'...")
        print(f"{'─'*80}")

        for _, row in final_df.iterrows():
            ticker = row["Ticker"]
            try:
                daily_df = loader.get_data(ticker, start_date="2021-01-01")
                if daily_df is not None and not daily_df.empty:
                    full_weekly_df = strategy.generate_signals(daily_df)
                    weeks_to_plot  = lookback * 2
                    plot_data = (
                        full_weekly_df.tail(weeks_to_plot).copy()
                        if len(full_weekly_df) > weeks_to_plot
                        else full_weekly_df.copy()
                    )
                    plot_path = os.path.join(output_folder, f"chart_{ticker}.png")
                    plot_weinstein_setup(
                        ticker=ticker,
                        w_df=plot_data,
                        lookback_weeks=lookback,
                        filename=plot_path,
                    )
                    market  = row.get("Market", "?")
                    sector  = row.get("Sector", "")
                    industry = row.get("Industry", "")
                    tag = f"[{market}]"
                    if sector:
                        tag += f" {sector}"
                    if industry:
                        tag += f" / {industry}"
                    print(f"   ✅ {ticker}  {tag}")
                else:
                    print(f"   ⚠️  {ticker}: Sin datos.")
            except Exception as e:
                print(f"   ❌ {ticker}: Error ({e})")

        print(f"\n✨ Proceso finalizado. {len(opportunities)} setup(s) encontrados.")

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido.")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
    finally:
        clean_data_folder()


if __name__ == "__main__":
    main()
