import sys
import random
import os
from src.data_loader import DataLoader
from src.screener import Screener
from src.finviz_engine import FinvizEngine
from src.utils import get_input_or_default, clean_data_folder
from strategies.weinstein_setup import WeinsteinSetup
from src.plotter import plot_weinstein_setup


def main():
    print("\n" + "=" * 80)
    print(" 🎯 VEKTOR: WEINSTEIN SETUP SCREENER (PRE-BREAKOUT)")
    print("=" * 80)

    try:
        # 1. CONFIGURACIÓN
        top_n_sectors = get_input_or_default("Número de mejores sectores a analizar", 3, int)
        perf_sector = get_input_or_default("Que tipo de performance sectorial quieres usar (Week(0),Quarter(1))", 0, int)
        if perf_sector == 0: col_target = 'Perf Week'
        elif perf_sector == 1: col_target = 'Perf Quart'

        print("\n⚙️  Configuración del Patrón (Setup):")
        sma_weeks = get_input_or_default("Numero de semanas para calcular la media movil", 30, int)
        lookback = get_input_or_default("Periodo de la Base (Semanas)", 30, int)
        prox_input = get_input_or_default("Distancia actual al techo (%)", 5.0, float)
        tolerance_input = get_input_or_default("Tolerancia para considerar 'toque' (%)", 2.0, float)
        touches = get_input_or_default("Mínimo de toques a resistencia", 2, int)

        # 2. INICIALIZAR ESTRATEGIA
        strategy = WeinsteinSetup(
            sma_weeks=sma_weeks,
            lookback_weeks=lookback,
            proximity_pct=prox_input / 100.0,
            min_touches=touches,
            touch_tolerance=tolerance_input / 100.0
        )

        # 3. OBTENCIÓN DE CANDIDATOS
        loader = DataLoader()
        finviz = FinvizEngine()

        top_sectors = finviz.get_top_sectors(top_n=top_n_sectors, col_target = col_target)
        if not top_sectors: return

        print(f"\n✅ Sectores Líderes: {top_sectors}")

        candidates = []
        for sector in top_sectors:
            tickers = finviz.get_tickers_in_sector(sector)
            print(f"   -> {sector}: {len(tickers)} acciones encontradas")
            candidates.extend(tickers)

        candidates = list(set(candidates))
        print(f"\n🎯 Total candidatos a procesar: {len(candidates)}")

        if not candidates: return

        # 4. ESCANEO
        screener = Screener(loader)
        opportunities = screener.scan(strategy, candidates)

        # 5. RESULTADOS
        print("\n" + "=" * 80)
        print(f" 🚀 RESULTADOS: ACCIONES EN FASE DE 'CARGA' (SETUP)")
        print("=" * 80)

        if not opportunities.empty:
            # --- CREAR CARPETA RESULTS ---
            output_folder = "results"
            os.makedirs(output_folder, exist_ok=True)
            # -----------------------------

            cols = ['Ticker', 'Price', 'Distance_to_Breakout', 'Touch_Count', 'Vol_Spike_2x', 'Vol_4W_Expansion']
            final_df = opportunities.sort_values(by="Distance_to_Breakout", ascending=True)[cols]
            print(final_df.to_string(index=False))

            print("\nℹ️  INFO DE VOLUMEN (1 = CUMPLE):")
            print("    [Vol_Spike_2x]    : Volumen Semanal > 2x Promedio mes anterior.")
            print("    [Vol_4W_Expansion]: Promedio Mensual actual > 2x Promedio previo.")

            # Guardar CSV
            csv_path = os.path.join(output_folder, "weinstein_setups.csv")
            opportunities.to_csv(csv_path, index=False)
            print(f"\n📄 CSV guardado en: '{csv_path}'")

            # 6. GENERACIÓN DE GRÁFICOS
            print("\n" + "-" * 80)
            print(f" 📸 GENERANDO GRÁFICOS EN '{output_folder}/'...")
            print("-" * 80)

            for ticker in final_df['Ticker'].tolist():
                try:
                    # Descargar historia suficiente (desde 2021)
                    daily_df = loader.get_data(ticker, start_date="2021-01-01")

                    if daily_df is not None and not daily_df.empty:
                        # Convertir a semanal
                        full_weekly_df = strategy.generate_signals(daily_df)

                        # Recortar datos: Contexto (Lookback) + Base (Lookback)
                        weeks_to_plot = lookback * 2

                        # Usamos .copy() para romper referencias y evitar bugs gráficos
                        if len(full_weekly_df) > weeks_to_plot:
                            plot_data = full_weekly_df.tail(weeks_to_plot).copy()
                        else:
                            plot_data = full_weekly_df.copy()

                        # Ruta del gráfico
                        plot_path = os.path.join(output_folder, f"chart_{ticker}.png")

                        plot_weinstein_setup(
                            ticker=ticker,
                            w_df=plot_data,
                            lookback_weeks=lookback,
                            filename=plot_path
                        )
                        print(f"   ✅ {ticker}")
                    else:
                        print(f"   ⚠️ {ticker}: Sin datos.")

                except Exception as e:
                    print(f"   ❌ {ticker}: Error ({e})")

            print(f"\n✨ Proceso finalizado.")

        else:
            print("Ninguna acción cumple los criterios estrictos hoy.")

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