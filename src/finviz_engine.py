# --- CORRECCIÓN AQUÍ: Importamos Performance en lugar de Overview ---
from finvizfinance.group.performance import Performance
from finvizfinance.screener.overview import Overview as ScreenerOverview


class FinvizEngine:
    def __init__(self):
        pass

    def get_top_sectors(self, top_n=3, col_target="Perf Week"):
        """
        Obtiene los sectores con mejor rendimiento trimestral usando la vista de Performance.
        """
        print("📊 Consultando sectores en Finviz (Vista Performance)...")
        try:
            # --- CORRECCIÓN AQUÍ: Usamos la clase Performance ---
            fgro = Performance()

            # Esto descarga la tabla de rendimiento (Performance) en lugar de la general
            df_sectors = fgro.screener_view(group='Sector')


            # Verificar si la columna existe antes de procesar
            if col_target not in df_sectors.columns:
                print(
                    f"⚠️ La columna '{col_target}' no se encontró. Columnas disponibles: {df_sectors.columns.tolist()}")
                # Intentamos usar 'Perf Month' o 'Perf Year' como respaldo si falla
                if 'Perf Month' in df_sectors.columns:
                    col_target = 'Perf Month'
                    print("⚠️ Usando 'Perf Month' como respaldo.")
                else:
                    return []

            # Función para limpiar porcentajes
            def clean_pct(x):
                if isinstance(x, str):
                    return float(x.strip('%')) / 100
                return x

            df_sectors[col_target] = df_sectors[col_target].apply(clean_pct)

            # Ordenamos
            df_sorted = df_sectors.sort_values(by=col_target, ascending=False)

            # Mostramos un pequeño resumen en consola
            print(f"\n🏆 Top Sectores ({col_target}):")
            print(df_sorted[['Name', col_target]].head(top_n))

            return df_sorted['Name'].head(top_n).tolist()

        except Exception as e:
            print(f"❌ Error conectando con Finviz (Sectores): {e}")
            return []

    def get_tickers_in_sector(self, sector_name):
        # (Este método se mantiene igual que antes, funciona bien)
        print(f"   📥 Descargando lista de acciones para: {sector_name}...")
        try:
            filters_dict = {
                'Sector': sector_name,
                'Market Cap.': '+Small (over $300mln)',
                'Average Volume': 'Over 300K',
            }

            fscreen = ScreenerOverview()
            fscreen.set_filter(filters_dict=filters_dict)
            df_results = fscreen.screener_view()

            if df_results.empty:
                return []

            return df_results['Ticker'].tolist()

        except Exception as e:
            print(f"❌ Error obteniendo tickers de {sector_name}: {e}")
            return []