import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def plot_weinstein_setup(ticker, w_df, lookback_weeks, filename):
    """
    Genera un gráfico robusto con Matplotlib estándar.
    - Panel 1: Precio (Línea) + Rango (Sombra) + Resistencia + Toques.
    - Panel 2: Volumen.
    """
    print(f"   📊 Generando gráfico para {ticker}...")

    # 1. Limpieza de datos (CRÍTICO para evitar colapso)
    df = w_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Ordenar por fecha por seguridad
    df = df.sort_index()

    # 2. Configurar la figura con 2 paneles (Ratio 3:1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    fig.suptitle(f"{ticker} - Setup Weinstein (Base: {lookback_weeks} semanas)",
                 fontsize=14, fontweight='bold')

    # ==============================================================================
    # PANEL 1: PRECIO
    # ==============================================================================

    # A) Rango Semanal (High-Low) como sombra gris
    ax1.fill_between(df.index, df['Low'], df['High'], color='gray', alpha=0.3, label="Rango Semanal")

    # B) Precio de Cierre (Línea Azul)
    ax1.plot(df.index, df['Close'], color='#0052cc', linewidth=2, label="Cierre")

    # C) Resistencia (Línea Roja Discontinua)
    # Proyectamos la resistencia actual hacia atrás
    current_resistance = df['Major_High'].iloc[-1]
    ax1.axhline(y=current_resistance, color='red', linestyle='--', linewidth=1.5,
                label=f"Resistencia (${current_resistance:.2f})")

    # D) Toques (Triángulos Morados)
    viz_tolerance = 0.02
    threshold = current_resistance * (1 - viz_tolerance)

    # Filtramos los puntos donde el High tocó la zona
    touches = df[df['High'] >= threshold]
    if not touches.empty:
        ax1.scatter(touches.index, touches['High'] * 1.01, color='purple', marker='v', s=100, zorder=5, label="Toque")

    # E) Zona de Base (Fondo Gris en la mitad derecha)
    # Calculamos la fecha de inicio de la base (hace N semanas desde el final)
    if len(df) > lookback_weeks:
        base_start_date = df.index[-lookback_weeks]
        ax1.axvspan(base_start_date, df.index[-1], color='black', alpha=0.05)  # Fondo sutil

    # Texto de información
    last_close = df['Close'].iloc[-1]
    dist_pct = ((current_resistance - last_close) / last_close) * 100
    info_text = (f"Precio: {last_close:.2f}\n"
                 f"Resistencia: {current_resistance:.2f}\n"
                 f"Distancia: {dist_pct:.1f}%")

    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, verticalalignment='top', bbox=props)

    ax1.set_ylabel("Precio")
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ==============================================================================
    # PANEL 2: VOLUMEN
    # ==============================================================================

    # Pintamos barras de volumen
    # Color: Verde si cierra más alto que abre (o cierre > cierre ayer), Rojo si baja.
    # Como es semanal simplificado, usaremos Azul para todo o lógica simple Close > Open.

    # Truco para colores:
    opens = df['Open'] if 'Open' in df.columns else df['Close'].shift(1)
    colors = np.where(df['Close'] >= opens, 'green', 'red')

    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7, width=3)  # width en días aprox

    # Media de volumen (opcional)
    if 'Vol_Avg_10W' in df.columns:
        ax2.plot(df.index, df['Vol_Avg_10W'], color='orange', linestyle='-', linewidth=1, label="Media Vol 10s")

    ax2.set_ylabel("Volumen")
    ax2.grid(True, linestyle=':', alpha=0.6)

    # ==============================================================================
    # FORMATO FECHAS (Crucial para que no colapse)
    # ==============================================================================

    # Formatear eje X
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Una marca por mes (o ajustar según zoom)
    plt.xticks(rotation=45)

    # Ajustar márgenes
    plt.tight_layout()

    # Guardar
    plt.savefig(filename, dpi=100)
    plt.close(fig)