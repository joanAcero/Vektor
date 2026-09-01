"""
industry_rotation.py
---------------------
O'Neil-style industry leadership: el 20% de las industrias Finviz con mejor
Mansfield RS, DENTRO de los sectores que ya son líderes Weinstein
(src/rotation.py::sector_rotation, filtrado por Etapa 2 + RS>0).

RS AQUÍ ES CONTRA SPY, NO CONTRA EL SECTOR PADRE -- confirmado explícitamente
en la conversación de diseño de este panel: "compáralas con el mercado
general el SPY, no con su sector". Es una decisión deliberada, no un
descuido: compara la industria contra el mismo benchmark que el resto de
VEKTOR usa para todo lo demás (acciones, sectores), así una industria con
RS=+5 aquí es comparable directamente a un sector con RS=+5 en el panel
anterior.

CÓMO SE OBTIENE EL PRECIO DE UNA INDUSTRIA -- no existe un ETF Finviz-industry
1:1 para la mayoría de las ~150 industrias, así que el precio semanal de cada
industria es un ÍNDICE SINTÉTICO: la media equal-weight del precio de CADA
ticker que Finviz lista en esa industria (FinvizEngine.get_ticker_details_in_industry).
Esto es deliberadamente costoso: se descarga vía DataLoader cada ticker
constituyente de cada industria candidata, no una muestra. Confirmado en la
conversación de diseño que el coste (minutos en frío, caché de DataLoader de
por medio a partir de la 2a carga) es aceptable frente a un proxy más barato
pero menos fiel.

QUÉ INDUSTRIAS SON "CANDIDATAS" -- FinvizEngine no tiene un mapeo
industria->sector independiente. Se deriva de get_ticker_details_in_sector():
para cada sector líder, sus constituyentes ya traen la columna Industry: los
valores únicos de esa columna son las industrias candidatas de ese sector.
Ninguna lista se mantiene a mano.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.benchmarks import benchmark_for, get_weekly_close, mansfield_rs

log = logging.getLogger(__name__)

TOP_PERCENT = 0.20
MIN_WEEKS = 52 + 30  # mismo umbral que Mansfield RS + MA30 necesitan para no ser ruido


GICS_TO_FINVIZ_SECTOR: dict[str, str] = {
    "Materials": "Basic Materials",
    "Communication Services": "Communication Services",
    "Energy": "Energy",
    "Financials": "Financial",
    "Industrials": "Industrials",
    "Technology": "Technology",
    "Consumer Staples": "Consumer Defensive",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Health Care": "Healthcare",
    "Consumer Discretionary": "Consumer Cyclical",
}


def _finviz_sector_name(gics_sector: str) -> str | None:
    """None si no hay traducción conocida -- el llamador debe saltar ese
    sector con un warning, no adivinar."""
    return GICS_TO_FINVIZ_SECTOR.get(gics_sector)

def _synthetic_weekly_close(loader, tickers: list[str], start_date: str) -> pd.Series | None:
    """
    Media equal-weight del precio semanal de cada ticker, reindexada a la
    unión de sus calendarios y promediada solo sobre los tickers con dato
    esa semana (ffill no se usa: una industria con datos parciales una
    semana concreta promedia lo que tiene, no rellena con precios viejos).
    """
    series = []
    for t in tickers:
        wk = get_weekly_close(loader, t, start_date=start_date)
        if wk is not None and not wk.empty:
            series.append(wk.rename(t))
    if not series:
        return None
    combined = pd.concat(series, axis=1)
    # Normalizado a base 100 en el primer punto de cada serie individual antes
    # de promediar -- si no, una acción de $900 (ej. una de precio alto)
    # domina la media frente a una de $20, aunque pesen igual en la industria.
    normalised = combined.apply(lambda s: (s / s.dropna().iloc[0]) * 100.0
                                if s.dropna().size else s)
    return normalised.mean(axis=1, skipna=True).dropna()


def _leader_sector_data(finviz_engine, leader_sector_names: list[str],
                        on_progress=None) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Una sola llamada a Finviz por sector líder (antes eran dos: una para
    industrias candidatas y otra para el mapeo industria->sector -- mismo
    dato, pedido dos veces, doblando 3-6 minutos de espera de Finviz por
    nada). Devuelve (industry_tickers, industry_to_sector) juntos.
    """
    industry_tickers: dict[str, list[str]] = {}
    industry_to_sector: dict[str, str] = {}

    for done, gics_sector in enumerate(leader_sector_names, start=1):
        if on_progress is not None:
            on_progress("fetching_sectors", done, len(leader_sector_names), gics_sector)

        finviz_sector = _finviz_sector_name(gics_sector)
        if finviz_sector is None:
            log.warning("No Finviz sector mapping for GICS sector %r; skipped.", gics_sector)
            continue
        df = finviz_engine.get_ticker_details_in_sector(finviz_sector)
        if df.empty:
            log.warning("Finviz returned no constituents for sector %r "
                       "(GICS: %r); its industries are skipped this run.",
                       finviz_sector, gics_sector)
            continue
        for industry, group in df.groupby("Industry"):
            if not industry:
                continue
            tickers = [t for t in group["Ticker"].tolist() if t]
            industry_tickers.setdefault(industry, []).extend(tickers)
            industry_to_sector.setdefault(industry, gics_sector)

    industry_tickers = {ind: list(dict.fromkeys(tks)) for ind, tks in industry_tickers.items()}
    return industry_tickers, industry_to_sector


def industry_leaders(loader, finviz_engine, sector_rows: list[dict], *,
                     start_date: str = "2018-01-01",
                     on_progress=None) -> dict:
    """
    on_progress: callback opcional, llamado en DOS fases (para que la
    barra de progreso no esté "muda" durante la fase lenta):
      on_progress("fetching_sectors", done, total_sectores, nombre_sector)
      on_progress("scoring_industries", done, total_industrias, nombre_industria)
    """
    leader_sectors = [s["sector"] for s in sector_rows if s.get("is_leader")]
    if not leader_sectors:
        return {"industries": [], "total_candidates": 0}

    industry_tickers, industry_to_sector = _leader_sector_data(
        finviz_engine, leader_sectors, on_progress=on_progress)
    total = len(industry_tickers)

    benchmark_symbol = benchmark_for("US")
    index_wk = get_weekly_close(loader, benchmark_symbol, start_date=start_date)
    if index_wk is None:
        log.error("Could not load benchmark %s for industry leaders.", benchmark_symbol)
        return {"industries": [], "total_candidates": 0}

    scored: list[dict] = []
    for done, (industry, tickers) in enumerate(industry_tickers.items(), start=1):
        synth = _synthetic_weekly_close(loader, tickers, start_date)
        if synth is not None and len(synth) >= MIN_WEEKS:
            index_aligned = index_wk.reindex(synth.index).dropna()
            mrs_series = mansfield_rs(synth, index_aligned)
            if not mrs_series.empty and pd.notna(mrs_series.iloc[-1]):
                scored.append({
                    "industry": industry,
                    "sector": industry_to_sector.get(industry, ""),
                    "mansfield_rs": round(float(mrs_series.iloc[-1]), 2),
                    "n_constituents": len(tickers),
                })
            else:
                log.warning("Industry %r produced no valid Mansfield RS; skipping.", industry)
        else:
            log.warning("Industry %r has insufficient synthetic history "
                       "(%d tickers resolved); skipping.", industry, len(tickers))

        if on_progress is not None:
            on_progress("scoring_industries", done, total, industry)

    scored.sort(key=lambda r: r["mansfield_rs"], reverse=True)
    cutoff = max(1, int(np.ceil(len(scored) * TOP_PERCENT))) if scored else 0
    return {"industries": scored[:cutoff], "total_candidates": len(scored)}