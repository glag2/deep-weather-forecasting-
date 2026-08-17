"""Converte i GRIB scaricati in Zarr e aggiorna il catalogo Parquet.

Ingerisce solo i mesi i cui GRIB sono presenti, quindi puo' girare mentre il download
prosegue: i mesi non ancora arrivati restano NaN nello store e vengono marcati come non
utilizzabili nel catalogo. Rilanciandolo si aggiungono i mesi nuovi.

Uso:
    python scripts/ingest_era5.py --list
    python scripts/ingest_era5.py --month 2024-01
    python scripts/ingest_era5.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from dwf.config import Config
from dwf.data.ingest import (
    IngestError,
    available_months,
    build_catalogue,
    build_folds_table,
    ingest_month,
    ingest_static,
    initialize_store,
    write_catalogue,
    write_folds_table,
    write_variables_table,
)
from dwf.slots import parse_month
from dwf.tables import SLOT_STATS, write_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def month_argument(value: str) -> tuple[int, int]:
    try:
        return parse_month(value)
    except ValueError as errore:
        raise argparse.ArgumentTypeError(str(errore)) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--month", type=month_argument, action="append", metavar="YYYY-MM")
    parser.add_argument("--list", action="store_true", help="Elenca i mesi disponibili.")
    parser.add_argument(
        "--recreate-store",
        action="store_true",
        help="Ricrea lo store Zarr da zero, perdendo i mesi gia' ingeriti.",
    )
    args = parser.parse_args()

    config = Config.load(args.config, project_root=PROJECT_ROOT)
    disponibili = available_months(config)
    attesi = config.time.months()

    print(f"periodo configurato: {config.time.start} .. {config.time.end}")
    print(f"mesi attesi: {len(attesi)} | GRIB presenti: {len(disponibili)}")
    if args.list:
        mancanti = [mese for mese in attesi if mese not in disponibili]
        print("presenti:", ", ".join(f"{y}-{m:02d}" for y, m in disponibili) or "nessuno")
        print("mancanti:", ", ".join(f"{y}-{m:02d}" for y, m in mancanti) or "nessuno")
        return

    da_fare = args.month or disponibili
    non_disponibili = [mese for mese in da_fare if mese not in disponibili]
    if non_disponibili:
        etichette = ", ".join(f"{y}-{m:02d}" for y, m in non_disponibili)
        print(f"GRIB assenti per: {etichette}")
        raise SystemExit(1)
    if not da_fare:
        print("Nessun mese da ingerire: eseguire prima scripts/download_era5.py")
        raise SystemExit(1)

    store = initialize_store(config, overwrite=args.recreate_store)
    print(f"store: {store}")

    if (config.raw_dir / "static.grib").exists():
        print(f"statici: {ingest_static(config)}")

    tutte_le_stats: list[pl.DataFrame] = []
    for indice, (year, month) in enumerate(da_fare, start=1):
        avvio = time.perf_counter()
        try:
            esito = ingest_month(config, year, month)
        except IngestError as errore:
            print(f"[{indice}/{len(da_fare)}] {year}-{month:02d} FALLITO: {errore}")
            raise SystemExit(1) from None
        tutte_le_stats.append(esito.stats)
        nan_totali = int(esito.stats.get_column("n_nan").sum())
        print(
            f"[{indice}/{len(da_fare)}] {year}-{month:02d} ok: {esito.n_slots} slot, "
            f"{len(esito.variables)} variabili, NaN={nan_totali} "
            f"({time.perf_counter() - avvio:.1f} s)"
        )

    stats = pl.concat(tutte_le_stats) if tutte_le_stats else None
    if stats is not None:
        percorso = write_table(stats, SLOT_STATS, config.tables_dir)
        print(f"controlli qualita': {percorso}")

    ingeriti = set(da_fare)
    catalogo = build_catalogue(config, ingeriti, stats)
    print(f"catalogo: {write_catalogue(config, catalogo)}")

    utilizzabili = catalogo.get_column("usable").to_numpy()
    folds = build_folds_table(config, utilizzabili)
    print(f"fold: {write_folds_table(config, folds)}")
    print(f"variabili: {write_variables_table(config)}")

    print(f"\nslot utilizzabili: {int(utilizzabili.sum())} su {len(utilizzabili)}")
    riepilogo = (
        folds.filter(pl.col("is_sample_start"))
        .group_by("fold", "split")
        .len()
        .sort("fold", "split")
    )
    print("\ncampioni ammessi per fold:")
    print(riepilogo)


if __name__ == "__main__":
    main()
