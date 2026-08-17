"""Allinea i dati locali a cio' che ERA5 ha pubblicato.

Scarica i mesi mancanti, riscarica il mese in corso se nel frattempo si e' allungato,
ingerisce cio' che e' cambiato e ricostruisce il catalogo.

Uso:
    python scripts/refresh_data.py --check          # solo diagnosi, nessun download
    python scripts/refresh_data.py --max-months 2   # aggiorna i due mesi piu' urgenti
    python scripts/refresh_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from dwf.config import Config
from dwf.data.freshness import query_availability, summarize_freshness
from dwf.data.refresh import refresh_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument(
        "--check", action="store_true", help="Mostra lo stato senza scaricare nulla."
    )
    parser.add_argument(
        "--max-months", type=int, default=None,
        help="Limita quanti mesi aggiornare in questa esecuzione.",
    )
    parser.add_argument(
        "--no-ingest", action="store_true", help="Scarica soltanto, senza ingerire."
    )
    args = parser.parse_args()

    config = Config.load(args.config, project_root=PROJECT_ROOT)
    disponibilita = query_availability()

    if args.check:
        print(disponibilita.describe())
        tabella = summarize_freshness(config, disponibilita)
        da_fare = tabella.filter("needs_download")
        print(f"\nmesi attesi: {tabella.height} | da aggiornare: {da_fare.height}")
        with pl.Config(tbl_rows=50, tbl_width_chars=140):
            print(da_fare if da_fare.height else tabella.tail(5))
        return

    rapporto = refresh_data(
        config,
        availability=disponibilita,
        max_months=args.max_months,
        ingest=not args.no_ingest,
    )
    if rapporto.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
