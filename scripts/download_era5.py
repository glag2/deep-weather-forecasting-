"""Scarica il periodo configurato dal Climate Data Store.

Il download e' ripartibile: rilanciando lo script i file gia' presenti e non vuoti
vengono saltati, quindi un'interruzione non obbliga a ricominciare.

Conviene misurare prima di accodare tutto: `--limit 2` scarica un solo mese e riporta
dimensione e tempo reali, da cui si stima il totale senza indovinare.

Uso:
    python scripts/download_era5.py --dry-run
    python scripts/download_era5.py --limit 2
    python scripts/download_era5.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from dwf.config import Config
from dwf.data.download import (
    DownloadOutcome,
    build_payload,
    build_tasks,
    make_client,
    outcomes_to_records,
    run_task,
)
from dwf.tables import DOWNLOADS, cast_to_schema, write_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def human_size(n_bytes: int) -> str:
    valore = float(n_bytes)
    for unita in ("B", "KB", "MB", "GB"):
        if valore < 1024 or unita == "GB":
            return f"{valore:.1f} {unita}"
        valore /= 1024
    return f"{valore:.1f} GB"


def show_plan(config: Config, tasks: list, limite: int | None) -> None:
    print(f"periodo: {config.time.start} .. {config.time.end}")
    print(f"mesi: {len(config.time.months())} | slot attesi: {config.time.n_slots}")
    print(f"area: {config.region.cds_area} | griglia {config.region.n_lat} x {config.region.n_lon}")
    print(f"destinazione: {config.raw_dir}")
    print(f"task totali: {len(tasks)}" + (f" (limitati a {limite})" if limite else ""))
    print()
    esempio = tasks[0]
    payload = build_payload(esempio, config)
    print(f"esempio di richiesta ({esempio.label}):")
    for chiave, valore in payload.items():
        testo = str(valore)
        if len(testo) > 110:
            testo = f"{testo[:107]}... ({len(valore)} elementi)"
        print(f"  {chiave}: {testo}")


def write_manifest(outcomes: list[DownloadOutcome], config: Config) -> Path | None:
    records = outcomes_to_records(outcomes)
    if not records:
        return None
    frame = cast_to_schema(pl.DataFrame(records), DOWNLOADS)
    return write_table(frame, DOWNLOADS, config.tables_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--limit", type=int, default=None, help="Esegui solo i primi N task.")
    parser.add_argument("--dry-run", action="store_true", help="Mostra il piano e termina.")
    parser.add_argument("--overwrite", action="store_true", help="Riscarica anche cio' che c'e'.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config, project_root=PROJECT_ROOT)
    tasks = build_tasks(config)
    show_plan(config, tasks, args.limit)

    if args.dry_run:
        print("\n--dry-run: nessuna richiesta inviata.")
        return

    if args.limit is not None:
        tasks = tasks[: args.limit]

    client = make_client(args.env_file if args.env_file.exists() else None)

    print(f"\navvio di {len(tasks)} richieste (sequenziali: il CDS limita le concorrenti)\n")
    outcomes: list[DownloadOutcome] = []
    scaricati = 0
    for indice, task in enumerate(tasks, start=1):
        esito = run_task(task, config, client, overwrite=args.overwrite)
        outcomes.append(esito)
        if esito.status == "downloaded":
            scaricati += esito.size_bytes
        dettaglio = human_size(esito.size_bytes) if esito.size_bytes else esito.message
        print(
            f"[{indice}/{len(tasks)}] {task.label:28s} {esito.status:11s} "
            f"{dettaglio} ({esito.seconds:.0f} s)"
        )
        if esito.status == "failed" and args.stop_on_error:
            print("\ninterrotto al primo errore, come richiesto.")
            break

    conteggi: dict[str, int] = {}
    for esito in outcomes:
        conteggi[esito.status] = conteggi.get(esito.status, 0) + 1
    riepilogo = ", ".join(f"{stato}={numero}" for stato, numero in sorted(conteggi.items()))
    print(f"\nriepilogo: {riepilogo}")
    print(f"scaricati in questa sessione: {human_size(scaricati)}")

    manifest = write_manifest(outcomes, config)
    if manifest is not None:
        print(f"manifest: {manifest}")

    falliti = [esito for esito in outcomes if esito.status == "failed"]
    if falliti:
        print(f"\n{len(falliti)} richieste fallite. Rilanciare lo script riprova solo quelle.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
