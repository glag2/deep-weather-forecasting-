"""Allinea i dati locali a cio' che ERA5 ha pubblicato.

E' il passo che il notebook di previsione esegue all'avvio: scarica i mesi mancanti,
riscarica il mese in corso se nel frattempo si e' allungato, ingerisce cio' che e'
cambiato e ricostruisce il catalogo degli slot.

La sottigliezza sta tutta nel mese in corso. Non e' un mese rotto: e' un mese che
finisce prima perche' i giorni successivi non sono ancora accaduti. Va quindi
riscaricato quando cresce, ma non a ogni esecuzione, e i suoi slot mancanti devono
restare marcati come inutilizzabili senza invalidare quelli che ci sono.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from dwf.data.download import (
    build_tasks,
    make_client,
    outcomes_to_records,
    run_tasks,
)
from dwf.data.freshness import (
    Availability,
    pending_months,
    query_availability,
    read_manifest,
)
from dwf.data.ingest import (
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
from dwf.tables import DOWNLOADS, SLOT_STATS, cast_to_schema, read_table, write_table

if TYPE_CHECKING:  # pragma: no cover - solo per i tipi
    from dwf.config import Config
    from dwf.data.download import RetrieveClient


@dataclass(slots=True)
class RefreshReport:
    """Che cosa e' stato fatto, per poterlo raccontare a chi guarda il notebook."""

    availability: Availability
    months_pending: tuple[str, ...] = ()
    months_downloaded: tuple[str, ...] = ()
    months_ingested: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    usable_slots: int = 0
    total_slots: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def up_to_date(self) -> bool:
        return not self.months_pending and not self.failures

    def describe(self) -> str:
        righe = [self.availability.describe()]
        if not self.months_pending:
            righe.append("Dati gia' allineati: nulla da scaricare.")
        else:
            righe.append(f"Mesi da aggiornare: {', '.join(self.months_pending)}")
        if self.months_downloaded:
            righe.append(f"Scaricati: {', '.join(self.months_downloaded)}")
        if self.months_ingested:
            righe.append(f"Ingeriti: {', '.join(self.months_ingested)}")
        if self.failures:
            righe.append(f"Falliti: {', '.join(self.failures)}")
        if self.total_slots:
            righe.append(f"Slot utilizzabili: {self.usable_slots} su {self.total_slots}")
        righe.extend(self.notes)
        return "\n".join(righe)


def _merge_stats(config: Config, nuove: list[pl.DataFrame]) -> pl.DataFrame | None:
    """Unisce le statistiche dei mesi appena ingeriti a quelle gia' registrate.

    Riscrivere solo i mesi nuovi perderebbe i precedenti; re-ingerire tutto per avere
    la tabella completa costerebbe minuti per nulla. Le righe vecchie degli stessi slot
    vengono sostituite, perche' un mese parziale riscaricato ha statistiche diverse.
    """
    if not nuove:
        return None
    aggiornate = pl.concat(nuove)
    percorso = config.tables_dir / SLOT_STATS.filename
    if percorso.exists():
        precedenti = read_table(SLOT_STATS, config.tables_dir)
        sostituiti = aggiornate.get_column("slot_index").unique().to_list()
        precedenti = precedenti.filter(~pl.col("slot_index").is_in(sostituiti))
        aggiornate = pl.concat([precedenti, aggiornate])
    return cast_to_schema(aggiornate.sort("slot_index", "variable"), SLOT_STATS)


def _record_downloads(config: Config, outcomes) -> None:
    """Accoda gli esiti al manifest, conservando lo storico."""
    if not outcomes:
        return
    nuovi = cast_to_schema(pl.DataFrame(outcomes_to_records(outcomes)), DOWNLOADS)
    precedenti = read_manifest(config)
    if precedenti is not None:
        nuovi = pl.concat([precedenti, nuovi])
    write_table(nuovi, DOWNLOADS, config.tables_dir)


def refresh_data(
    config: Config,
    *,
    availability: Availability | None = None,
    client: RetrieveClient | None = None,
    download: bool = True,
    ingest: bool = True,
    max_months: int | None = None,
    verbose: bool = True,
) -> RefreshReport:
    """Porta i dati locali allo stato piu' recente pubblicato.

    Con `download=False` si ottiene solo la diagnosi, senza traffico di rete: utile per
    vedere che cosa manca prima di impegnare ore di scaricamento.
    """
    avvio = time.perf_counter()
    disponibilita = availability or query_availability()
    da_fare = pending_months(config, disponibilita)
    rapporto = RefreshReport(
        availability=disponibilita,
        months_pending=tuple(stato.label for stato in da_fare),
    )

    def racconta(messaggio: str) -> None:
        if verbose:
            print(messaggio)

    racconta(disponibilita.describe())
    if not disponibilita.is_authoritative:
        rapporto.notes.append(
            "Frontiera stimata: se il catalogo torna raggiungibile potrebbero "
            "risultare disponibili altri giorni."
        )

    if not da_fare:
        racconta("Nulla da scaricare: i dati locali coprono tutto il pubblicato.")
    else:
        racconta(f"Mesi da aggiornare: {len(da_fare)}")
        for stato in da_fare:
            racconta(f"  {stato.label}: {stato.reason}")

    if not download or not da_fare:
        rapporto.seconds = time.perf_counter() - avvio
        return rapporto

    selezionati = da_fare[:max_months] if max_months else da_fare
    mesi_richiesti = {(stato.year, stato.month) for stato in selezionati}
    # I task nascono dalla frontiera, quindi il mese in corso viene chiesto solo fino
    # ai giorni pubblicati; poi si tengono i mesi che servono davvero.
    tutti = build_tasks(config, until=disponibilita.end_date)
    task = [
        attivita
        for attivita in tutti
        if attivita.year is None or (attivita.year, attivita.month) in mesi_richiesti
    ]
    statici = config.raw_dir / "static.grib"
    if statici.exists() and statici.stat().st_size > 0:
        task = [attivita for attivita in task if attivita.kind != "static"]

    # Un mese parziale gia' presente va sovrascritto: il file vecchio esiste ed e' non
    # vuoto, quindi la ripresa da interruzione lo salterebbe.
    da_sovrascrivere = {
        (stato.year, stato.month) for stato in selezionati if not stato.missing_kinds
    }
    racconta(f"\nrichieste al CDS: {len(task)}")
    cliente = client or make_client(config.project_root / ".env")

    esiti = []
    for attivita in task:
        sovrascrivi = (attivita.year, attivita.month) in da_sovrascrivere
        esito = run_tasks([attivita], config, cliente, overwrite=sovrascrivi)[0]
        esiti.append(esito)
        racconta(
            f"  {attivita.label}: {esito.status} "
            f"({esito.size_bytes / 1e6:.1f} MB, {esito.seconds:.0f} s)"
        )

    _record_downloads(config, esiti)
    riusciti = {
        (esito.task.year, esito.task.month)
        for esito in esiti
        if esito.status in {"downloaded", "skipped"} and esito.task.year is not None
    }
    rapporto.failures = tuple(
        esito.task.label for esito in esiti if esito.status == "failed"
    )
    rapporto.months_downloaded = tuple(
        f"{anno:04d}-{mese:02d}" for anno, mese in sorted(riusciti)
    )

    if not ingest:
        rapporto.seconds = time.perf_counter() - avvio
        return rapporto

    presenti = set(available_months(config))
    da_ingerire = sorted(riusciti & presenti)
    if not da_ingerire:
        rapporto.seconds = time.perf_counter() - avvio
        return rapporto

    initialize_store(config)
    if statici.exists():
        ingest_static(config)

    racconta(f"\ningestione di {len(da_ingerire)} mesi")
    statistiche = []
    ingeriti = []
    for anno, mese in da_ingerire:
        esito_ingestione = ingest_month(config, anno, mese)
        statistiche.append(esito_ingestione.stats)
        ingeriti.append(f"{anno:04d}-{mese:02d}")
        racconta(f"  {anno:04d}-{mese:02d}: {esito_ingestione.n_slots} slot")
    rapporto.months_ingested = tuple(ingeriti)

    stats = _merge_stats(config, statistiche)
    if stats is not None:
        write_table(stats, SLOT_STATS, config.tables_dir)

    # Il catalogo si ricostruisce su **tutti** i mesi presenti, non solo su quelli
    # appena ingeriti, altrimenti i precedenti risulterebbero assenti.
    catalogo = build_catalogue(config, set(available_months(config)), stats)
    write_catalogue(config, catalogo)
    utilizzabili = catalogo.get_column("usable").to_numpy()
    write_folds_table(config, build_folds_table(config, utilizzabili))
    write_variables_table(config)

    rapporto.usable_slots = int(utilizzabili.sum())
    rapporto.total_slots = int(utilizzabili.size)
    rapporto.seconds = time.perf_counter() - avvio
    racconta(
        f"\nslot utilizzabili: {rapporto.usable_slots} su {rapporto.total_slots} "
        f"({rapporto.seconds:.0f} s)"
    )
    return rapporto


__all__ = ["RefreshReport", "refresh_data"]
