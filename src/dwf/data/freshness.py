"""Quanto sono aggiornati i dati locali, e che cosa manca per allinearli.

Il problema che risolve questo modulo e' che **il mese corrente e' sempre parziale**.
ERA5 non e' una previsione ma una rianalisi: viene pubblicata con alcuni giorni di
ritardo, quindi in qualunque momento l'ultimo mese disponibile si ferma a meta'. Un
file che copre mezzo mese non e' corrotto, e non va nemmeno riscaricato ogni volta:
va riscaricato **solo quando ERA5 ha pubblicato altri giorni**.

Per saperlo serve la frontiera di pubblicazione. Invece di assumere una latenza fissa
la si legge dal catalogo STAC del CDS, che la dichiara in
``extent.temporal.interval``. Se il catalogo non e' raggiungibile si ripiega su una
stima prudente, dichiarandolo.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from dwf.data.download import DATASET, days_to_request
from dwf.tables import DOWNLOADS, cast_to_schema

if TYPE_CHECKING:  # pragma: no cover - solo per i tipi
    from dwf.config import Config

CATALOGUE_URL = "https://cds.climate.copernicus.eu/api/catalogue/v1/collections/{collection}"

# Usata solo se il catalogo non risponde. Deliberatamente prudente: chiedere un giorno
# non ancora pubblicato fa fallire l'intera richiesta, mentre chiederne uno in meno
# costa solo un aggiornamento rimandato.
FALLBACK_LATENCY_DAYS = 8


class FreshnessError(RuntimeError):
    """Impossibile determinare lo stato dei dati."""


@dataclass(frozen=True, slots=True)
class Availability:
    """Fino a quando ERA5 e' pubblicato, e come lo si e' saputo."""

    end_date: date
    source: str
    checked_at: datetime

    @property
    def is_authoritative(self) -> bool:
        return self.source == "catalogue"

    def describe(self) -> str:
        origine = (
            "dichiarata dal catalogo CDS"
            if self.is_authoritative
            else f"stimata ({FALLBACK_LATENCY_DAYS} giorni di ritardo), catalogo non raggiungibile"
        )
        return f"ERA5 pubblicato fino al {self.end_date.isoformat()} ({origine})"


def _parse_catalogue(payload: dict[str, Any]) -> date:
    try:
        intervallo = payload["extent"]["temporal"]["interval"][0]
        fine = intervallo[1]
    except (KeyError, IndexError, TypeError) as errore:
        raise FreshnessError(
            "Il catalogo CDS non dichiara extent.temporal.interval nella forma attesa"
        ) from errore
    if fine is None:
        raise FreshnessError("Il catalogo CDS dichiara un intervallo temporale aperto")
    # Il campo e' ISO 8601 con offset esplicito; `fromisoformat` lo accetta da 3.11.
    return datetime.fromisoformat(fine).date()


def query_availability(
    *,
    collection: str = DATASET,
    timeout: float = 30.0,
    today: date | None = None,
    opener: Any = None,
) -> Availability:
    """Frontiera di pubblicazione di ERA5.

    `opener` esiste per i test: permette di sostituire la chiamata di rete senza
    toccare la logica di interpretazione e di ripiego.
    """
    adesso = datetime.now(UTC)
    oggi = today or adesso.date()
    richiedi = opener or urllib.request.urlopen
    url = CATALOGUE_URL.format(collection=collection)
    try:
        with richiedi(url, timeout=timeout) as risposta:
            payload = json.load(risposta)
        return Availability(_parse_catalogue(payload), "catalogue", adesso)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, FreshnessError):
        # Nessuna eccezione propagata: l'assenza di rete non deve impedire di lavorare
        # sui dati gia' scaricati.
        from datetime import timedelta

        return Availability(oggi - timedelta(days=FALLBACK_LATENCY_DAYS), "estimated", adesso)


@dataclass(frozen=True, slots=True)
class MonthStatus:
    """Stato di un mese: quanto e' disponibile, quanto se ne ha, che cosa fare."""

    year: int
    month: int
    days_available: int
    days_downloaded: int
    missing_kinds: tuple[str, ...]
    partial: bool

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def needs_download(self) -> bool:
        # Manca un file, oppure ne esiste uno che copre meno giorni di quelli ora
        # pubblicati: in entrambi i casi c'e' qualcosa da scaricare.
        return bool(self.missing_kinds) or self.days_downloaded < self.days_available

    @property
    def reason(self) -> str:
        if self.missing_kinds:
            return f"mancano i file: {', '.join(self.missing_kinds)}"
        if self.days_downloaded < self.days_available:
            return (
                f"parziale: {self.days_downloaded} giorni scaricati su "
                f"{self.days_available} ora pubblicati"
            )
        return "completo"


def read_manifest(config: Config) -> pl.DataFrame | None:
    """Manifest dei download, tollerante verso i file scritti da versioni precedenti.

    Le colonne sui giorni sono state aggiunte dopo i primi scaricamenti: un manifest
    piu' vecchio non le ha. Farlo fallire costringerebbe a cancellare lo storico, quindi
    le colonne assenti vengono aggiunte vuote e il file si riallinea alla prima scrittura.
    """
    percorso = config.tables_dir / DOWNLOADS.filename
    if not percorso.exists():
        return None
    grezzo = pl.read_parquet(percorso)
    mancanti = [nome for nome in DOWNLOADS.schema if nome not in grezzo.columns]
    if mancanti:
        grezzo = grezzo.with_columns(
            [pl.lit(None).alias(nome) for nome in mancanti]
        )
    return cast_to_schema(grezzo, DOWNLOADS)


def _downloaded_days(config: Config) -> dict[tuple[int, int, str], int]:
    """Giorni coperti da ogni file secondo il manifest, per mese e famiglia.

    Si tiene la riga piu' recente per chiave: un mese parziale riscaricato piu' volte
    lascia piu' righe, e conta l'ultima.
    """
    manifest = read_manifest(config)
    if manifest is None or manifest.height == 0:
        return {}
    recenti = (
        manifest.filter(
            (pl.col("status") == "downloaded") & pl.col("year").is_not_null()
        )
        .sort("recorded_at")
        .group_by(["year", "month", "kind"])
        .last()
    )
    coperti: dict[tuple[int, int, str], int] = {}
    for riga in recenti.iter_rows(named=True):
        if riga["n_days"] is None:
            # Riga scritta prima che i giorni venissero registrati: il chiamante la
            # tratta come copertura completa, cosi' non si riscarica alla cieca.
            continue
        coperti[(int(riga["year"]), int(riga["month"]), riga["kind"])] = int(riga["n_days"])
    return coperti


def expected_kinds(config: Config) -> tuple[str, ...]:
    generi = []
    if config.variables.instantaneous:
        generi.append("instantaneous")
    if config.variables.accumulated:
        generi.append("accumulated")
    return tuple(generi)


def month_statuses(config: Config, availability: Availability) -> list[MonthStatus]:
    """Stato di ogni mese del periodo configurato, alla luce di cio' che e' pubblicato."""
    manifest = _downloaded_days(config)
    generi = expected_kinds(config)
    stati: list[MonthStatus] = []

    for anno, mese in config.time.months():
        pubblicati = days_to_request(config, anno, mese, availability.end_date)
        if not pubblicati:
            # Mese interamente nel futuro: non c'e' nulla da attendersi.
            continue
        nel_periodo = days_to_request(config, anno, mese)

        mancanti = []
        coperti = []
        for genere in generi:
            percorso = config.raw_dir / f"{genere}_{anno:04d}-{mese:02d}.grib"
            if not percorso.exists() or percorso.stat().st_size == 0:
                mancanti.append(genere)
            else:
                # Un file presente ma non nel manifest e' stato scaricato prima che il
                # conteggio dei giorni esistesse: lo si considera coperto per intero,
                # altrimenti verrebbe riscaricato senza motivo.
                coperti.append(manifest.get((anno, mese, genere), len(pubblicati)))

        stati.append(
            MonthStatus(
                year=anno,
                month=mese,
                days_available=len(pubblicati),
                days_downloaded=min(coperti) if coperti else 0,
                missing_kinds=tuple(mancanti),
                partial=len(pubblicati) < len(nel_periodo),
            )
        )
    return stati


def pending_months(config: Config, availability: Availability) -> list[MonthStatus]:
    return [stato for stato in month_statuses(config, availability) if stato.needs_download]


def summarize_freshness(config: Config, availability: Availability) -> pl.DataFrame:
    """Tabella leggibile dello stato, un record per mese atteso."""
    stati = month_statuses(config, availability)
    return pl.DataFrame(
        [
            {
                "month": stato.label,
                "days_available": stato.days_available,
                "days_downloaded": stato.days_downloaded,
                "partial": stato.partial,
                "needs_download": stato.needs_download,
                "reason": stato.reason,
            }
            for stato in stati
        ]
    )


__all__ = [
    "Availability",
    "FreshnessError",
    "MonthStatus",
    "month_statuses",
    "pending_months",
    "query_availability",
    "summarize_freshness",
]
