"""Scarico dei dati ERA5 dal Climate Data Store.

Differenze rispetto a ``Data/era5-request.py``, che scaricava un solo mese con
parametri fissi:

- una richiesta per mese e per **famiglia di variabili**, non una richiesta unica.
  Mescolare campi di analisi e campi *mean rate* in un solo GRIB e' cio' che
  provocava il ``DatasetBuildError: key present and new value is different:
  key='time'`` visibile negli output di ``EDA.ipynb``: le due famiglie hanno assi
  temporali diversi e cfgrib non puo' unirle;
- le variabili istantanee sono richieste solo agli slot previsti, quelle cumulate a
  cadenza oraria perche' vanno aggregate sulla finestra;
- ritaglio all'area configurata, invece del globo intero;
- ripresa dopo interruzione: un file gia' presente e non vuoto non viene riscaricato;
- ``retrieve(..., target=...)`` senza concatenare ``.download()``, come indicato dalla
  documentazione CDS corrente e compatibile con entrambi i client restituiti da
  ``cdsapi.Client`` (legacy e ``ecmwf.datastores``).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from dwf.config import NATIVE_GRID_DEG, Config
from dwf.credentials import require_credentials

DATASET = "reanalysis-era5-single-levels"

# Anno, mese, giorno e ora usati per i campi invarianti nel tempo: un solo istante
# basta, e questo e' arbitrario ma fissato per rendere il file riproducibile.
STATIC_REFERENCE = (2024, 1, 1, 0)


class RetrieveClient(Protocol):
    """Sottoinsieme di ``cdsapi.Client`` effettivamente usato.

    Dichiararlo come protocollo permette di collaudare la pipeline con un client
    finto, senza credenziali e senza traffico di rete.
    """

    def retrieve(self, name: str, request: dict[str, Any], target: str | None = ...) -> Any: ...


@dataclass(frozen=True, slots=True)
class DownloadTask:
    """Una singola richiesta al CDS e il file che deve produrre."""

    kind: str
    variables: tuple[str, ...]
    hours: tuple[int, ...]
    year: int | None
    month: int | None
    target: Path

    @property
    def label(self) -> str:
        if self.year is None or self.month is None:
            return f"{self.kind}"
        return f"{self.kind} {self.year:04d}-{self.month:02d}"


def _hour_strings(hours: Sequence[int]) -> list[str]:
    return [f"{hour:02d}:00" for hour in sorted(hours)]


def _day_strings(config: Config, year: int, month: int) -> list[str]:
    """Giorni da chiedere per quel mese, limitati al periodo configurato.

    Il primo e l'ultimo mese del periodo sono in genere parziali: chiedere il mese
    intero scaricherebbe dati fuori intervallo, e nell'ultimo mese giorni non ancora
    pubblicati, che il CDS rifiuta.
    """
    return [f"{day:02d}" for day in config.time.days_in_month(year, month)]


def build_tasks(config: Config) -> list[DownloadTask]:
    """Elenca tutte le richieste necessarie a coprire il periodo configurato.

    L'ordine e' deliberato: prima i campi statici (una sola richiesta, veloce, e se
    fallisce non vale la pena accodare decine di mesi), poi mese per mese.
    """
    raw_dir = config.raw_dir
    tasks: list[DownloadTask] = []

    if config.variables.static:
        tasks.append(
            DownloadTask(
                kind="static",
                variables=tuple(config.variables.static),
                hours=(STATIC_REFERENCE[3],),
                year=None,
                month=None,
                target=raw_dir / "static.grib",
            )
        )

    slot_hours = tuple(config.time.slot_hours)
    hourly_hours = tuple(config.time.hourly_hours)

    for year, month in config.time.months():
        if config.variables.instantaneous:
            tasks.append(
                DownloadTask(
                    kind="instantaneous",
                    variables=tuple(config.variables.instantaneous),
                    hours=slot_hours,
                    year=year,
                    month=month,
                    target=raw_dir / f"instantaneous_{year:04d}-{month:02d}.grib",
                )
            )
        if config.variables.accumulated:
            tasks.append(
                DownloadTask(
                    kind="accumulated",
                    variables=tuple(config.variables.accumulated),
                    hours=hourly_hours,
                    year=year,
                    month=month,
                    target=raw_dir / f"accumulated_{year:04d}-{month:02d}.grib",
                )
            )
    return tasks


def build_payload(task: DownloadTask, config: Config) -> dict[str, Any]:
    """Costruisce il dizionario di richiesta CDS per un task."""
    if task.kind == "static":
        year, month, day, _ = STATIC_REFERENCE
        years = [f"{year:04d}"]
        months = [f"{month:02d}"]
        days = [f"{day:02d}"]
    else:
        if task.year is None or task.month is None:
            raise ValueError(f"Task {task.kind!r} senza anno o mese")
        years = [f"{task.year:04d}"]
        months = [f"{task.month:02d}"]
        days = _day_strings(config, task.year, task.month)
        if not days:
            raise ValueError(
                f"Nessun giorno da richiedere per {task.year}-{task.month:02d}: "
                "task incoerente con il periodo configurato"
            )

    payload: dict[str, Any] = {
        "product_type": ["reanalysis"],
        "variable": list(task.variables),
        "year": years,
        "month": months,
        "day": days,
        "time": _hour_strings(task.hours),
        "data_format": "grib",
        "download_format": "unarchived",
        "area": config.region.cds_area,
    }

    # `grid` chiede a MARS di regrigliare lato server. Non e' fra i campi del form
    # web e alcune installazioni del CDS lo rifiutano, quindi lo si invia solo
    # quando serve davvero, cioe' quando la risoluzione richiesta non e' la nativa.
    step = config.region.grid_step
    if abs(step - NATIVE_GRID_DEG) > 1e-9:
        payload["grid"] = [f"{step}", f"{step}"]

    return payload


def _needs_download(target: Path, *, overwrite: bool) -> bool:
    if overwrite:
        return True
    if not target.exists():
        return True
    # Un file vuoto o troncato e' il residuo di un download interrotto.
    return target.stat().st_size == 0


def make_client(env_file: Path | None = None) -> RetrieveClient:
    """Crea il client CDS dopo aver verificato le credenziali.

    ``cdsapi.Client`` risolve le credenziali dentro ``__new__`` e solleva un errore
    generico se mancano: il controllo esplicito a monte produce un messaggio con la
    procedura da seguire. ``debug`` resta ``False`` perche' a quel livello il client
    registrerebbe la chiave nei log.
    """
    require_credentials(env_file)
    import cdsapi

    return cdsapi.Client(quiet=False, debug=False, wait_until_complete=True)


@dataclass(frozen=True, slots=True)
class DownloadOutcome:
    """Esito di un singolo task, per il manifest e per il resoconto a schermo."""

    task: DownloadTask
    status: str
    size_bytes: int
    seconds: float
    message: str = ""


def run_task(
    task: DownloadTask,
    config: Config,
    client: RetrieveClient,
    *,
    overwrite: bool = False,
) -> DownloadOutcome:
    """Esegue una richiesta, con tentativi ripetuti e scrittura atomica."""
    if not _needs_download(task.target, overwrite=overwrite):
        return DownloadOutcome(
            task=task,
            status="skipped",
            size_bytes=task.target.stat().st_size,
            seconds=0.0,
            message="file gia' presente",
        )

    task.target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(task, config)
    # Si scrive su un file temporaneo e si rinomina solo a download completato, cosi'
    # un'interruzione non lascia un GRIB troncato che sembrerebbe valido.
    staging = task.target.with_suffix(task.target.suffix + ".partial")

    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, config.download.max_retries + 1):
        try:
            client.retrieve(DATASET, payload, str(staging))
            if not staging.exists() or staging.stat().st_size == 0:
                raise RuntimeError("il CDS ha restituito un file vuoto")
            staging.replace(task.target)
            return DownloadOutcome(
                task=task,
                status="downloaded",
                size_bytes=task.target.stat().st_size,
                seconds=time.perf_counter() - started,
            )
        except Exception as exc:  # il client CDS solleva Exception generiche
            last_error = exc
            staging.unlink(missing_ok=True)
            if attempt < config.download.max_retries:
                time.sleep(config.download.retry_backoff_seconds * attempt)

    return DownloadOutcome(
        task=task,
        status="failed",
        size_bytes=0,
        seconds=time.perf_counter() - started,
        message=f"{type(last_error).__name__}: {last_error}",
    )


def run_tasks(
    tasks: Iterable[DownloadTask],
    config: Config,
    client: RetrieveClient,
    *,
    overwrite: bool = False,
    stop_on_error: bool = False,
) -> list[DownloadOutcome]:
    """Esegue i task in sequenza, raccogliendo gli esiti.

    Sequenziale e non parallelo: il CDS limita le richieste concorrenti per utente e
    accodarne molte insieme peggiora i tempi invece di migliorarli.
    """
    outcomes: list[DownloadOutcome] = []
    for task in tasks:
        outcome = run_task(task, config, client, overwrite=overwrite)
        outcomes.append(outcome)
        if outcome.status == "failed" and stop_on_error:
            break
    return outcomes


def outcomes_to_records(outcomes: Sequence[DownloadOutcome]) -> list[dict[str, Any]]:
    """Trasforma gli esiti in record per la tabella Parquet del manifest."""
    now = datetime.now(UTC)
    return [
        {
            "kind": outcome.task.kind,
            "year": outcome.task.year,
            "month": outcome.task.month,
            "filename": outcome.task.target.name,
            "n_variables": len(outcome.task.variables),
            "n_hours": len(outcome.task.hours),
            "status": outcome.status,
            "size_bytes": outcome.size_bytes,
            "seconds": outcome.seconds,
            "message": outcome.message,
            "recorded_at": now,
        }
        for outcome in outcomes
    ]
