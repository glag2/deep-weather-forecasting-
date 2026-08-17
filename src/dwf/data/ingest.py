"""Ingestione GRIB -> Zarr, con catalogo Parquet degli slot.

Struttura reale dei file, verificata sui GRIB scaricati e non assunta:

- **istantanee**: `(time, latitude, longitude)`, dove `time` e' direttamente l'istante
  valido. Un mese di 31 giorni con 3 slot da' 93 istanti. Mappatura uno a uno.
- **cumulate**: `(time, step, latitude, longitude)`. ERA5 le serve come corse di
  previsione: `time` e' l'istante base (06 o 18 UTC), `step` va da 1 a 12 ore e
  l'istante valido e' `time + step`. Le due corse giornaliere tassellano le 24 ore
  senza sovrapporsi (verificato: 756 istanti validi, 0 duplicati). E' questa
  differenza di struttura fra le due famiglie a produrre il `DatasetBuildError` se si
  prova ad aprirle nello stesso dataset, ed e' il motivo per cui il download le
  separa.
- **statici**: `(latitude, longitude)`, senza asse temporale.

Il valore cumulato di ERA5 si riferisce all'ora precedente all'istante valido, quindi
il totale sulla finestra e' la somma dei valori orari in essa contenuti. Poiche'
nessuna finestra attraversa la mezzanotte (vincolo imposto in `slots.py`), ogni mese e'
autosufficiente e puo' essere ingerito in modo indipendente dagli altri.

Lo store Zarr viene creato una volta con l'intero asse temporale del periodo e poi
riempito per regioni, un mese alla volta. Cosi' i mesi possono arrivare in qualunque
ordine, che e' quello che serve quando il download procede a ondate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from dwf.config import Config
from dwf.slots import GAP_LABEL, accumulation_hours, build_split_layout, split_labels
from dwf.tables import (
    FOLDS,
    SLOT_STATS,
    SLOTS,
    VARIABLES,
    cast_to_schema,
    write_table,
)
from dwf.variables import spec_by_cds_name

FILL_VALUE = np.float32(np.nan)


def as_naive_utc(tempi: list[datetime]) -> np.ndarray:
    """Converte istanti UTC consapevoli in `datetime64[ns]` senza fuso.

    Nel progetto ogni istante e' UTC per costruzione, mentre `datetime64` non
    rappresenta i fusi: la conversione esplicita evita che numpy scarti il fuso
    silenziosamente e rende il confronto con i tempi letti dai GRIB, anch'essi senza
    fuso, corretto per costruzione.
    """
    return np.array([momento.replace(tzinfo=None) for momento in tempi], dtype="datetime64[ns]")


class IngestError(RuntimeError):
    """L'ingestione non puo' proseguire senza produrre dati scorretti."""


@dataclass(frozen=True, slots=True)
class MonthResult:
    """Esito dell'ingestione di un mese."""

    year: int
    month: int
    n_slots: int
    variables: tuple[str, ...]
    stats: pl.DataFrame


# --------------------------------------------------------------------------- #
# Apertura dei GRIB
# --------------------------------------------------------------------------- #


def open_grib(path: Path, **kwargs: Any) -> xr.Dataset:
    """Apre un GRIB con cfgrib, con un errore esplicito se manca o non e' leggibile."""
    if not path.exists():
        raise IngestError(f"File GRIB assente: {path}. Eseguire prima il download.")
    try:
        return xr.open_dataset(path, engine="cfgrib", **kwargs)
    except Exception as exc:
        raise IngestError(f"Impossibile aprire {path}: {type(exc).__name__}: {exc}") from exc


def check_grid(dataset: xr.Dataset, config: Config) -> None:
    """Verifica che la griglia del file coincida con quella configurata.

    Un disallineamento silenzioso produrrebbe tensori mescolati fra risoluzioni
    diverse, difficilissimi da diagnosticare a valle.
    """
    attesa = (config.region.n_lat, config.region.n_lon)
    trovata = (dataset.sizes.get("latitude"), dataset.sizes.get("longitude"))
    if trovata != attesa:
        raise IngestError(f"Griglia inattesa: attesa {attesa}, trovata {trovata}")

    if float(dataset.latitude.values[0]) < float(dataset.latitude.values[-1]):
        raise IngestError(
            "Latitudine crescente: ERA5 la fornisce decrescente (da nord a sud) e il "
            "resto della pipeline assume quell'ordine"
        )


ACCUMULATED_DIMS = ("valid_time", "latitude", "longitude")


def flatten_accumulated(dataset: xr.Dataset) -> xr.Dataset:
    """Trasforma `(time, step)` in un unico asse `valid_time` ordinato e univoco.

    L'ordine delle dimensioni viene imposto esplicitamente: `stack` sposta il nuovo
    asse in ultima posizione, quindi senza `transpose` gli array uscirebbero come
    `(latitude, longitude, valid_time)` e ogni indicizzazione temporale colpirebbe la
    latitudine.
    """
    if "step" not in dataset.dims:
        # Alcune richieste con una sola ora restituiscono `step` come scalare.
        piatto = dataset.swap_dims({"time": "valid_time"}).sortby("valid_time")
        return piatto.transpose(*ACCUMULATED_DIMS)

    stacked = dataset.stack(record=("time", "step"), create_index=False)
    stacked = stacked.assign_coords(valid_time=("record", stacked.valid_time.values))
    stacked = stacked.swap_dims({"record": "valid_time"}).sortby("valid_time")

    tempi = pd.to_datetime(stacked.valid_time.values)
    _, primi = np.unique(tempi, return_index=True)
    if len(primi) != len(tempi):
        stacked = stacked.isel(valid_time=np.sort(primi))
    stacked = stacked.drop_vars(["record", "time", "step"], errors="ignore")
    return stacked.transpose(*ACCUMULATED_DIMS)


# --------------------------------------------------------------------------- #
# Creazione dello store
# --------------------------------------------------------------------------- #


def dynamic_short_names(config: Config) -> list[str]:
    """Nomi brevi delle variabili dinamiche, nell'ordine dichiarato in configurazione."""
    nomi: list[str] = []
    for cds_name in [*config.variables.instantaneous, *config.variables.accumulated]:
        nomi.append(spec_by_cds_name(cds_name).short_name)
    return nomi


def initialize_store(config: Config, *, overwrite: bool = False) -> Path:
    """Crea lo store Zarr vuoto con l'intero asse temporale del periodo.

    Pre-allocare tutto il periodo permette di scrivere i mesi per regione e in
    qualunque ordine. Gli slot non ancora ingeriti restano NaN e il catalogo li marca
    come non utilizzabili, cosi' un periodo parzialmente scaricato e' comunque
    utilizzabile senza risultati silenziosamente sbagliati.
    """
    destinazione = config.zarr_path
    if destinazione.exists() and not overwrite:
        return destinazione

    tempi = config.time.slot_times()
    latitudini = config.region.latitudes
    longitudini = config.region.longitudes

    forma = (len(tempi), len(latitudini), len(longitudini))
    vuoto = np.full(forma, FILL_VALUE, dtype=np.float32)

    dimensioni = ("slot", "latitude", "longitude")
    dataset = xr.Dataset(
        {nome: (dimensioni, vuoto.copy()) for nome in dynamic_short_names(config)},
        coords={
            "slot": np.arange(len(tempi), dtype=np.int32),
            "valid_time": ("slot", as_naive_utc(tempi)),
            "latitude": latitudini.astype(np.float32),
            "longitude": longitudini.astype(np.float32),
        },
        attrs={
            "periodo_inizio": config.time.start,
            "periodo_fine": config.time.end,
            "slot_hours": str(config.time.slot_hours),
            "accum_window_hours": config.time.accum_window_hours,
            "nota": "unita' native ERA5; le trasformazioni sono applicate a valle",
        },
    )

    chunks = {
        "slot": config.paths.chunk_slots,
        "latitude": len(latitudini),
        "longitude": len(longitudini),
    }
    destinazione.parent.mkdir(parents=True, exist_ok=True)
    dataset.chunk(chunks).to_zarr(destinazione, mode="w", consolidated=True)
    return destinazione


def slot_positions(config: Config, year: int, month: int) -> tuple[np.ndarray, list[datetime]]:
    """Posizioni nello store e istanti degli slot appartenenti a quel mese."""
    tutti = config.time.slot_times()
    posizioni = [
        indice
        for indice, momento in enumerate(tutti)
        if (momento.year, momento.month) == (year, month)
    ]
    if not posizioni:
        raise IngestError(
            f"{year}-{month:02d} non appartiene al periodo configurato "
            f"({config.time.start} .. {config.time.end})"
        )
    return np.array(posizioni, dtype=int), [tutti[indice] for indice in posizioni]


# --------------------------------------------------------------------------- #
# Ingestione
# --------------------------------------------------------------------------- #


def read_instantaneous(
    config: Config, year: int, month: int, tempi: list[datetime]
) -> dict[str, np.ndarray]:
    """Estrae i campi istantanei agli slot richiesti."""
    percorso = config.raw_dir / f"instantaneous_{year:04d}-{month:02d}.grib"
    with open_grib(percorso) as dataset:
        check_grid(dataset, config)
        richiesti = as_naive_utc(tempi)
        disponibili = pd.to_datetime(dataset.time.values)
        mancanti = [
            str(momento) for momento in richiesti if momento not in disponibili.values
        ]
        if mancanti:
            raise IngestError(
                f"{percorso.name}: mancano {len(mancanti)} istanti richiesti, "
                f"primi: {mancanti[:3]}"
            )
        selezione = dataset.sel(time=richiesti)
        return {
            nome: np.asarray(selezione[nome].values, dtype=np.float32)
            for nome in selezione.data_vars
        }


def read_accumulated(
    config: Config, year: int, month: int, tempi: list[datetime]
) -> dict[str, np.ndarray]:
    """Aggrega i campi cumulati sulla finestra centrata su ogni slot.

    Il totale della finestra e' la somma dei valori orari, perche' ogni valore ERA5
    copre l'ora precedente al proprio istante valido.
    """
    percorso = config.raw_dir / f"accumulated_{year:04d}-{month:02d}.grib"
    finestra = config.time.accum_window_hours

    with open_grib(percorso) as grezzo:
        check_grid(grezzo, config)
        dataset = flatten_accumulated(grezzo)
        disponibili = pd.to_datetime(dataset.valid_time.values)
        posizione_di = {momento: indice for indice, momento in enumerate(disponibili)}

        # Per ogni slot, gli istanti orari che compongono la sua finestra.
        finestre: list[list[int]] = []
        for momento in tempi:
            ore = accumulation_hours(momento.hour, finestra)
            indici: list[int] = []
            for ora in ore:
                istante = pd.Timestamp(momento.replace(hour=ora, minute=0, tzinfo=None))
                indice = posizione_di.get(istante)
                if indice is None:
                    raise IngestError(
                        f"{percorso.name}: manca l'ora {istante} necessaria alla "
                        f"finestra dello slot {momento}"
                    )
                indici.append(indice)
            finestre.append(indici)

        risultato: dict[str, np.ndarray] = {}
        for nome in dataset.data_vars:
            # Si carica una volta la serie oraria completa e si somma per finestra:
            # sommare direttamente su xarray creerebbe un array intermedio enorme.
            orario = np.asarray(
                dataset[nome].transpose(*ACCUMULATED_DIMS).values, dtype=np.float32
            )
            accumulato = np.empty(
                (len(tempi), orario.shape[-2], orario.shape[-1]), dtype=np.float32
            )
            for posizione, indici in enumerate(finestre):
                accumulato[posizione] = orario[indici].sum(axis=0)
            spec = spec_by_cds_name_safe(nome)
            if spec is not None and spec.non_negative:
                # Le cumulate ERA5 hanno rumore numerico che puo' dare valori
                # negativi minuscoli, privi di senso fisico.
                np.clip(accumulato, 0.0, None, out=accumulato)
            risultato[nome] = accumulato
        return risultato


def spec_by_cds_name_safe(short_name: str) -> Any:
    """Cerca la specifica per nome breve, senza sollevare se assente."""
    from dwf.variables import spec_by_short_name

    try:
        return spec_by_short_name(short_name)
    except (KeyError, ValueError):
        return None


def compute_stats(campi: dict[str, np.ndarray], posizioni: np.ndarray) -> pl.DataFrame:
    """Statistiche per slot e variabile, usate come controllo qualita'."""
    righe: list[dict[str, Any]] = []
    for nome, valori in campi.items():
        for offset, indice in enumerate(posizioni):
            fetta = valori[offset]
            validi = np.isfinite(fetta)
            n_validi = int(validi.sum())
            righe.append(
                {
                    "slot_index": int(indice),
                    "variable": nome,
                    "n_nan": int(fetta.size - n_validi),
                    "n_valid": n_validi,
                    "minimum": float(fetta[validi].min()) if n_validi else float("nan"),
                    "maximum": float(fetta[validi].max()) if n_validi else float("nan"),
                    "mean": float(fetta[validi].mean()) if n_validi else float("nan"),
                }
            )
    return cast_to_schema(pl.DataFrame(righe), SLOT_STATS)


def ingest_month(config: Config, year: int, month: int) -> MonthResult:
    """Ingerisce un mese nello store Zarr, scrivendo solo la sua regione."""
    posizioni, tempi = slot_positions(config, year, month)
    campi: dict[str, np.ndarray] = {}
    if config.variables.instantaneous:
        campi.update(read_instantaneous(config, year, month, tempi))
    if config.variables.accumulated:
        campi.update(read_accumulated(config, year, month, tempi))

    attese = set(dynamic_short_names(config))
    trovate = set(campi)
    if trovate != attese:
        raise IngestError(
            f"{year}-{month:02d}: variabili inattese. mancanti={sorted(attese - trovate)}, "
            f"in eccesso={sorted(trovate - attese)}"
        )

    # Le posizioni di un mese sono contigue, quindi si scrive una sola regione.
    inizio, fine = int(posizioni[0]), int(posizioni[-1]) + 1
    if fine - inizio != len(posizioni):
        raise IngestError(f"{year}-{month:02d}: posizioni non contigue nello store")

    parziale = xr.Dataset(
        {nome: (("slot", "latitude", "longitude"), valori) for nome, valori in campi.items()}
    )
    parziale.to_zarr(config.zarr_path, region={"slot": slice(inizio, fine)})

    return MonthResult(
        year=year,
        month=month,
        n_slots=len(posizioni),
        variables=tuple(sorted(campi)),
        stats=compute_stats(campi, posizioni),
    )


def ingest_static(config: Config) -> Path:
    """Converte i campi invarianti nel tempo in uno store Zarr dedicato."""
    percorso = config.raw_dir / "static.grib"
    with open_grib(percorso) as dataset:
        check_grid(dataset, config)
        pulito = dataset.drop_vars(
            ["number", "time", "step", "surface", "valid_time"], errors="ignore"
        )
        pulito = pulito.assign_coords(
            latitude=pulito.latitude.astype(np.float32),
            longitude=pulito.longitude.astype(np.float32),
        )
        for nome in pulito.data_vars:
            pulito[nome] = pulito[nome].astype(np.float32)
        config.static_path.parent.mkdir(parents=True, exist_ok=True)
        pulito.to_zarr(config.static_path, mode="w", consolidated=True)
    return config.static_path


# --------------------------------------------------------------------------- #
# Catalogo
# --------------------------------------------------------------------------- #


def build_catalogue(
    config: Config, ingested: set[tuple[int, int]], stats: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Costruisce il catalogo degli slot con assegnazione di split e utilizzabilita'.

    Uno slot e' utilizzabile solo se il suo mese e' stato ingerito e se nessuna
    variabile presenta valori non finiti: un NaN silenzioso in input si propagherebbe
    a tutta la finestra di addestramento.
    """
    tempi = config.time.slot_times()
    n_slots = len(tempi)

    utilizzabile = np.array(
        [(momento.year, momento.month) in ingested for momento in tempi], dtype=bool
    )
    if stats is not None and stats.height:
        con_nan = set(
            stats.filter(pl.col("n_nan") > 0).get_column("slot_index").unique().to_list()
        )
        for indice in con_nan:
            utilizzabile[indice] = False

    # Riferimento cronologico solo indicativo: l'assegnazione autorevole per il
    # training e' per fold e vive in `folds.parquet`, perche' con la finestra mobile
    # lo stesso slot cambia split da un fold all'altro.
    riferimento = build_split_layout(
        n_slots,
        train_fraction=config.split.train_fraction,
        val_fraction=config.split.val_fraction,
        gap_slots=config.split.gap_slots,
        input_slots=config.windows.input_slots,
        output_slots=config.windows.output_slots,
    )
    etichette = split_labels(n_slots, riferimento)

    ore = config.time.slot_hours
    frame = pl.DataFrame(
        {
            "slot_index": list(range(n_slots)),
            "valid_time": tempi,
            "year": [momento.year for momento in tempi],
            "month": [momento.month for momento in tempi],
            "day": [momento.day for momento in tempi],
            "hour": [momento.hour for momento in tempi],
            "slot_of_day": [ore.index(momento.hour) for momento in tempi],
            "day_of_year": [momento.timetuple().tm_yday for momento in tempi],
            "source_month": [f"{momento.year:04d}-{momento.month:02d}" for momento in tempi],
            "split": [str(etichetta) for etichetta in etichette],
            "usable": utilizzabile.tolist(),
        }
    )
    return cast_to_schema(frame, SLOTS)


def write_catalogue(config: Config, catalogue: pl.DataFrame) -> Path:
    return write_table(catalogue, SLOTS, config.tables_dir)


def build_folds_table(config: Config, usable: np.ndarray | None = None) -> pl.DataFrame:
    """Assegnazione slot -> split per ciascun fold, con gli inizi di campione ammessi.

    Un inizio di campione viene marcato solo se **tutti** gli slot della finestra
    input+target sono utilizzabili: altrimenti il campione conterrebbe slot mai
    ingeriti o con valori non finiti.
    """
    finestra = config.windows.input_slots + config.windows.output_slots
    righe: list[dict[str, Any]] = []

    for indice_fold, fold in enumerate(config.build_folds()):
        for split, (inizio, fine) in fold.bounds.items():
            ammessi = set(fold.sample_starts[split])
            for slot_index in range(inizio, fine):
                if usable is None:
                    valido = slot_index in ammessi
                else:
                    valido = slot_index in ammessi and bool(
                        usable[slot_index : slot_index + finestra].all()
                    )
                righe.append(
                    {
                        "fold": indice_fold,
                        "split": split,
                        "slot_index": slot_index,
                        "is_sample_start": valido,
                    }
                )
    return cast_to_schema(pl.DataFrame(righe), FOLDS)


def write_folds_table(config: Config, folds: pl.DataFrame) -> Path:
    return write_table(folds, FOLDS, config.tables_dir)


def write_variables_table(config: Config) -> Path:
    """Materializza il registro delle variabili usate, marcando i target."""
    from dwf.tables import build_variables_table

    specs = [
        spec_by_cds_name(nome)
        for nome in [
            *config.variables.instantaneous,
            *config.variables.accumulated,
            *config.variables.static,
        ]
    ]
    frame = build_variables_table(specs, targets=config.target_names)
    return write_table(frame, VARIABLES, config.tables_dir)


def available_months(config: Config) -> list[tuple[int, int]]:
    """Mesi per i quali entrambi i GRIB necessari sono presenti e non vuoti."""
    presenti: list[tuple[int, int]] = []
    for year, month in config.time.months():
        richiesti = []
        if config.variables.instantaneous:
            richiesti.append(config.raw_dir / f"instantaneous_{year:04d}-{month:02d}.grib")
        if config.variables.accumulated:
            richiesti.append(config.raw_dir / f"accumulated_{year:04d}-{month:02d}.grib")
        if all(path.exists() and path.stat().st_size > 0 for path in richiesti):
            presenti.append((year, month))
    return presenti


__all__ = [
    "GAP_LABEL",
    "IngestError",
    "MonthResult",
    "available_months",
    "build_catalogue",
    "build_folds_table",
    "flatten_accumulated",
    "ingest_month",
    "ingest_static",
    "initialize_store",
    "slot_positions",
    "write_catalogue",
    "write_folds_table",
    "write_variables_table",
]
