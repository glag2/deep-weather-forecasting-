"""Layer tabellare Polars/Parquet: registro autorevole dei dati puliti.

Zarr conserva i tensori numerici; qui vive tutto cio' che descrive, cataloga,
misura o produce quei tensori. In particolare ``slots.parquet`` e' la fonte di
verita' su quali slot esistono, quali sono utilizzabili e a quale split
appartengono: dataset, valutazione e inferenza leggono da lui, mai dal solo Zarr.

Ogni tabella ha uno schema dichiarato e verificato in scrittura e in rilettura.
Uno schema solo dichiarativo non protegge da un Parquet scritto da una versione
precedente del codice, quindi il controllo e' eseguito a runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from dwf.slots import GAP_LABEL, SPLIT_NAMES

UTC_TIMESTAMP = pl.Datetime("us", "UTC")
SPLIT_ENUM = pl.Enum([*SPLIT_NAMES, GAP_LABEL])

TableSchema = Mapping[str, pl.DataType]


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Nome del file e schema atteso di una tabella del layer 2."""

    filename: str
    schema: TableSchema
    description: str

    def path(self, tables_dir: Path) -> Path:
        return tables_dir / self.filename


# --------------------------------------------------------------------------- #
# Schemi
# --------------------------------------------------------------------------- #

SLOTS = TableSpec(
    filename="slots.parquet",
    description="Catalogo degli slot temporali con assegnazione di split.",
    schema={
        "slot_index": pl.Int32,
        "valid_time": UTC_TIMESTAMP,
        "year": pl.Int16,
        "month": pl.Int8,
        "day": pl.Int8,
        "hour": pl.Int8,
        "slot_of_day": pl.Int8,
        "day_of_year": pl.Int16,
        "source_month": pl.String,
        "split": SPLIT_ENUM,
        "usable": pl.Boolean,
    },
)

SLOT_STATS = TableSpec(
    filename="slot_stats.parquet",
    description="Controlli qualita' per slot e variabile, in forma lunga.",
    schema={
        "slot_index": pl.Int32,
        "variable": pl.String,
        "n_nan": pl.Int32,
        "n_valid": pl.Int32,
        "minimum": pl.Float64,
        "maximum": pl.Float64,
        "mean": pl.Float64,
    },
)

VARIABLES = TableSpec(
    filename="variables.parquet",
    description="Schema delle variabili ERA5 usate, con unita' e trasformazioni.",
    schema={
        "cds_name": pl.String,
        "short_name": pl.String,
        "kind": pl.String,
        "units": pl.String,
        "description": pl.String,
        "transform": pl.String,
        "non_negative": pl.Boolean,
        "is_target": pl.Boolean,
    },
)

CHANNELS = TableSpec(
    filename="channels.parquet",
    description="Schema dei canali di input del modello, in ordine di stacking.",
    schema={
        "channel_index": pl.Int32,
        "name": pl.String,
        "group": pl.String,
        "source_variable": pl.String,
        "lag": pl.Int16,
        "transform": pl.String,
        "normalized": pl.Boolean,
    },
)

NORM_STATS = TableSpec(
    filename="norm_stats.parquet",
    description="Statistiche di normalizzazione, calcolate solo sullo split di train.",
    schema={
        "variable": pl.String,
        "transform": pl.String,
        "mean": pl.Float64,
        "std": pl.Float64,
        "minimum": pl.Float64,
        "maximum": pl.Float64,
        "n_values": pl.Int64,
        "computed_on_split": pl.String,
    },
)

METRICS = TableSpec(
    filename="metrics.parquet",
    description="Metriche per modello, split, variabile e lead time.",
    schema={
        "model": pl.String,
        "split": pl.String,
        "variable": pl.String,
        "lead_slot": pl.Int16,
        "metric": pl.String,
        "value": pl.Float64,
        "n_values": pl.Int64,
    },
)

RELIABILITY = TableSpec(
    filename="reliability.parquet",
    description="Bin di calibrazione delle probabilita' previste (diagramma di affidabilita').",
    schema={
        "model": pl.String,
        "split": pl.String,
        "variable": pl.String,
        "lead_slot": pl.Int16,
        "bin_lower": pl.Float64,
        "bin_upper": pl.Float64,
        "forecast_mean": pl.Float64,
        "observed_frequency": pl.Float64,
        "count": pl.Int64,
    },
)

FORECAST = TableSpec(
    filename="forecast.parquet",
    description="Previsione in forma lunga: un record per punto di griglia e lead time.",
    schema={
        "init_time": UTC_TIMESTAMP,
        "valid_time": UTC_TIMESTAMP,
        "lead_slot": pl.Int16,
        "lead_hours": pl.Int16,
        "latitude": pl.Float32,
        "longitude": pl.Float32,
        "t2m_mean": pl.Float32,
        "t2m_std": pl.Float32,
        "precip_probability": pl.Float32,
        "precip_amount": pl.Float32,
        "snow_probability": pl.Float32,
    },
)

DOWNLOADS = TableSpec(
    filename="downloads.parquet",
    description="Esito delle richieste al CDS, una riga per file richiesto.",
    schema={
        # `year` e `month` sono nulli per i campi statici, che non hanno un mese.
        "kind": pl.String,
        "year": pl.Int16,
        "month": pl.Int8,
        "filename": pl.String,
        "n_variables": pl.Int32,
        "n_hours": pl.Int32,
        "status": pl.String,
        "size_bytes": pl.Int64,
        "seconds": pl.Float64,
        "message": pl.String,
        "recorded_at": UTC_TIMESTAMP,
    },
)

ALL_SPECS: tuple[TableSpec, ...] = (
    SLOTS,
    SLOT_STATS,
    VARIABLES,
    DOWNLOADS,
    CHANNELS,
    NORM_STATS,
    METRICS,
    RELIABILITY,
    FORECAST,
)


# --------------------------------------------------------------------------- #
# Lettura e scrittura validate
# --------------------------------------------------------------------------- #


def _describe_mismatch(actual: Mapping[str, Any], expected: TableSchema) -> str:
    missing = [name for name in expected if name not in actual]
    unexpected = [name for name in actual if name not in expected]
    wrong = [
        f"{name}: atteso {expected[name]}, trovato {actual[name]}"
        for name in expected
        if name in actual and actual[name] != expected[name]
    ]
    parts = []
    if missing:
        parts.append(f"colonne mancanti {missing}")
    if unexpected:
        parts.append(f"colonne inattese {unexpected}")
    if wrong:
        parts.append("tipi errati -> " + "; ".join(wrong))
    return "; ".join(parts) if parts else "ordine delle colonne diverso dallo schema"


def validate_schema(frame: pl.DataFrame, spec: TableSpec) -> None:
    """Verifica nomi, tipi e ordine delle colonne rispetto allo schema dichiarato."""
    actual = dict(frame.schema)
    if actual == dict(spec.schema) and list(actual) == list(spec.schema):
        return
    raise ValueError(
        f"La tabella {spec.filename} non rispetta lo schema: "
        f"{_describe_mismatch(actual, spec.schema)}"
    )


def write_table(frame: pl.DataFrame, spec: TableSpec, tables_dir: Path) -> Path:
    """Valida e scrive una tabella in Parquet compresso, creando la directory."""
    validate_schema(frame, spec)
    tables_dir.mkdir(parents=True, exist_ok=True)
    target = spec.path(tables_dir)
    frame.write_parquet(target, compression="zstd", statistics=True)
    return target


def read_table(spec: TableSpec, tables_dir: Path) -> pl.DataFrame:
    """Rilegge una tabella verificandone lo schema."""
    target = spec.path(tables_dir)
    if not target.exists():
        raise FileNotFoundError(
            f"Tabella assente: {target}. Eseguire prima la fase che la produce "
            f"({spec.description})"
        )
    frame = pl.read_parquet(target)
    validate_schema(frame, spec)
    return frame


def empty_table(spec: TableSpec) -> pl.DataFrame:
    """DataFrame vuoto conforme allo schema, utile come punto di partenza."""
    return pl.DataFrame(schema=dict(spec.schema))


def cast_to_schema(frame: pl.DataFrame, spec: TableSpec) -> pl.DataFrame:
    """Riordina e converte le colonne allo schema dichiarato.

    Evita di sparpagliare cast espliciti nei costruttori delle tabelle, ma
    fallisce in modo esplicito se una colonna attesa manca del tutto.
    """
    missing = [name for name in spec.schema if name not in frame.columns]
    if missing:
        raise ValueError(f"Colonne mancanti per {spec.filename}: {missing}")
    return frame.select(
        [pl.col(name).cast(dtype, strict=True) for name, dtype in spec.schema.items()]
    )


# --------------------------------------------------------------------------- #
# Costruttori
# --------------------------------------------------------------------------- #


def build_slots_table(
    times: list[datetime],
    slot_of_day_values: list[int],
    splits: list[str],
    usable: list[bool],
) -> pl.DataFrame:
    """Costruisce il catalogo degli slot a partire da sequenze allineate per posizione."""
    lengths = {
        "times": len(times),
        "slot_of_day": len(slot_of_day_values),
        "splits": len(splits),
        "usable": len(usable),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Sequenze di lunghezza diversa: {lengths}")

    unknown = sorted({value for value in splits} - {*SPLIT_NAMES, GAP_LABEL})
    if unknown:
        raise ValueError(f"Etichette di split non ammesse: {unknown}")

    frame = pl.DataFrame(
        {
            "slot_index": list(range(len(times))),
            "valid_time": times,
            "year": [moment.year for moment in times],
            "month": [moment.month for moment in times],
            "day": [moment.day for moment in times],
            "hour": [moment.hour for moment in times],
            "slot_of_day": slot_of_day_values,
            "day_of_year": [moment.timetuple().tm_yday for moment in times],
            "source_month": [f"{moment.year:04d}-{moment.month:02d}" for moment in times],
            "split": splits,
            "usable": usable,
        }
    )
    return cast_to_schema(frame, SLOTS)


def build_variables_table(specs: list[Any], targets: list[str]) -> pl.DataFrame:
    """Materializza il registro delle variabili in tabella, marcando i target."""
    target_set = set(targets)
    frame = pl.DataFrame(
        {
            "cds_name": [spec.cds_name for spec in specs],
            "short_name": [spec.short_name for spec in specs],
            "kind": [spec.kind for spec in specs],
            "units": [spec.units for spec in specs],
            "description": [spec.description for spec in specs],
            "transform": [spec.transform for spec in specs],
            "non_negative": [spec.non_negative for spec in specs],
            "is_target": [spec.short_name in target_set for spec in specs],
        }
    )
    return cast_to_schema(frame, VARIABLES)
