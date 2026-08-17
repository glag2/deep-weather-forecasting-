"""Dataset torch che serve finestre di input e target dallo store Zarr.

Il campionamento e' guidato da `folds.parquet`, che e' l'unica fonte autorevole di
quali finestre siano ammesse in ciascun fold e split: una finestra e' ammessa solo se
tutti i suoi slot sono stati ingeriti e non contengono valori non finiti.

Una misura ha guidato il disegno. Leggere un ritaglio da Zarr costa quanto leggere il
dominio intero (0,215 s contro 0,230 s per 30 slot), perche' i chunk coprono tutto lo
spazio e vanno comunque decompressi; suddividerli spazialmente peggiora le cose,
perche' l'overhead per chunk supera i byte risparmiati. La conseguenza e' che il costo
di lettura va **ammortizzato**: una finestra letta una volta serve piu' ritagli, e la
cache tiene le finestre usate di recente.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

from dwf.data.features import (
    WIND_SPEED,
    FeatureError,
    InputLayout,
    NormStats,
    build_input_tensor,
)
from dwf.tables import FOLDS, read_table

if TYPE_CHECKING:  # pragma: no cover - solo per i tipi
    from dwf.config import Config

# Chiavi dei tensori restituiti, cosi' che loss e valutazione non usino stringhe libere.
KEY_FEATURES = "features"
KEY_SLOT = "start_slot"


class DatasetError(RuntimeError):
    """Errore nella costruzione dei campioni."""


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Come si costruisce il target di una variabile a partire dai valori grezzi."""

    name: str
    head: str
    threshold: float | None
    reference: str | None


def target_specs(config: Config) -> tuple[TargetSpec, ...]:
    return tuple(
        TargetSpec(
            name=target.name,
            head=target.head,
            threshold=target.threshold,
            reference=target.reference,
        )
        for target in config.targets
    )


# --------------------------------------------------------------------------- #
# Lettura dallo store
# --------------------------------------------------------------------------- #


class ZarrWindowReader:
    """Legge finestre di slot dallo store, tenendo in cache le piu' recenti.

    La cache e' indispensabile: senza, ogni ritaglio pagherebbe per intero la
    decompressione dell'intera finestra spaziale.
    """

    def __init__(
        self,
        zarr_path: Any,
        variables: Sequence[str],
        *,
        cache_size: int = 2,
        static_path: Any | None = None,
        static_variables: Sequence[str] = (),
    ) -> None:
        import xarray as xr

        self._store = xr.open_zarr(zarr_path, consolidated=True)
        self._variables = tuple(variables)
        self._cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._order: list[tuple[int, int]] = []
        self._cache_size = max(1, cache_size)
        self._lock = threading.Lock()

        # Gli istanti si caricano una volta sola: leggerli dallo store a ogni campione
        # attraversa dask e costava 0,42 s per campione, cioe' quasi tutto il tempo di
        # costruzione del campione stesso.
        import pandas as pd

        self._valid_time = pd.to_datetime(self._store.valid_time.values).to_pydatetime()

        self.static: dict[str, np.ndarray] = {}
        if static_path is not None and static_variables:
            statico = xr.open_zarr(static_path, consolidated=True)
            for nome in static_variables:
                if nome not in statico:
                    raise DatasetError(
                        f"Campo statico assente dallo store: {nome!r}. "
                        f"Disponibili: {sorted(statico.data_vars)}"
                    )
                self.static[nome] = np.asarray(statico[nome].values, dtype=np.float32)

        mancanti = [nome for nome in self._variables if nome not in self._store]
        if mancanti:
            raise DatasetError(
                f"Variabili assenti dallo store: {mancanti}. "
                f"Disponibili: {sorted(self._store.data_vars)}"
            )

    @property
    def latitudes(self) -> np.ndarray:
        return np.asarray(self._store.latitude.values, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._store.sizes["latitude"]), int(self._store.sizes["longitude"])

    def valid_time(self, slot_index: int) -> datetime:
        return self._valid_time[slot_index]

    def read_window(self, start: int, length: int) -> dict[str, np.ndarray]:
        """Tutte le variabili sugli slot `[start, start + length)`, dominio intero."""
        chiave = (start, length)
        with self._lock:
            if chiave in self._cache:
                self._order.remove(chiave)
                self._order.append(chiave)
                return self._cache[chiave]

        finestra = {
            nome: np.asarray(
                self._store[nome].isel(slot=slice(start, start + length)).values,
                dtype=np.float32,
            )
            for nome in self._variables
        }

        with self._lock:
            self._cache[chiave] = finestra
            self._order.append(chiave)
            while len(self._order) > self._cache_size:
                vecchia = self._order.pop(0)
                self._cache.pop(vecchia, None)
        return finestra


# --------------------------------------------------------------------------- #
# Costruzione dei target
# --------------------------------------------------------------------------- #


def build_targets(
    specs: Sequence[TargetSpec],
    window: dict[str, np.ndarray],
    stats: NormStats,
) -> dict[str, torch.Tensor]:
    """Target e maschere per ciascuna testa, dai valori grezzi degli slot previsti.

    Le maschere esistono perche' non tutti i punti contribuiscono a tutte le loss: la
    quantita' di pioggia si addestra solo dove piove, e la frazione di neve solo dove
    c'e' precipitazione misurabile.
    """
    uscite: dict[str, torch.Tensor] = {}

    for spec in specs:
        if spec.name not in window:
            raise DatasetError(f"Manca la variabile target {spec.name!r} nella finestra")
        grezzi = window[spec.name]

        if spec.head == "gaussian":
            # Normalizzato: la NLL gaussiana su kelvin grezzi avrebbe gradienti
            # sbilanciati rispetto alle altre teste.
            uscite[f"target_{spec.name}"] = torch.from_numpy(
                np.ascontiguousarray(stats.normalize(spec.name, grezzi))
            )

        elif spec.head == "hurdle":
            if spec.threshold is None:
                raise DatasetError(f"target {spec.name!r}: soglia mancante")
            occorrenza = (grezzi > spec.threshold).astype(np.float32)
            uscite[f"target_{spec.name}_occurrence"] = torch.from_numpy(
                np.ascontiguousarray(occorrenza)
            )
            uscite[f"target_{spec.name}_amount"] = torch.from_numpy(
                np.ascontiguousarray(stats.normalize(spec.name, grezzi))
            )
            uscite[f"mask_{spec.name}_amount"] = torch.from_numpy(
                np.ascontiguousarray(occorrenza)
            )

        elif spec.head == "fraction_of":
            if spec.reference is None:
                raise DatasetError(f"target {spec.name!r}: riferimento mancante")
            riferimento = window[spec.reference]
            soglia = _reference_threshold(specs, spec.reference)
            valido = riferimento > soglia
            # Il rapporto e' definito solo dove il riferimento e' misurabile; altrove
            # dividerebbe per un valore dominato dal rumore di quantizzazione.
            frazione = np.zeros_like(riferimento, dtype=np.float32)
            np.divide(grezzi, riferimento, out=frazione, where=valido)
            # I GRIB impacchettano `tp` e `sf` su griglie di quantizzazione diverse,
            # quindi il rapporto puo' superare 1 di un passo di quantizzazione. Il
            # clip non altera i dati archiviati: vincola solo il bersaglio a restare
            # una frazione.
            np.clip(frazione, 0.0, 1.0, out=frazione)
            uscite[f"target_{spec.name}_fraction"] = torch.from_numpy(
                np.ascontiguousarray(frazione)
            )
            uscite[f"mask_{spec.name}_fraction"] = torch.from_numpy(
                np.ascontiguousarray(valido.astype(np.float32))
            )
        else:  # pragma: no cover - le teste ammesse sono validate in configurazione
            raise DatasetError(f"Testa non supportata: {spec.head!r}")

    return uscite


def _reference_threshold(specs: Sequence[TargetSpec], reference: str) -> float:
    for spec in specs:
        if spec.name == reference and spec.threshold is not None:
            return spec.threshold
    raise DatasetError(
        f"La variabile di riferimento {reference!r} non dichiara una soglia: "
        f"senza soglia il rapporto sarebbe dominato dal rumore vicino allo zero"
    )


# --------------------------------------------------------------------------- #
# Indice dei campioni
# --------------------------------------------------------------------------- #


def sample_starts(config: Config, fold: int, split: str) -> list[int]:
    """Slot iniziali ammessi per un fold e uno split, letti da `folds.parquet`."""
    tabella = read_table(FOLDS, config.tables_dir)
    selezione = tabella.filter(
        (pl.col("fold") == fold)
        & (pl.col("split") == split)
        & pl.col("is_sample_start")
    ).sort("slot_index")
    return selezione.get_column("slot_index").to_list()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class WeatherWindowDataset(Dataset):
    """Un campione e' una finestra di input piu' i target degli slot successivi."""

    def __init__(
        self,
        config: Config,
        layout: InputLayout,
        stats: NormStats,
        starts: Sequence[int],
        reader: ZarrWindowReader,
        *,
        crop_size: int | None = None,
        crops_per_window: int = 1,
        seed: int = 0,
    ) -> None:
        if not starts:
            raise DatasetError(
                "Nessuna finestra ammessa: servono mesi ingeriti e slot utilizzabili"
            )
        self.config = config
        self.layout = layout
        self.stats = stats
        self.starts = list(starts)
        self.reader = reader
        self.crop_size = crop_size
        self.crops_per_window = max(1, crops_per_window)
        self.specs = target_specs(config)
        self.input_slots = config.windows.input_slots
        self.output_slots = config.windows.output_slots
        self.slot_hours = tuple(config.time.slot_hours)
        self._rng = np.random.default_rng(seed)

        altezza, larghezza = reader.shape
        if crop_size is not None and (crop_size > altezza or crop_size > larghezza):
            raise DatasetError(
                f"crop_size {crop_size} eccede il dominio {altezza} x {larghezza}"
            )

    def __len__(self) -> int:
        return len(self.starts) * self.crops_per_window

    def window_of(self, index: int) -> int:
        return index // self.crops_per_window

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        posizione = self.window_of(index)
        if posizione >= len(self.starts):
            raise IndexError(index)
        inizio = self.starts[posizione]
        totale = self.input_slots + self.output_slots

        finestra = self.reader.read_window(inizio, totale)
        riga, colonna = self._crop_origin()
        ritagliata = self._crop(finestra, riga, colonna)

        ingresso = {
            nome: valori[: self.input_slots] for nome, valori in ritagliata.items()
        }
        uscita = {nome: valori[self.input_slots :] for nome, valori in ritagliata.items()}

        statici = self._crop_static(riga, colonna)
        latitudini = self.reader.latitudes
        if self.crop_size is not None:
            latitudini = latitudini[riga : riga + self.crop_size]

        # L'istante di riferimento e' l'ultimo slot osservato: e' da li' che il
        # modello estrapola.
        riferimento = self.reader.valid_time(inizio + self.input_slots - 1)

        caratteristiche = build_input_tensor(
            self.layout,
            ingresso,
            self.stats,
            static_fields=statici,
            latitudes=latitudini,
            reference_time=riferimento,
            slot_hours=self.slot_hours,
        )

        campione: dict[str, torch.Tensor] = {
            KEY_FEATURES: torch.from_numpy(np.ascontiguousarray(caratteristiche)),
            KEY_SLOT: torch.tensor(inizio, dtype=torch.int32),
        }
        campione.update(build_targets(self.specs, uscita, self.stats))
        return campione

    def _crop_origin(self) -> tuple[int, int]:
        if self.crop_size is None:
            return 0, 0
        altezza, larghezza = self.reader.shape
        riga = int(self._rng.integers(0, altezza - self.crop_size + 1))
        colonna = int(self._rng.integers(0, larghezza - self.crop_size + 1))
        return riga, colonna

    def _crop(
        self, finestra: dict[str, np.ndarray], riga: int, colonna: int
    ) -> dict[str, np.ndarray]:
        if self.crop_size is None:
            return finestra
        taglio = self.crop_size
        return {
            nome: valori[:, riga : riga + taglio, colonna : colonna + taglio]
            for nome, valori in finestra.items()
        }

    def _crop_static(self, riga: int, colonna: int) -> dict[str, np.ndarray]:
        if self.crop_size is None:
            return self.reader.static
        taglio = self.crop_size
        return {
            nome: campo[riga : riga + taglio, colonna : colonna + taglio]
            for nome, campo in self.reader.static.items()
        }


class WindowBatchSampler(Sampler[list[int]]):
    """Raggruppa in uno stesso batch i ritagli che condividono la finestra letta.

    Senza questo raggruppamento ogni elemento del batch pagherebbe una lettura
    completa da Zarr; con esso la lettura viene ammortizzata su tutto il batch.
    """

    def __init__(
        self,
        n_windows: int,
        crops_per_window: int,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        max_batches: int | None = None,
    ) -> None:
        self.n_windows = n_windows
        self.crops_per_window = max(1, crops_per_window)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_batches = max_batches
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[list[int]]:
        ordine = np.arange(self.n_windows)
        if self.shuffle:
            self._rng.shuffle(ordine)

        prodotti = 0
        for finestra in ordine:
            base = int(finestra) * self.crops_per_window
            indici = list(range(base, base + self.crops_per_window))
            for inizio in range(0, len(indici), self.batch_size):
                lotto = indici[inizio : inizio + self.batch_size]
                if not lotto:
                    continue
                yield lotto
                prodotti += 1
                if self.max_batches is not None and prodotti >= self.max_batches:
                    return

    def __len__(self) -> int:
        per_finestra = max(
            1, (self.crops_per_window + self.batch_size - 1) // self.batch_size
        )
        totale = self.n_windows * per_finestra
        if self.max_batches is not None:
            return min(totale, self.max_batches)
        return totale


def build_reader(config: Config, layout: InputLayout) -> ZarrWindowReader:
    """Lettore configurato con le variabili richieste dal layout."""
    dinamiche = [nome for nome in layout.dynamic_variables]
    for spec in config.targets:
        if spec.name not in dinamiche:
            raise FeatureError(
                f"Il target {spec.name!r} non e' fra le variabili dinamiche scaricate"
            )
    if layout.include_wind_speed and WIND_SPEED in dinamiche:  # pragma: no cover
        dinamiche.remove(WIND_SPEED)
    return ZarrWindowReader(
        config.zarr_path,
        dinamiche,
        static_path=config.static_path if layout.static_variables else None,
        static_variables=layout.static_variables,
    )


__all__ = [
    "KEY_FEATURES",
    "KEY_SLOT",
    "DatasetError",
    "TargetSpec",
    "WeatherWindowDataset",
    "WindowBatchSampler",
    "ZarrWindowReader",
    "build_reader",
    "build_targets",
    "sample_starts",
    "target_specs",
]
