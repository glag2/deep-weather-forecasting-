"""Costruzione dei canali di input del modello.

La rete riceve un solo tensore `(C, H, W)` che impila tutta la finestra di input: lo
stato di ogni variabile a ogni slot, le tendenze, la velocita' del vento, i campi
statici e la codifica del tempo. Sbagliare l'ordine di quei canali non fa fallire
nulla, produce solo un modello che impara associazioni sbagliate, quindi il layout e'
dichiarato esplicitamente qui ed e' l'unica fonte autorevole: modello, dataset,
normalizzazione, valutazione e notebook lo leggono da questo modulo.

La normalizzazione usa statistiche calcolate **solo sugli slot di train del fold**.
Calcolarle su tutti i dati farebbe filtrare nel modello informazione proveniente dal
futuro, e la validazione a finestra mobile perderebbe significato.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from dwf.slots import time_encoding
from dwf.tables import CHANNELS, NORM_STATS, cast_to_schema
from dwf.variables import spec_by_short_name

if TYPE_CHECKING:  # pragma: no cover - solo per i tipi
    from dwf.config import Config

# Nome della velocita' del vento derivata, trattata come una variabile a se' stante
# perche' ha una propria distribuzione e quindi proprie statistiche.
WIND_SPEED = "wspd"

# Gruppi di canali, nell'ordine in cui vengono impilati.
GROUP_STATE = "state"
GROUP_TENDENCY = "tendency"
GROUP_WIND = "wind_speed"
GROUP_STATIC = "static"
GROUP_LATITUDE = "latitude"
GROUP_TIME = "time"

# Nomi dei canali temporali, nell'ordine restituito da `slots.time_encoding`.
TIME_CHANNEL_NAMES = ("year_sin", "year_cos", "day_sin", "day_cos")

# Vento: la velocita' si costruisce da queste due componenti, se presenti entrambe.
WIND_COMPONENTS = ("u10", "v10")

# Sotto questa deviazione standard la variabile e' costante sullo split di train e
# dividere amplificherebbe solo rumore numerico.
MIN_STD = 1e-6


class FeatureError(RuntimeError):
    """Errore nella costruzione dei canali di input."""


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """Un canale di input, con l'origine che lo ha prodotto."""

    index: int
    name: str
    group: str
    source_variable: str
    # Distanza in slot dall'ultimo slot di input: 0 e' il piu' recente. Vale -1 per i
    # canali che non dipendono dal tempo (statici, latitudine).
    lag: int
    transform: str
    normalized: bool


@dataclass(frozen=True, slots=True)
class InputLayout:
    """Ordine dei canali di input. Contratto condiviso fra dataset e modello."""

    channels: tuple[ChannelSpec, ...]
    dynamic_variables: tuple[str, ...]
    static_variables: tuple[str, ...]
    input_slots: int
    tendency_lags: tuple[int, ...]
    include_wind_speed: bool

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def indices_of_group(self, group: str) -> list[int]:
        return [channel.index for channel in self.channels if channel.group == group]

    def index_of(self, name: str) -> int:
        for channel in self.channels:
            if channel.name == name:
                return channel.index
        raise KeyError(f"Canale assente dal layout: {name!r}")

    @property
    def normalized_variables(self) -> tuple[str, ...]:
        """Variabili per cui servono statistiche di normalizzazione."""
        nomi: list[str] = []
        for channel in self.channels:
            if channel.normalized and channel.source_variable not in nomi:
                nomi.append(channel.source_variable)
        return tuple(nomi)

    @classmethod
    def from_config(cls, config: Config) -> InputLayout:
        dinamiche = tuple(config.variables.dynamic_short_names)
        statiche = tuple(config.variables.static_short_names)
        input_slots = config.windows.input_slots
        lags = tuple(config.features.tendency_lags)

        if max(lags, default=0) >= input_slots:
            raise FeatureError(
                f"features.tendency_lags contiene {max(lags)}, ma la finestra di input "
                f"ha solo {input_slots} slot: la tendenza uscirebbe dalla finestra"
            )

        vento_disponibile = all(nome in dinamiche for nome in WIND_COMPONENTS)
        if config.features.include_wind_speed and not vento_disponibile:
            raise FeatureError(
                f"include_wind_speed richiede {WIND_COMPONENTS} fra le variabili "
                f"dinamiche, presenti: {dinamiche}"
            )

        canali: list[ChannelSpec] = []

        def aggiungi(nome: str, gruppo: str, sorgente: str, lag: int, normalizza: bool) -> None:
            trasformazione = transform_of(sorgente)[0] if normalizza else "identity"
            canali.append(
                ChannelSpec(
                    index=len(canali),
                    name=nome,
                    group=gruppo,
                    source_variable=sorgente,
                    lag=lag,
                    transform=trasformazione,
                    normalized=normalizza,
                )
            )

        # 1. Stato di ogni variabile a ogni slot della finestra, dal piu' vecchio al
        #    piu' recente: l'ordine cronologico rende leggibili i canali di tendenza.
        for variabile in dinamiche:
            for lag in reversed(range(input_slots)):
                aggiungi(f"{variabile}_t-{lag}", GROUP_STATE, variabile, lag, True)

        # 2. Tendenze: differenza fra l'ultimo slot e quello di `lag` slot prima. La
        #    convoluzione vede solo lo spazio, quindi senza queste differenze dovrebbe
        #    dedurre l'evoluzione temporale confrontando canali distanti fra loro.
        for variabile in dinamiche:
            for lag in lags:
                aggiungi(f"{variabile}_delta{lag}", GROUP_TENDENCY, variabile, lag, True)

        # 3. Velocita' del vento: le componenti u e v da sole rendono l'intensita' una
        #    funzione non lineare che la rete dovrebbe ricostruire da capo.
        if config.features.include_wind_speed:
            for lag in reversed(range(input_slots)):
                aggiungi(f"{WIND_SPEED}_t-{lag}", GROUP_WIND, WIND_SPEED, lag, True)

        # 4. Campi invarianti: orografia e maschera terra/mare spiegano gran parte
        #    della struttura spaziale persistente.
        if config.features.include_static:
            for variabile in statiche:
                aggiungi(variabile, GROUP_STATIC, variabile, -1, True)

        # 5. Latitudine: su un dominio di 65 gradi il comportamento fisico cambia
        #    molto con la latitudine, che una convoluzione invariante per traslazione
        #    non puo' dedurre.
        if config.features.include_latitude_encoding:
            aggiungi("lat_norm", GROUP_LATITUDE, "latitude", -1, False)
            aggiungi("lat_cos", GROUP_LATITUDE, "latitude", -1, False)

        # 6. Tempo: stagione e ora del giorno, gia' cicliche.
        if config.features.include_time_encoding:
            for nome in TIME_CHANNEL_NAMES:
                aggiungi(nome, GROUP_TIME, "time", 0, False)

        return cls(
            channels=tuple(canali),
            dynamic_variables=dinamiche,
            static_variables=statiche,
            input_slots=input_slots,
            tendency_lags=lags,
            include_wind_speed=config.features.include_wind_speed,
        )

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "channel_index": channel.index,
                "name": channel.name,
                "group": channel.group,
                "source_variable": channel.source_variable,
                "lag": channel.lag,
                "transform": channel.transform,
                "normalized": channel.normalized,
            }
            for channel in self.channels
        ]

    def to_table(self) -> pl.DataFrame:
        return cast_to_schema(pl.DataFrame(self.describe()), CHANNELS)


# --------------------------------------------------------------------------- #
# Trasformazioni e normalizzazione
# --------------------------------------------------------------------------- #


def apply_transform(values: np.ndarray, transform: str, scale: float = 1.0) -> np.ndarray:
    """Trasformazione stabilizzante applicata prima della standardizzazione.

    La precipitazione ha una distribuzione fortemente asimmetrica: senza `log1p` la
    media e la deviazione standard sarebbero dominate da pochi eventi estremi e i
    valori ordinari finirebbero schiacciati vicino a zero. `scale` porta la variabile
    nell'unita' in cui `log1p` comprime davvero, vedi `VariableSpec.transform_scale`.
    """
    if transform == "identity":
        return values
    if transform == "log1p":
        # I valori sono non negativi per costruzione, ma il clip protegge dal rumore
        # numerico delle cumulate, che puo' produrre negativi minuscoli.
        return np.log1p(np.clip(values, 0.0, None) * scale)
    raise FeatureError(f"Trasformazione sconosciuta: {transform!r}")


def invert_transform(values: np.ndarray, transform: str, scale: float = 1.0) -> np.ndarray:
    """Inversa di `apply_transform`, per riportare le previsioni in unita' fisiche."""
    if transform == "identity":
        return values
    if transform == "log1p":
        return np.expm1(values) / scale
    raise FeatureError(f"Trasformazione sconosciuta: {transform!r}")


def transform_of(variable: str) -> tuple[str, float]:
    """Trasformazione e scala dichiarate per una variabile; identita' se non registrata."""
    if variable == WIND_SPEED:
        return "identity", 1.0
    try:
        spec = spec_by_short_name(variable)
    except (KeyError, ValueError):
        return "identity", 1.0
    return spec.transform, spec.transform_scale


def store_offset_of(variable: str) -> float:
    """Scarto dichiarato fra unita' di archiviazione e unita' di lavoro."""
    if variable == WIND_SPEED:
        return 0.0
    try:
        spec = spec_by_short_name(variable)
    except (KeyError, ValueError):
        return 0.0
    return spec.store_offset


def to_working_units(variable: str, values: np.ndarray) -> np.ndarray:
    """Converte i dati grezzi dello store nell'unita' usata dalla pipeline.

    Va applicata **una sola volta**, subito dopo la lettura. Concentrare qui la
    conversione fa si' che normalizzazione, target, metriche e previsioni parlino tutti
    la stessa unita', e che `normalize` e `denormalize` restino l'una l'inversa
    dell'altra: convertire piu' avanti le renderebbe asimmetriche.
    """
    offset = store_offset_of(variable)
    if offset == 0.0:
        return values
    return (np.asarray(values, dtype=np.float32) + np.float32(offset)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class NormStats:
    """Media e deviazione standard per variabile, con la trasformazione usata."""

    mean: dict[str, float]
    std: dict[str, float]
    transform: dict[str, str]
    scale: dict[str, float]
    computed_on_split: str

    def normalize(self, variable: str, values: np.ndarray) -> np.ndarray:
        self._require(variable)
        trasformati = apply_transform(
            values, self.transform[variable], self.scale[variable]
        )
        return (trasformati - self.mean[variable]) / self.std[variable]

    def denormalize(self, variable: str, values: np.ndarray) -> np.ndarray:
        """Riporta valori normalizzati alle unita' fisiche della variabile."""
        self._require(variable)
        trasformati = values * self.std[variable] + self.mean[variable]
        return invert_transform(trasformati, self.transform[variable], self.scale[variable])

    def _require(self, variable: str) -> None:
        if variable not in self.mean:
            raise FeatureError(
                f"Mancano le statistiche di normalizzazione per {variable!r}. "
                f"Disponibili: {sorted(self.mean)}"
            )

    def to_table(self) -> pl.DataFrame:
        righe = [
            {
                "variable": variabile,
                "transform": self.transform[variabile],
                "transform_scale": self.scale[variabile],
                "mean": self.mean[variabile],
                "std": self.std[variabile],
                "minimum": float("nan"),
                "maximum": float("nan"),
                "n_values": 0,
                "computed_on_split": self.computed_on_split,
            }
            for variabile in sorted(self.mean)
        ]
        return cast_to_schema(pl.DataFrame(righe), NORM_STATS)

    @classmethod
    def from_table(cls, table: pl.DataFrame) -> NormStats:
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        transform: dict[str, str] = {}
        scale: dict[str, float] = {}
        split = ""
        for riga in table.iter_rows(named=True):
            nome = riga["variable"]
            mean[nome] = float(riga["mean"])
            std[nome] = float(riga["std"])
            transform[nome] = riga["transform"]
            scale[nome] = float(riga["transform_scale"])
            split = riga["computed_on_split"]
        return cls(
            mean=mean,
            std=std,
            transform=transform,
            scale=scale,
            computed_on_split=split,
        )


class _Accumulator:
    """Somma e somma dei quadrati, per calcolare media e varianza in un solo passaggio.

    Serve perche' gli slot di train non entrano in memoria tutti insieme: due anni di
    dati a piena risoluzione sono decine di gigabyte.
    """

    __slots__ = ("count", "total", "total_squared")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_squared = 0.0

    def update(self, values: np.ndarray) -> None:
        finiti = values[np.isfinite(values)].astype(np.float64)
        if finiti.size == 0:
            return
        self.count += finiti.size
        self.total += float(finiti.sum())
        self.total_squared += float(np.square(finiti).sum())

    def result(self) -> tuple[float, float]:
        if self.count == 0:
            raise FeatureError("Nessun valore finito per calcolare le statistiche")
        media = self.total / self.count
        varianza = max(self.total_squared / self.count - media * media, 0.0)
        return media, max(varianza**0.5, MIN_STD)


def compute_norm_stats(
    reader: SlotReader,
    variables: Sequence[str],
    slot_indices: Sequence[int],
    *,
    split_name: str = "train",
    batch_slots: int = 64,
) -> NormStats:
    """Statistiche di normalizzazione sugli slot indicati, letti a blocchi.

    `slot_indices` deve contenere **solo** slot di train: e' questa restrizione a
    impedire che informazione del futuro entri nella normalizzazione.
    """
    if not slot_indices:
        raise FeatureError("Servono slot di train per calcolare le statistiche")

    accumulatori = {nome: _Accumulator() for nome in variables}
    dichiarate = {nome: transform_of(nome) for nome in variables}
    trasformazioni = {nome: coppia[0] for nome, coppia in dichiarate.items()}
    scale = {nome: coppia[1] for nome, coppia in dichiarate.items()}
    ordinati = sorted(slot_indices)

    for inizio in range(0, len(ordinati), batch_slots):
        blocco = ordinati[inizio : inizio + batch_slots]
        letti = reader.read_slots(blocco, variables)
        for nome in variables:
            accumulatori[nome].update(
                apply_transform(letti[nome], trasformazioni[nome], scale[nome])
            )

    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for nome in variables:
        media, deviazione = accumulatori[nome].result()
        mean[nome] = media
        std[nome] = deviazione
    return NormStats(
        mean=mean,
        std=std,
        transform=trasformazioni,
        scale=scale,
        computed_on_split=split_name,
    )


# --------------------------------------------------------------------------- #
# Lettore degli slot
# --------------------------------------------------------------------------- #


class SlotReader:
    """Accesso alle variabili dello store, con la velocita' del vento derivata.

    Isola il resto del modulo dal formato di archiviazione: le funzioni di feature
    ricevono array e non sanno se provengono da Zarr, da un test o dalla memoria.
    """

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays

    @property
    def n_slots(self) -> int:
        for array in self._arrays.values():
            if array.ndim == 3:
                return int(array.shape[0])
        raise FeatureError(
            "Nessuna variabile con asse degli istanti: il lettore contiene solo "
            "campi statici e non puo' dire quanti istanti esistano"
        )

    def read_slots(
        self, slot_indices: Sequence[int], variables: Iterable[str]
    ) -> dict[str, np.ndarray]:
        indici = np.asarray(slot_indices, dtype=int)
        letti: dict[str, np.ndarray] = {}
        for nome in variables:
            if nome == WIND_SPEED:
                letti[nome] = wind_speed(
                    self._arrays[WIND_COMPONENTS[0]][indici],
                    self._arrays[WIND_COMPONENTS[1]][indici],
                )
                continue
            if nome not in self._arrays:
                raise FeatureError(
                    f"Variabile assente dallo store: {nome!r}. "
                    f"Disponibili: {sorted(self._arrays)}"
                )
            letti[nome] = to_working_units(nome, self._estrai(nome, indici))
        return letti

    def _estrai(self, nome: str, indici: np.ndarray) -> np.ndarray:
        """Riporta la variabile sull'asse degli istanti richiesto.

        Un campo statico e' memorizzato come (lat, lon) e non ha l'asse degli istanti:
        indicizzarlo con uno slot restituirebbe una riga della griglia invece del campo,
        in silenzio finche' l'indice resta sotto il numero di righe. Va replicato, non
        indicizzato, cosi' chi legge ottiene sempre la stessa prima dimensione.
        """
        dato = self._arrays[nome]
        if dato.ndim == 2:
            campo = np.asarray(dato, dtype=np.float32)
            return np.broadcast_to(campo, (len(indici), *campo.shape))
        return np.asarray(dato[indici], dtype=np.float32)


def wind_speed(u_component: np.ndarray, v_component: np.ndarray) -> np.ndarray:
    """Intensita' del vento dalle due componenti."""
    return np.hypot(
        np.asarray(u_component, dtype=np.float32), np.asarray(v_component, dtype=np.float32)
    )


# --------------------------------------------------------------------------- #
# Costruzione del tensore di input
# --------------------------------------------------------------------------- #


def latitude_channels(latitudes: np.ndarray, n_lon: int) -> np.ndarray:
    """Due canali costanti per riga: latitudine normalizzata e suo coseno.

    Il coseno da' il fattore di convergenza dei meridiani, la latitudine normalizzata
    da' il gradiente nord-sud: insieme distinguono emisferi e fascia climatica.
    """
    lat = np.asarray(latitudes, dtype=np.float32).reshape(-1, 1)
    normalizzata = np.broadcast_to(lat / 90.0, (lat.shape[0], n_lon))
    coseno = np.broadcast_to(np.cos(np.deg2rad(lat)), (lat.shape[0], n_lon))
    return np.stack([normalizzata, coseno]).astype(np.float32)


def build_input_tensor(
    layout: InputLayout,
    window: dict[str, np.ndarray],
    stats: NormStats,
    *,
    static_fields: dict[str, np.ndarray] | None = None,
    latitudes: np.ndarray | None = None,
    reference_time: datetime | None = None,
    slot_hours: Sequence[int] | None = None,
) -> np.ndarray:
    """Impila i canali di input per una singola finestra.

    `window` contiene, per ogni variabile dinamica, un array `(input_slots, H, W)` in
    ordine cronologico. L'ordine dei canali prodotti coincide, per costruzione, con
    quello dichiarato da `layout`.
    """
    attesi = layout.input_slots
    primo = next(iter(window.values()))
    if primo.shape[0] != attesi:
        raise FeatureError(
            f"La finestra ha {primo.shape[0]} slot, il layout ne richiede {attesi}"
        )
    altezza, larghezza = primo.shape[-2:]

    # Le variabili dinamiche si normalizzano una volta sola e si riusano per stato,
    # tendenze e vento: ricalcolarle per ogni canale triplicherebbe il lavoro.
    normalizzate: dict[str, np.ndarray] = {}
    for variabile in layout.dynamic_variables:
        if variabile not in window:
            raise FeatureError(f"Manca la variabile {variabile!r} nella finestra")
        normalizzate[variabile] = stats.normalize(variabile, window[variabile])
    if layout.include_wind_speed:
        velocita = wind_speed(window[WIND_COMPONENTS[0]], window[WIND_COMPONENTS[1]])
        normalizzate[WIND_SPEED] = stats.normalize(WIND_SPEED, velocita)

    canali = np.empty((layout.n_channels, altezza, larghezza), dtype=np.float32)
    tempo: np.ndarray | None = None

    for canale in layout.channels:
        if canale.group in (GROUP_STATE, GROUP_WIND):
            # lag 0 e' l'ultimo slot della finestra.
            canali[canale.index] = normalizzate[canale.source_variable][attesi - 1 - canale.lag]
        elif canale.group == GROUP_TENDENCY:
            serie = normalizzate[canale.source_variable]
            canali[canale.index] = serie[attesi - 1] - serie[attesi - 1 - canale.lag]
        elif canale.group == GROUP_STATIC:
            if static_fields is None or canale.source_variable not in static_fields:
                raise FeatureError(
                    f"Il layout richiede il campo statico {canale.source_variable!r}"
                )
            canali[canale.index] = stats.normalize(
                canale.source_variable, static_fields[canale.source_variable]
            )
        elif canale.group == GROUP_LATITUDE:
            if latitudes is None:
                raise FeatureError("Il layout richiede le latitudini della griglia")
            coppia = latitude_channels(latitudes, larghezza)
            canali[canale.index] = coppia[0] if canale.name == "lat_norm" else coppia[1]
        elif canale.group == GROUP_TIME:
            if reference_time is None or slot_hours is None:
                raise FeatureError("Il layout richiede l'istante di riferimento")
            if tempo is None:
                tempo = time_encoding([reference_time], slot_hours)[0]
            posizione = TIME_CHANNEL_NAMES.index(canale.name)
            canali[canale.index] = tempo[posizione]
        else:  # pragma: no cover - il layout non produce altri gruppi
            raise FeatureError(f"Gruppo di canali sconosciuto: {canale.group!r}")

    return canali


__all__ = [
    "GROUP_LATITUDE",
    "GROUP_STATE",
    "GROUP_STATIC",
    "GROUP_TENDENCY",
    "GROUP_TIME",
    "GROUP_WIND",
    "WIND_SPEED",
    "ChannelSpec",
    "FeatureError",
    "InputLayout",
    "NormStats",
    "SlotReader",
    "apply_transform",
    "build_input_tensor",
    "compute_norm_stats",
    "latitude_channels",
    "store_offset_of",
    "to_working_units",
    "transform_of",
    "wind_speed",
]
