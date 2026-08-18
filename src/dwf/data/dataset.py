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
    to_working_units,
)
from dwf.models.losses import KEY_SPATIAL_WEIGHT
from dwf.slots import diurnal_reference_index
from dwf.tables import FOLDS, SLOTS, read_table
from dwf.weighting import spatial_weight

if TYPE_CHECKING:  # pragma: no cover - solo per i tipi
    from dwf.config import Config

# Chiavi dei tensori restituiti, cosi' che loss e valutazione non usino stringhe libere.
KEY_FEATURES = "features"
KEY_SLOT = "start_slot"
# Prefisso delle chiavi che portano il riferimento diurno usato per l'ancoraggio.
KEY_BASELINE_PREFIX = "baseline_"


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
    def longitudes(self) -> np.ndarray:
        return np.asarray(self._store.longitude.values, dtype=np.float32)

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
            nome: to_working_units(
                nome,
                np.asarray(
                    self._store[nome].isel(slot=slice(start, start + length)).values,
                    dtype=np.float32,
                ),
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


def diurnal_baselines(
    config: Config, stats: NormStats, window: dict[str, np.ndarray]
) -> dict[str, torch.Tensor]:
    """Riferimento diurno normalizzato per ogni testa gaussiana, scadenza per scadenza.

    E' l'osservazione piu' recente alla stessa ora del giorno del bersaglio: la rete vi
    somma sopra la propria uscita e impara quindi soltanto lo scarto. Serve la sola
    parte osservata della finestra, quindi la funzione va bene sia in addestramento sia
    in inferenza, dove il futuro non esiste.
    """
    if not config.model.anchor_diurnal:
        return {}
    slot_al_giorno = config.time.slots_per_day
    input_slots = config.windows.input_slots
    indici = [
        diurnal_reference_index(scadenza, input_slots, slot_al_giorno)
        for scadenza in range(config.windows.output_slots)
    ]
    ampiezza_occorrenza = float(getattr(config.model, "occurrence_anchor_logit", 0.0))
    riferimenti: dict[str, torch.Tensor] = {}
    for spec in target_specs(config):
        if spec.name not in window:
            if spec.head == "gaussian":
                raise DatasetError(f"Manca la variabile {spec.name!r} per l'ancoraggio")
            continue
        if spec.head == "gaussian":
            valori = stats.normalize(spec.name, window[spec.name][indici])
            riferimenti[spec.name] = torch.from_numpy(np.ascontiguousarray(valori))
        elif spec.head == "hurdle" and ampiezza_occorrenza > 0.0 and spec.threshold is not None:
            # Lo scarto viaggia gia' in logit: chi lo somma non deve sapere come e'
            # stato costruito, e l'ampiezza resta un solo numero in configurazione.
            # +ampiezza dove ieri alla stessa ora pioveva, -ampiezza dove non pioveva.
            occorrenza = (window[spec.name][indici] > spec.threshold).astype(np.float32)
            scarto = ampiezza_occorrenza * (2.0 * occorrenza - 1.0)
            riferimenti[spec.name] = torch.from_numpy(np.ascontiguousarray(scarto))
    return riferimenti


def split_baselines(
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Separa i bersagli dai riferimenti di ancoraggio dentro un batch.

    I riferimenti viaggiano nel campione perche' dipendono dalla finestra, ma non sono
    bersagli: passarli alla loss la farebbe lamentare di chiavi sconosciute.
    """
    bersagli: dict[str, torch.Tensor] = {}
    riferimenti: dict[str, torch.Tensor] = {}
    for chiave, valore in batch.items():
        if chiave in (KEY_FEATURES, KEY_SLOT):
            continue
        if chiave.startswith(KEY_BASELINE_PREFIX):
            riferimenti[chiave[len(KEY_BASELINE_PREFIX) :]] = valore
        else:
            bersagli[chiave] = valore
    return bersagli, riferimenti


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


def usable_mask(config: Config) -> np.ndarray:
    """Quali slot sono davvero nello store, letti dal catalogo."""
    catalogo = read_table(SLOTS, config.tables_dir).sort("slot_index")
    return catalogo.get_column("usable").to_numpy().astype(bool)


def split_bounds(config: Config, fold: int, split: str) -> tuple[int, int] | None:
    """Estremi del blocco, come intervallo semiaperto ``[inizio, fine)``.

    I confini appartengono alla partizione temporale e non dipendono dalla finestra:
    sono l'unico dato di `folds.parquet` che resta valido anche cambiando
    `input_slots`.
    """
    tabella = read_table(FOLDS, config.tables_dir)
    selezione = tabella.filter((pl.col("fold") == fold) & (pl.col("split") == split))
    if not selezione.height:
        return None
    estremi = selezione.select(
        pl.col("slot_index").min().alias("min"), pl.col("slot_index").max().alias("max")
    ).row(0)
    return int(estremi[0]), int(estremi[1]) + 1


def sample_starts(config: Config, fold: int, split: str) -> list[int]:
    """Slot iniziali ammessi per un fold e uno split, alla finestra **corrente**.

    Gli inizi non vengono letti da `folds.parquet` ma rienumerati dai confini del
    blocco. La colonna `is_sample_start` e' calcolata una volta sola, con la finestra
    in vigore al momento dell'ingestione, e resta congelata: riusarla con una finestra
    diversa produce due errori opposti e silenziosi.

    Con una finestra piu' **corta** si perdono inizi che il blocco consentirebbe,
    perche' nessuno li ha mai marcati: l'impostazione viene misurata con meno dati di
    quelli che le spettano. Con una finestra piu' **lunga** si tengono inizi la cui
    coda esce dal blocco e finisce nell'intervallo cuscinetto, o oltre: quei campioni
    non dovrebbero esistere, e avvicinano i bersagli di addestramento al blocco
    successivo proprio dove il cuscinetto serviva a separarli.

    Rienumerare costa una scansione e rende il conteggio corretto per costruzione.
    """
    confini = split_bounds(config, fold, split)
    if confini is None:
        return []
    inizio_blocco, fine_blocco = confini

    utilizzabili = usable_mask(config)
    finestra = config.windows.input_slots + config.windows.output_slots
    ultimo_inizio = min(fine_blocco, utilizzabili.size) - finestra
    return [
        inizio
        for inizio in range(inizio_blocco, ultimo_inizio + 1)
        if bool(utilizzabili[inizio : inizio + finestra].all())
    ]


def data_fingerprint(config: Config, fold: int) -> dict[str, object]:
    """Su quali dati e' stato addestrato un modello, in forma confrontabile.

    Ingerire altri mesi riscrive `slots.parquet` e `folds.parquet`: gli stessi indici
    numerici passano a indicare istanti diversi e i confini fra addestramento,
    validazione e test si spostano. Un checkpoint precedente continuerebbe a caricarsi
    senza lamentarsi, e la sua valutazione girerebbe su finestre che al momento
    dell'addestramento stavano dall'altra parte del confine. Nessun errore visibile, un
    risultato falso e ottimista.

    Registrare qui la forma dei dati permette di accorgersene dopo, confrontando questa
    impronta con quella del momento.
    """
    utilizzabili = usable_mask(config)
    impronta: dict[str, object] = {
        "input_slots": int(config.windows.input_slots),
        "output_slots": int(config.windows.output_slots),
        "slot_utilizzabili": int(utilizzabili.sum()),
        "slot_catalogati": int(utilizzabili.size),
    }
    for split in ("train", "val", "test"):
        inizi = sample_starts(config, fold, split)
        impronta[split] = {
            "finestre": len(inizi),
            "primo": int(inizi[0]) if inizi else None,
            "ultimo": int(inizi[-1]) if inizi else None,
        }
    return impronta


def confronta_impronte(
    registrata: dict[str, object] | None, corrente: dict[str, object]
) -> list[str]:
    """Differenze fra i dati di addestramento e quelli attuali.

    Un elenco vuoto significa che i due insiemi coincidono. L'assenza dell'impronta non
    e' assenza di differenze: viene detta a parte, perche' un checkpoint che non
    dichiara la propria origine resta non verificato.
    """
    if not registrata:
        return [
            "Il checkpoint non dichiara su quali dati e' stato addestrato: e' stato "
            "salvato prima che l'impronta venisse registrata. Non si puo' verificare "
            "che i confini dei fold siano ancora quelli."
        ]

    differenze: list[str] = []
    for chiave, etichetta in (
        ("input_slots", "slot in ingresso"),
        ("output_slots", "slot in uscita"),
        ("slot_utilizzabili", "slot utilizzabili nello store"),
        ("slot_catalogati", "slot catalogati"),
    ):
        prima, adesso = registrata.get(chiave), corrente.get(chiave)
        if prima != adesso:
            differenze.append(f"{etichetta}: {prima} all'addestramento, {adesso} adesso")

    for split in ("train", "val", "test"):
        prima = registrata.get(split) or {}
        adesso = corrente.get(split) or {}
        if not isinstance(prima, dict) or not isinstance(adesso, dict):
            continue
        if prima.get("primo") != adesso.get("primo") or prima.get("ultimo") != adesso.get(
            "ultimo"
        ):
            differenze.append(
                f"confini di {split}: {prima.get('primo')}-{prima.get('ultimo')} "
                f"all'addestramento, {adesso.get('primo')}-{adesso.get('ultimo')} adesso"
            )
        elif prima.get("finestre") != adesso.get("finestre"):
            differenze.append(
                f"finestre di {split}: {prima.get('finestre')} all'addestramento, "
                f"{adesso.get('finestre')} adesso"
            )
    return differenze


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
        deterministic_crops: bool = False,
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
        self.anchor_diurnal = config.model.anchor_diurnal
        self.deterministic_crops = deterministic_crops
        self._seed = seed
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
        riga, colonna = self._crop_origin(index)
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
        campione.update(self._diurnal_baselines(ritagliata))
        campione[KEY_SPATIAL_WEIGHT] = torch.from_numpy(
            self._spatial_weight(latitudini, colonna)
        )
        return campione

    def _spatial_weight(self, latitudes: np.ndarray, column: int) -> np.ndarray:
        """Peso per punto del ritaglio corrente.

        Va ricalcolato per ogni ritaglio e non una volta sola: con ritagli casuali sia
        le latitudini sia la posizione del punto di interesse cambiano da un campione
        all'altro, e un peso fisso finirebbe applicato alla porzione sbagliata di
        dominio.
        """
        parametri = self.config.training.spatial_weighting
        longitudini = self.reader.longitudes
        if self.crop_size is not None:
            longitudini = longitudini[column : column + self.crop_size]
        return spatial_weight(
            latitudes,
            longitudini,
            use_area=parametri.use_area,
            focus_gain=parametri.focus_gain,
            focus_radius_deg=parametri.focus_radius_deg,
            center_lat=parametri.focus_lat,
            center_lon=parametri.focus_lon,
        )

    def _diurnal_baselines(self, window: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """Riferimenti di ancoraggio del campione, col prefisso che li distingue dai target."""
        return {
            f"{KEY_BASELINE_PREFIX}{nome}": valore
            for nome, valore in diurnal_baselines(self.config, self.stats, window).items()
        }

    def _crop_origin(self, index: int) -> tuple[int, int]:
        if self.crop_size is None:
            return 0, 0
        altezza, larghezza = self.reader.shape
        if self.deterministic_crops:
            # Il ritaglio dipende solo dall'indice, non da quante volte il dataset e'
            # stato percorso: in validazione un ritaglio diverso a ogni epoca aggiunge
            # rumore proprio alla misura che decide quale epoca conservare.
            generatore = np.random.default_rng((self._seed, index))
        else:
            generatore = self._rng
        riga = int(generatore.integers(0, altezza - self.crop_size + 1))
        colonna = int(generatore.integers(0, larghezza - self.crop_size + 1))
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

    Con ``windows_per_batch`` maggiore di uno il lotto attinge da piu' finestre. Il
    risparmio di letture resta, perche' le finestre del gruppo stanno insieme nella cache
    del lettore, ma il gradiente smette di descrivere una sola situazione
    meteorologica: con una finestra per lotto ogni passo di ottimizzazione vede un solo
    giorno, e le sue peculiarita' pesano come se fossero regola.
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
        windows_per_batch: int = 1,
    ) -> None:
        self.n_windows = n_windows
        self.crops_per_window = max(1, crops_per_window)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_batches = max_batches
        self.windows_per_batch = max(1, min(windows_per_batch, max(1, n_windows)))
        self._rng = np.random.default_rng(seed)

    @property
    def crops_per_batch_per_window(self) -> int:
        """Quanti ritagli ogni finestra del gruppo contribuisce a un lotto."""
        return max(1, self.batch_size // self.windows_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        ordine = np.arange(self.n_windows)
        if self.shuffle:
            self._rng.shuffle(ordine)

        per_finestra = self.crops_per_batch_per_window
        prodotti = 0
        for avvio in range(0, len(ordine), self.windows_per_batch):
            gruppo = ordine[avvio : avvio + self.windows_per_batch]
            indici_gruppo = [
                list(
                    range(
                        int(finestra) * self.crops_per_window,
                        int(finestra) * self.crops_per_window + self.crops_per_window,
                    )
                )
                for finestra in gruppo
            ]
            for taglio in range(0, self.crops_per_window, per_finestra):
                lotto = [
                    indice
                    for indici in indici_gruppo
                    for indice in indici[taglio : taglio + per_finestra]
                ]
                if not lotto:
                    continue
                yield lotto
                prodotti += 1
                if self.max_batches is not None and prodotti >= self.max_batches:
                    return

    def __len__(self) -> int:
        gruppi = (self.n_windows + self.windows_per_batch - 1) // self.windows_per_batch
        per_gruppo = max(
            1,
            (self.crops_per_window + self.crops_per_batch_per_window - 1)
            // self.crops_per_batch_per_window,
        )
        totale = gruppi * per_gruppo
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
        # Le finestre che compongono un lotto devono stare tutte in cache insieme,
        # altrimenti attingere da piu' finestre le farebbe rileggere una per elemento e
        # il risparmio del raggruppamento andrebbe perduto.
        cache_size=max(2, config.training.windows_per_batch + 1),
        static_path=config.static_path if layout.static_variables else None,
        static_variables=layout.static_variables,
    )


__all__ = [
    "KEY_BASELINE_PREFIX",
    "KEY_FEATURES",
    "KEY_SLOT",
    "DatasetError",
    "TargetSpec",
    "WeatherWindowDataset",
    "WindowBatchSampler",
    "ZarrWindowReader",
    "build_reader",
    "build_targets",
    "diurnal_baselines",
    "sample_starts",
    "split_baselines",
    "target_specs",
]
