"""Configurazione validata del progetto.

Un solo oggetto ``Config`` descrive area, periodo, variabili, finestre temporali,
modello e percorsi, cosi' che download, preprocessing, training e inferenza non
possano divergere su parametri condivisi (griglia, slot orari, lista variabili).
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dwf.slots import (
    SplitLayout,
    accumulation_coverage,
    build_rolling_folds,
    build_split_layout,
    days_for_month,
    months_between,
    required_hours,
    slot_times_between,
    validate_accumulation_fits,
)
from dwf.variables import VariableSpec, pressure_spec, spec_by_cds_name

# Risoluzione nativa di ERA5 single levels: qualunque `grid` richiesto al CDS
# deve esserne un multiplo, altrimenti MARS interpola creando punti non allineati.
NATIVE_GRID_DEG = 0.25


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegionConfig(_Base):
    """Ritaglio geografico richiesto al CDS."""

    name: str = "euro_atlantic"
    north: Annotated[float, Field(ge=-90.0, le=90.0)]
    west: Annotated[float, Field(ge=-180.0, le=180.0)]
    south: Annotated[float, Field(ge=-90.0, le=90.0)]
    east: Annotated[float, Field(ge=-180.0, le=180.0)]
    grid: Annotated[float, Field(gt=0.0, le=10.0)] | None = NATIVE_GRID_DEG

    @model_validator(mode="after")
    def _check_extent(self) -> Self:
        if self.north <= self.south:
            raise ValueError(f"north ({self.north}) deve essere maggiore di south ({self.south})")
        if self.east <= self.west:
            raise ValueError(f"east ({self.east}) deve essere maggiore di west ({self.west})")
        step = self.grid_step
        # Controllo piu' fondamentale per primo: se il passo non e' un multiplo del
        # nativo, il messaggio sull'allineamento dell'estensione sarebbe fuorviante.
        ratio = step / NATIVE_GRID_DEG
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                f"grid ({step}) non e' un multiplo della risoluzione nativa "
                f"{NATIVE_GRID_DEG}: MARS interpolerebbe su punti non allineati."
            )
        extents = (
            ("latitudine", self.north - self.south),
            ("longitudine", self.east - self.west),
        )
        for label, span in extents:
            n_steps = span / step
            if abs(n_steps - round(n_steps)) > 1e-6:
                raise ValueError(
                    f"L'estensione in {label} ({span}) non e' un multiplo intero della "
                    f"griglia ({step}): la griglia risultante non sarebbe allineata."
                )
        return self

    @property
    def grid_step(self) -> float:
        return NATIVE_GRID_DEG if self.grid is None else self.grid

    @property
    def n_lat(self) -> int:
        return round((self.north - self.south) / self.grid_step) + 1

    @property
    def n_lon(self) -> int:
        return round((self.east - self.west) / self.grid_step) + 1

    @property
    def cds_area(self) -> list[float]:
        """Area nell'ordine richiesto dal CDS: [North, West, South, East]."""
        return [self.north, self.west, self.south, self.east]

    @property
    def latitudes(self) -> np.ndarray:
        """Latitudini della griglia, **decrescenti** come le fornisce ERA5."""
        return np.linspace(self.north, self.south, self.n_lat)

    @property
    def longitudes(self) -> np.ndarray:
        """Longitudini della griglia, crescenti."""
        return np.linspace(self.west, self.east, self.n_lon)


class TimeConfig(_Base):
    """Periodo storico e cadenza degli slot previsti."""

    start: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    end: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    slot_hours: Annotated[list[Annotated[int, Field(ge=0, le=23)]], Field(min_length=1)]
    accum_window_hours: Annotated[int, Field(ge=1, le=24)] = 8

    @model_validator(mode="after")
    def _check_period(self) -> Self:
        # La regex accetta la forma ma non la validita': 2024-02-31 la supererebbe.
        try:
            start_date, end_date = self.start_date, self.end_date
        except ValueError as exc:
            raise ValueError(f"data non valida: {exc}") from None
        if start_date > end_date:
            raise ValueError(f"start ({self.start}) successivo a end ({self.end})")
        if sorted(self.slot_hours) != self.slot_hours:
            raise ValueError("slot_hours deve essere ordinato in modo crescente")
        if len(set(self.slot_hours)) != len(self.slot_hours):
            raise ValueError("slot_hours contiene duplicati")
        validate_accumulation_fits(self.slot_hours, self.accum_window_hours)
        return self

    @property
    def accumulation_coverage(self) -> tuple[int, int]:
        """Ore del giorno coperte dalle finestre di accumulo e conteggi ridondanti."""
        return accumulation_coverage(self.slot_hours, self.accum_window_hours)

    @property
    def hourly_hours(self) -> list[int]:
        """Ore da richiedere al CDS per le variabili cumulate."""
        return required_hours(self.slot_hours, self.accum_window_hours)

    @property
    def slots_per_day(self) -> int:
        return len(self.slot_hours)

    @property
    def start_date(self) -> date:
        return date.fromisoformat(self.start)

    @property
    def end_date(self) -> date:
        return date.fromisoformat(self.end)

    def months(self) -> list[tuple[int, int]]:
        """Mesi toccati dal periodo, estremi inclusi. Il primo e l'ultimo sono parziali."""
        return months_between(self.start_date, self.end_date)

    def days_in_month(self, year: int, month: int) -> list[int]:
        """Giorni da richiedere per quel mese, limitati al periodo configurato."""
        return days_for_month(year, month, self.start_date, self.end_date)

    def slot_times(self) -> list[datetime]:
        """Tutti gli istanti di slot attesi nel periodo, in ordine crescente."""
        return slot_times_between(self.start_date, self.end_date, self.slot_hours)

    @property
    def n_slots(self) -> int:
        """Numero di slot attesi se non manca nulla."""
        n_days = (self.end_date - self.start_date).days + 1
        return n_days * self.slots_per_day


class PressureLevelConfig(_Base):
    """Una variabile ERA5 richiesta a un singolo livello di pressione."""

    variable: str
    level: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def _check_registered(self) -> Self:
        # `pressure_spec` segnala nome e livello sconosciuti con KeyError, che pydantic
        # non converte in ValidationError.
        try:
            pressure_spec(self.variable, self.level)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None
        return self

    @property
    def spec(self) -> VariableSpec:
        """Spec con il nome interno univoco (variabile + livello)."""
        return pressure_spec(self.variable, self.level)


class VariablesConfig(_Base):
    """Variabili richieste al CDS, divise per modalita' di campionamento."""

    instantaneous: Annotated[list[str], Field(min_length=1)]
    accumulated: list[str] = []
    static: list[str] = []
    # Campi su livelli di pressione: collection CDS distinta e parametro `pressure_level`
    # in piu'. Campionati agli slot come gli istantanei di superficie.
    pressure: list[PressureLevelConfig] = []

    @model_validator(mode="after")
    def _check_pressure(self) -> Self:
        coppie = [(entry.variable, entry.level) for entry in self.pressure]
        if len(set(coppie)) != len(coppie):
            raise ValueError("variables.pressure contiene duplicati (variabile, livello)")
        return self

    @model_validator(mode="after")
    def _check_kinds(self) -> Self:
        for field_name in ("instantaneous", "accumulated", "static"):
            names = getattr(self, field_name)
            if len(set(names)) != len(names):
                raise ValueError(f"variables.{field_name} contiene duplicati")
            for name in names:
                # `spec_by_cds_name` segnala un nome sconosciuto con KeyError, che
                # pydantic non converte in ValidationError: solo ValueError e
                # AssertionError vengono catturati. Senza questa conversione un
                # refuso nello YAML produrrebbe un traceback grezzo.
                try:
                    spec = spec_by_cds_name(name)
                except KeyError as exc:
                    raise ValueError(exc.args[0]) from None
                if spec.kind != field_name:
                    raise ValueError(
                        f"{name!r} e' registrata come {spec.kind!r} ma compare in "
                        f"variables.{field_name}"
                    )
        return self

    @property
    def dynamic_cds_names(self) -> list[str]:
        """Nomi CDS dei soli campi di superficie: i livelli di pressione stanno a parte.

        Lo stesso nome CDS puo' comparire a piu' livelli, quindi per quelli non esiste un
        elenco di nomi univoci: usare `pressure_specs`.
        """
        return [*self.instantaneous, *self.accumulated]

    @property
    def pressure_specs(self) -> list[VariableSpec]:
        return [entry.spec for entry in self.pressure]

    def pressure_by_level(self) -> dict[int, tuple[str, ...]]:
        """Nomi CDS raggruppati per livello, in ordine di livello crescente.

        Il CDS restituisce il prodotto cartesiano di `variable` x `pressure_level`:
        raggruppare per livello e' quindi il solo modo di chiedere insiemi diversi di
        variabili a livelli diversi senza scaricare campi non richiesti.
        """
        gruppi: dict[int, list[str]] = {}
        for entry in self.pressure:
            gruppi.setdefault(entry.level, []).append(entry.variable)
        return {livello: tuple(gruppi[livello]) for livello in sorted(gruppi)}

    @property
    def dynamic_specs(self) -> list[VariableSpec]:
        # I livelli di pressione stanno in coda: l'ordine dei canali di input segue
        # questa lista, e anteporli sposterebbe tutti i canali esistenti.
        return [spec_by_cds_name(name) for name in self.dynamic_cds_names] + self.pressure_specs

    @property
    def static_specs(self) -> list[VariableSpec]:
        return [spec_by_cds_name(name) for name in self.static]

    @property
    def dynamic_short_names(self) -> list[str]:
        return [spec.short_name for spec in self.dynamic_specs]

    @property
    def static_short_names(self) -> list[str]:
        return [spec.short_name for spec in self.static_specs]


class TargetConfig(_Base):
    """Una variabile prevista e il tipo di testa probabilistica che la modella.

    - ``gaussian``: media e log-varianza, loss NLL gaussiana. Per variabili continue
      e approssimativamente simmetriche come la temperatura.
    - ``hurdle``: probabilita' di superare ``threshold`` piu' quantita' condizionata.
      Necessaria per la precipitazione, che ha una massa di probabilita' concentrata
      esattamente in zero e che una gaussiana non puo' rappresentare.
    - ``fraction_of``: logit del rapporto rispetto a una variabile di riferimento,
      addestrato solo dove il riferimento supera la propria soglia. Modella "nevica
      invece di piovere" senza dover prevedere due volte la quantita' totale.
    """

    name: str
    head: Literal["gaussian", "hurdle", "fraction_of"]
    # Soglia in unita' fisiche della variabile; obbligatoria per `hurdle`.
    threshold: Annotated[float, Field(ge=0.0)] | None = None
    # Variabile di riferimento; obbligatoria per `fraction_of`.
    reference: str | None = None

    @model_validator(mode="after")
    def _check_head_arguments(self) -> Self:
        if self.head == "hurdle" and self.threshold is None:
            raise ValueError(f"target {self.name!r}: la testa 'hurdle' richiede 'threshold'")
        if self.head == "fraction_of":
            if self.reference is None:
                raise ValueError(
                    f"target {self.name!r}: la testa 'fraction_of' richiede 'reference'"
                )
            if self.reference == self.name:
                raise ValueError(f"target {self.name!r}: 'reference' non puo' essere se stessa")
        if self.head != "hurdle" and self.threshold is not None:
            raise ValueError(
                f"target {self.name!r}: 'threshold' e' ammesso solo con la testa 'hurdle'"
            )
        if self.head != "fraction_of" and self.reference is not None:
            raise ValueError(
                f"target {self.name!r}: 'reference' e' ammesso solo con la testa 'fraction_of'"
            )
        return self

    @property
    def n_output_channels(self) -> int:
        """Canali prodotti per ogni lead time da questa testa."""
        return {"gaussian": 2, "hurdle": 2, "fraction_of": 1}[self.head]


class FeaturesConfig(_Base):
    """Canali derivati aggiunti allo stack di input."""

    tendency_lags: list[Annotated[int, Field(ge=1)]] = [1, 3, 9]
    include_wind_speed: bool = True
    include_time_encoding: bool = True
    include_static: bool = True
    include_latitude_encoding: bool = True
    # Raggi, in punti di griglia, dei descrittori topografici di vicinato ricavati dalla
    # quota. Vuoto significa disattivati: l'effetto va misurato prima di adottarlo.
    topographic_radii: list[Annotated[int, Field(ge=1)]] = []

    @model_validator(mode="after")
    def _check_lags(self) -> Self:
        if len(set(self.tendency_lags)) != len(self.tendency_lags):
            raise ValueError("features.tendency_lags contiene duplicati")
        if len(set(self.topographic_radii)) != len(self.topographic_radii):
            raise ValueError("features.topographic_radii contiene duplicati")
        if self.topographic_radii != sorted(self.topographic_radii):
            raise ValueError("features.topographic_radii deve essere crescente")
        if self.topographic_radii and not self.include_static:
            raise ValueError(
                "features.topographic_radii richiede include_static: i descrittori si "
                "ricavano dal campo di quota, che senza i campi statici non viene letto"
            )
        return self


class WindowsConfig(_Base):
    """Lunghezza della finestra di input e dell'orizzonte previsto, in slot."""

    input_slots: Annotated[int, Field(ge=1)] = 21
    output_slots: Annotated[int, Field(ge=1)] = 9


class SplitConfig(_Base):
    """Suddivisione temporale contigua in train / validation / test."""

    mode: Literal["rolling", "chronological"] = "rolling"
    gap_slots: Annotated[int, Field(ge=0)] = 30

    # Usati solo con mode="chronological".
    train_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.7
    val_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.15

    # Usati solo con mode="rolling", espressi in giorni.
    initial_train_days: Annotated[int, Field(ge=1)] = 330
    val_days: Annotated[int, Field(ge=1)] = 60
    test_days: Annotated[int, Field(ge=1)] = 90
    step_days: Annotated[int, Field(ge=1)] = 90
    expanding: bool = True
    n_folds: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def _check_fractions(self) -> Self:
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                "train_fraction + val_fraction deve essere < 1.0 per lasciare spazio al test"
            )
        # Con passo maggiore del test i blocchi lasciano buchi non valutati; con passo
        # minore si sovrappongono e le metriche pesano due volte gli stessi giorni.
        if self.step_days > self.test_days:
            raise ValueError(
                f"step_days ({self.step_days}) maggiore di test_days ({self.test_days}): "
                "alcuni periodi non verrebbero mai valutati"
            )
        return self


class ModelConfig(_Base):
    """Dimensionamento della rete e parametrizzazione della sua uscita."""

    base_channels: Annotated[int, Field(ge=8)] = 48
    depth: Annotated[int, Field(ge=1, le=5)] = 3
    blocks_per_level: Annotated[int, Field(ge=1, le=4)] = 2
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.0
    # Se attivo, le teste gaussiane prevedono lo scarto dall'ultima osservazione alla
    # stessa ora del giorno invece del valore assoluto. Resta configurabile perche' il
    # guadagno va misurato, non dato per scontato.
    anchor_diurnal: bool = True
    # Ampiezza, in logit, dell'ancoraggio delle teste di occorrenza alla persistenza
    # diurna. A 0 l'uscita e' identica a prima e nulla cambia. Sopra 0 la rete parte
    # dalla risposta "come ieri alla stessa ora" e impara lo scarto, esattamente come
    # gia' fanno le teste gaussiane. Serve perche' la probabilita' di pioggia misurata
    # varia con la scadenza circa nove volte meno del vero, e alla scadenza piu' breve
    # perde contro la persistenza. Con 1.1 si parte da circa 0,75 dove ieri pioveva e
    # 0,25 dove non pioveva.
    occurrence_anchor_logit: Annotated[float, Field(ge=0.0, le=4.0)] = 0.0
    # Nome della variante di blocco da usare nel corpo della rete.
    variant: str = "conv"
    # Quale architettura costruire. `unet` e' lo scheletro a U convoluzionale, dove
    # `variant` sceglie il blocco; `global` e' la rete rivale, che sostituisce il
    # ragionamento locale con attenzione fra tutti i token di una griglia grossolana.
    # Convivono perche' il confronto va misurato: il campo recettivo *efficace* della
    # rete a U, misurato sul modello addestrato, concentra meta' dell'influenza su una
    # previsione entro 130 km, mentre a tre giorni l'informazione che conta parte da
    # 1500-3000 km.
    architecture: Literal["unet", "global"] = "unet"
    # Parametri della sola architettura `global`. Il costo dell'attenzione cresce col
    # quadrato del numero di token, quindi cala con la quarta potenza di `patch`:
    # dimezzare `patch` costa sedici volte tanto.
    patch: Annotated[int, Field(ge=2, le=32)] = 8
    embed_channels: Annotated[int, Field(ge=32, le=1024)] = 192
    global_blocks: Annotated[int, Field(ge=1, le=12)] = 4
    heads: Annotated[int, Field(ge=1, le=16)] = 4
    # Logit appreso nel denominatore del softmax: una testa puo' non attendere nulla.
    attention_sink: bool = False
    # Ramo a contesto compresso: i token vengono ridotti di questo fattore e attesi in
    # modo denso fra loro. 0 spegne il ramo. Spento per default finche' non e' misurato.
    hca_pool: Annotated[int, Field(ge=0, le=16)] = 0
    hca_blocks: Annotated[int, Field(ge=1, le=6)] = 1

    @model_validator(mode="after")
    def _nucleo_globale_coerente(self) -> Self:
        """Un valore incoerente fallirebbe altrimenti solo alla costruzione della rete.

        A quel punto dati e finestre sono gia' stati preparati, quindi l'errore arriva
        dopo minuti di lavoro e senza dire quale dei due numeri sia quello sbagliato.
        """
        if self.embed_channels % self.heads:
            raise ValueError(
                f"model.embed_channels ({self.embed_channels}) deve essere divisibile "
                f"per model.heads ({self.heads})"
            )
        if self.patch & (self.patch - 1):
            raise ValueError(f"model.patch deve essere una potenza di due: {self.patch}")
        if self.hca_pool == 1:
            raise ValueError(
                "model.hca_pool 1 non comprime nulla: usare 0 per spegnere il ramo"
            )
        return self

    @field_validator("variant")
    @classmethod
    def _variante_registrata(cls, valore: str) -> str:
        """Un nome non registrato fallirebbe solo alla costruzione della rete.

        Con una configurazione modificabile dall'esterno il ritardo conta: l'errore
        arriverebbe dopo la lettura dei dati e la preparazione delle finestre invece che
        subito, e senza dire quali nomi esistano.
        """
        from dwf.models import variants

        disponibili = variants.available()
        if valore not in disponibili:
            raise ValueError(
                f"Variante sconosciuta: {valore!r}. Disponibili: {sorted(disponibili)}"
            )
        return valore


class LossWeights(_Base):
    gaussian: Annotated[float, Field(ge=0.0)] = 1.0
    precip_occurrence: Annotated[float, Field(ge=0.0)] = 0.5
    precip_amount: Annotated[float, Field(ge=0.0)] = 1.0
    snow_fraction: Annotated[float, Field(ge=0.0)] = 0.5
    # Termine spettrale contro la doppia penalizzazione. Zero lo disattiva, cosi' il
    # suo effetto si misura invece di darlo per acquisito.
    spectral: Annotated[float, Field(ge=0.0)] = 0.0


class SpatialWeighting(_Base):
    """Pesi per punto applicati a tutte le teste della perdita."""

    # Peso di area proporzionale al coseno della latitudine: la griglia e' regolare in
    # gradi, quindi senza correzione l'Artico conterebbe quanto le medie latitudini pur
    # coprendo un terzo dell'area.
    use_area: bool = True
    # Attenzione extra attorno al luogo di interesse. Volutamente contenuta: alzarla
    # troppo trasformerebbe un modello di dominio in un modello locale.
    focus_gain: Annotated[float, Field(ge=0.0, le=5.0)] = 0.5
    focus_radius_deg: Annotated[float, Field(gt=0.0)] = 1.5
    focus_lat: Annotated[float, Field(ge=-90.0, le=90.0)] = 46.5031
    focus_lon: Annotated[float, Field(ge=-180.0, le=180.0)] = 12.5308


class TrainingConfig(_Base):
    """Iperparametri di ottimizzazione."""

    seed: int = 1234
    epochs: Annotated[int, Field(ge=1)] = 20
    batch_size: Annotated[int, Field(ge=1)] = 4
    crop_size: Annotated[int, Field(ge=16)] | None = 96
    samples_per_epoch: Annotated[int, Field(ge=1)] = 512
    # Quante finestre temporali distinte compongono un lotto. Con 1 il gradiente di ogni
    # passo descrive una sola situazione meteorologica.
    windows_per_batch: Annotated[int, Field(ge=1)] = 1
    learning_rate: Annotated[float, Field(gt=0.0)] = 3e-4
    # `cmuon` ortogonalizza l'aggiornamento delle matrici interne (Muon con le matrici
    # fuse spezzate, arXiv:2608.02502) e lascia AdamW su norme, bias e i due estremi della
    # rete. Attacca il collo di bottiglia misurato, cioe' il numero di passi, ma i
    # risultati del paper sono a 675M parametri e lotto 1024: qui resta spento finche' non
    # vince sul banco.
    optimizer: Literal["adamw", "cmuon"] = "adamw"
    lr_schedule: Literal["constant", "cosine"] = "constant"
    # Frazione dei passi totali spesa a salire dal passo nullo a quello pieno.
    warmup_fraction: Annotated[float, Field(ge=0.0, lt=0.5)] = 0.05
    weight_decay: Annotated[float, Field(ge=0.0)] = 1e-5
    grad_clip_norm: Annotated[float, Field(gt=0.0)] | None = 1.0
    num_workers: Annotated[int, Field(ge=0)] = 0
    # Disposizione dei tensori in memoria. Misurata su questa CPU con
    # `tmp/diagnostica/velocita_passo.py`: 2780 -> 2565 ms per passo sulla U-Net e
    # 545 -> 502 ms sulla rete globale, cioe' un 8% gratuito su entrambe. Il collo di
    # bottiglia del progetto sono i passi, quindi vale la pena tenerla accesa.
    # Nella stessa misura bfloat16 e' risultato *dannoso* (2780 -> 5603 ms): questa CPU
    # non lo supporta nativamente e l'autocast lo emula. Per questo non esiste un
    # interruttore per la mezza precisione: sarebbe un modo di peggiorare.
    channels_last: bool = True
    loss_weights: LossWeights = LossWeights()
    spatial_weighting: SpatialWeighting = SpatialWeighting()

    @model_validator(mode="after")
    def _check_crop(self) -> Self:
        if self.crop_size is not None and self.crop_size % 8 != 0:
            raise ValueError("crop_size deve essere multiplo di 8 per i downsampling della rete")
        if self.windows_per_batch > self.batch_size:
            raise ValueError(
                f"windows_per_batch {self.windows_per_batch} supera batch_size "
                f"{self.batch_size}: ogni finestra del gruppo deve contribuire almeno "
                f"un ritaglio"
            )
        if self.batch_size % self.windows_per_batch != 0:
            raise ValueError(
                f"batch_size {self.batch_size} non e' divisibile per windows_per_batch "
                f"{self.windows_per_batch}: le finestre contribuirebbero un numero "
                f"diverso di ritagli e il lotto sarebbe sbilanciato"
            )
        return self


class PathsConfig(_Base):
    """Percorsi relativi alla radice dei dati."""

    data_root: str = "data"
    raw_subdir: str = "raw"
    zarr_name: str = "era5_slots.zarr"
    static_name: str = "era5_static.zarr"
    tables_subdir: str = "tables"
    artifacts_subdir: str = "artifacts"

    # I modelli addestrati stanno fuori da `data_root`, alla radice del progetto: sono
    # il risultato del lavoro, non dati grezzi rigenerabili con un nuovo download.
    models_dir: str = "models"

    # Ramo sotto `models_dir` in cui isolare i checkpoint di una singola esecuzione.
    # Serve ai banchi di prova: senza di esso ogni prova scriverebbe e rileggerebbe la
    # stessa cartella, e due esecuzioni contemporanee si sovrascriverebbero il
    # checkpoint fra addestramento e valutazione.
    models_subdir: str = ""

    # Slot per blocco Zarr. Compromesso misurabile: blocchi grandi riducono il numero
    # di letture ma per un crop piccolo trasferiscono dati inutili, blocchi piccoli
    # fanno il contrario. Lo spazio resta non suddiviso perche' l'inferenza legge
    # sempre il dominio intero.
    chunk_slots: Annotated[int, Field(ge=1)] = 8


class DownloadConfig(_Base):
    max_retries: Annotated[int, Field(ge=1, le=10)] = 3
    retry_backoff_seconds: Annotated[float, Field(ge=0.0)] = 30.0


class Config(_Base):
    """Configurazione completa del progetto."""

    region: RegionConfig
    time: TimeConfig
    variables: VariablesConfig
    targets: Annotated[list[TargetConfig], Field(min_length=1)]
    features: FeaturesConfig = FeaturesConfig()
    windows: WindowsConfig = WindowsConfig()
    split: SplitConfig = SplitConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    paths: PathsConfig = PathsConfig()
    download: DownloadConfig = DownloadConfig()

    # Radice usata per risolvere `paths.data_root` quando questo e' relativo.
    project_root: Path = Path.cwd()

    @model_validator(mode="after")
    def _check_targets(self) -> Self:
        available = set(self.variables.dynamic_short_names)
        names = [target.name for target in self.targets]

        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(
                f"targets {missing} non presenti tra le variabili dinamiche scaricate "
                f"({sorted(available)})"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"targets contiene nomi duplicati: {names}")

        # Una testa `fraction_of` divide per il proprio riferimento, che deve quindi
        # essere anch'esso previsto e avere una soglia sotto la quale il rapporto
        # non e' definito.
        by_name = {target.name: target for target in self.targets}
        for target in self.targets:
            if target.reference is None:
                continue
            reference = by_name.get(target.reference)
            if reference is None:
                raise ValueError(
                    f"target {target.name!r}: riferimento {target.reference!r} non e' "
                    f"tra i target previsti ({names})"
                )
            if reference.threshold is None:
                raise ValueError(
                    f"target {target.name!r}: il riferimento {target.reference!r} deve "
                    f"avere una 'threshold' per definire dove il rapporto e' valido"
                )

        max_lag = max(self.features.tendency_lags, default=0)
        if max_lag >= self.windows.input_slots:
            raise ValueError(
                f"tendency_lags contiene un ritardo ({max_lag}) non minore di "
                f"input_slots ({self.windows.input_slots})"
            )
        return self

    @property
    def target_names(self) -> list[str]:
        return [target.name for target in self.targets]

    @property
    def n_output_channels(self) -> int:
        """Canali totali prodotti dalla rete: teste x lead time previsti."""
        per_slot = sum(target.n_output_channels for target in self.targets)
        return per_slot * self.windows.output_slots

    # --- suddivisione temporale ---

    def build_folds(self, n_slots: int | None = None) -> list[SplitLayout]:
        """Fold di valutazione, uno solo se ``split.mode`` e' ``chronological``.

        Accetta ``n_slots`` per poter usare il numero di slot effettivamente presenti
        nell'archivio, che per dati mancanti puo' essere inferiore a quello atteso.
        """
        if n_slots is None:
            n_slots = self.time.n_slots
        per_day = self.time.slots_per_day

        if self.split.mode == "chronological":
            return [
                build_split_layout(
                    n_slots,
                    train_fraction=self.split.train_fraction,
                    val_fraction=self.split.val_fraction,
                    gap_slots=self.split.gap_slots,
                    input_slots=self.windows.input_slots,
                    output_slots=self.windows.output_slots,
                )
            ]

        return build_rolling_folds(
            n_slots,
            initial_train_slots=self.split.initial_train_days * per_day,
            val_slots=self.split.val_days * per_day,
            test_slots=self.split.test_days * per_day,
            gap_slots=self.split.gap_slots,
            step_slots=self.split.step_days * per_day,
            input_slots=self.windows.input_slots,
            output_slots=self.windows.output_slots,
            expanding=self.split.expanding,
            n_folds=self.split.n_folds,
        )

    # --- percorsi derivati ---

    @property
    def data_root(self) -> Path:
        root = Path(self.paths.data_root).expanduser()
        if not root.is_absolute():
            root = self.project_root / root
        return root.resolve()

    def _under_data_root(self, *parts: str) -> Path:
        """Risolve un percorso e verifica che non esca da `data_root`."""
        candidate = self.data_root.joinpath(*parts).resolve()
        if candidate != self.data_root and self.data_root not in candidate.parents:
            raise ValueError(f"Percorso fuori da data_root: {candidate}")
        return candidate

    @property
    def raw_dir(self) -> Path:
        return self._under_data_root(self.paths.raw_subdir)

    @property
    def zarr_path(self) -> Path:
        return self._under_data_root(self.paths.zarr_name)

    @property
    def static_path(self) -> Path:
        return self._under_data_root(self.paths.static_name)

    @property
    def tables_dir(self) -> Path:
        """Directory delle tabelle Polars/Parquet, registro autorevole dei dati puliti."""
        return self._under_data_root(self.paths.tables_subdir)

    @property
    def artifacts_dir(self) -> Path:
        return self._under_data_root(self.paths.artifacts_subdir)

    @property
    def models_dir(self) -> Path:
        """Directory dei modelli addestrati, risolta rispetto alla radice del progetto."""
        root = Path(self.paths.models_dir).expanduser()
        if not root.is_absolute():
            root = self.project_root / root
        return root.resolve()

    def fold_dir(self, fold: int) -> Path:
        """Directory del modello di un fold: unica fonte del percorso per tutto il codice."""
        radice = self.models_dir
        if self.paths.models_subdir:
            radice = (radice / self.paths.models_subdir).resolve()
            # Il ramo arriva da riga di comando: verificare il percorso risolto e' l'unico
            # controllo che regge, perche' `..` e i collegamenti sono gia' stati espansi.
            if radice != self.models_dir and self.models_dir not in radice.parents:
                raise ValueError(
                    f"models_subdir esce dalla cartella dei modelli: {self.paths.models_subdir!r}"
                )
        return radice / f"fold_{fold:02d}"

    # --- costruzione ---

    @classmethod
    def load(cls, path: str | os.PathLike[str], project_root: Path | None = None) -> Config:
        """Carica e valida la configurazione da un file YAML."""
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            payload: Any = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"{config_path} non contiene una mappa YAML valida")
        payload["project_root"] = (project_root or config_path.parent.parent).resolve()
        return cls.model_validate(payload)

    def to_yaml(self) -> str:
        """Serializza la configurazione, utile per registrarla accanto ai checkpoint."""
        payload = self.model_dump(mode="json", exclude={"project_root"})
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
