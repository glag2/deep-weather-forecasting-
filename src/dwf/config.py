"""Configurazione validata del progetto.

Un solo oggetto ``Config`` descrive area, periodo, variabili, finestre temporali,
modello e percorsi, cosi' che download, preprocessing, training e inferenza non
possano divergere su parametri condivisi (griglia, slot orari, lista variabili).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dwf.slots import accumulation_coverage, required_hours, validate_accumulation_fits
from dwf.variables import VariableSpec, spec_by_cds_name

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


class TimeConfig(_Base):
    """Periodo storico e cadenza degli slot previsti."""

    start: Annotated[str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
    end: Annotated[str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
    slot_hours: Annotated[list[Annotated[int, Field(ge=0, le=23)]], Field(min_length=1)]
    accum_window_hours: Annotated[int, Field(ge=1, le=24)] = 8

    @model_validator(mode="after")
    def _check_period(self) -> Self:
        if self.start > self.end:
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

    def months(self) -> list[tuple[int, int]]:
        """Elenco inclusivo di (anno, mese) coperto dalla configurazione."""
        start_year, start_month = (int(part) for part in self.start.split("-"))
        end_year, end_month = (int(part) for part in self.end.split("-"))
        out: list[tuple[int, int]] = []
        year, month = start_year, start_month
        while (year, month) <= (end_year, end_month):
            out.append((year, month))
            month += 1
            if month == 13:
                year, month = year + 1, 1
        return out


class VariablesConfig(_Base):
    """Variabili richieste al CDS, divise per modalita' di campionamento."""

    instantaneous: Annotated[list[str], Field(min_length=1)]
    accumulated: list[str] = []
    static: list[str] = []

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
        return [*self.instantaneous, *self.accumulated]

    @property
    def dynamic_specs(self) -> list[VariableSpec]:
        return [spec_by_cds_name(name) for name in self.dynamic_cds_names]

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

    @model_validator(mode="after")
    def _check_lags(self) -> Self:
        if len(set(self.tendency_lags)) != len(self.tendency_lags):
            raise ValueError("features.tendency_lags contiene duplicati")
        return self


class WindowsConfig(_Base):
    """Lunghezza della finestra di input e dell'orizzonte previsto, in slot."""

    input_slots: Annotated[int, Field(ge=1)] = 21
    output_slots: Annotated[int, Field(ge=1)] = 9


class SplitConfig(_Base):
    """Suddivisione temporale contigua in train / validation / test."""

    train_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.7
    val_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.15
    gap_slots: Annotated[int, Field(ge=0)] = 30

    @model_validator(mode="after")
    def _check_fractions(self) -> Self:
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                "train_fraction + val_fraction deve essere < 1.0 per lasciare spazio al test"
            )
        return self


class ModelConfig(_Base):
    """Dimensionamento della rete."""

    base_channels: Annotated[int, Field(ge=8)] = 48
    depth: Annotated[int, Field(ge=1, le=5)] = 3
    blocks_per_level: Annotated[int, Field(ge=1, le=4)] = 2
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.0


class LossWeights(_Base):
    gaussian: Annotated[float, Field(ge=0.0)] = 1.0
    precip_occurrence: Annotated[float, Field(ge=0.0)] = 0.5
    precip_amount: Annotated[float, Field(ge=0.0)] = 1.0
    snow_fraction: Annotated[float, Field(ge=0.0)] = 0.5


class TrainingConfig(_Base):
    """Iperparametri di ottimizzazione."""

    seed: int = 1234
    epochs: Annotated[int, Field(ge=1)] = 20
    batch_size: Annotated[int, Field(ge=1)] = 4
    crop_size: Annotated[int, Field(ge=16)] | None = 96
    samples_per_epoch: Annotated[int, Field(ge=1)] = 512
    learning_rate: Annotated[float, Field(gt=0.0)] = 3e-4
    weight_decay: Annotated[float, Field(ge=0.0)] = 1e-5
    grad_clip_norm: Annotated[float, Field(gt=0.0)] | None = 1.0
    num_workers: Annotated[int, Field(ge=0)] = 0
    loss_weights: LossWeights = LossWeights()

    @model_validator(mode="after")
    def _check_crop(self) -> Self:
        if self.crop_size is not None and self.crop_size % 8 != 0:
            raise ValueError("crop_size deve essere multiplo di 8 per i downsampling della rete")
        return self


class PathsConfig(_Base):
    """Percorsi relativi alla radice dei dati."""

    data_root: str = "data"
    raw_subdir: str = "raw"
    zarr_name: str = "era5_slots.zarr"
    static_name: str = "era5_static.zarr"
    tables_subdir: str = "tables"
    artifacts_subdir: str = "artifacts"


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
