"""Registro delle variabili ERA5 usate dal progetto.

Il CDS accetta nomi lunghi (``2m_temperature``) mentre i file GRIB espongono le
short name ECMWF (``t2m``): serve una tabella esplicita perche' la corrispondenza
non e' derivabile dalla stringa. La colonna ``kind`` distingue i campi istantanei
da quelli cumulati sull'ora precedente, che vanno aggregati e non campionati.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VariableKind = Literal["instantaneous", "accumulated", "static"]
Transform = Literal["identity", "log1p"]


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Metadati di una singola variabile ERA5."""

    cds_name: str
    short_name: str
    kind: VariableKind
    units: str
    description: str
    transform: Transform = "identity"
    # Fattore applicato prima di `log1p`. Serve perche' `log1p` comprime solo valori
    # dell'ordine dell'unita': la precipitazione in metri vale ~7e-4 e `log1p` la
    # lascerebbe praticamente invariata, vanificando la compressione della coda.
    # Portandola in millimetri, 0.1 mm e 70 mm diventano 0.095 e 4.26.
    transform_scale: float = 1.0
    non_negative: bool = False
    # Valore sommato al dato grezzo appena letto dallo store, per portarlo nell'unita'
    # in cui lavora tutta la pipeline. ERA5 esprime le temperature in kelvin mentre il
    # progetto le vuole in gradi Celsius: convertire qui, in lettura, e' l'unico punto
    # che tiene normalizzazione, target, metriche e previsioni nella stessa unita'.
    store_offset: float = 0.0

    @property
    def working_units(self) -> str:
        """Unita' in cui la variabile viene usata, dopo `store_offset`."""
        if self.units == "K" and self.store_offset != 0.0:
            return "degC"
        return self.units


# Zero Celsius in kelvin. ERA5 archivia le temperature in kelvin; il progetto le
# usa in gradi Celsius, che e' l'unita' richiesta per i risultati.
KELVIN_AT_ZERO_CELSIUS = 273.15


_SPECS: tuple[VariableSpec, ...] = (
    # --- istantanei ---
    VariableSpec(
        "2m_temperature", "t2m", "instantaneous", "K", "Temperatura a 2 m",
        store_offset=-KELVIN_AT_ZERO_CELSIUS,
    ),
    VariableSpec(
        "2m_dewpoint_temperature", "d2m", "instantaneous", "K", "Punto di rugiada a 2 m",
        store_offset=-KELVIN_AT_ZERO_CELSIUS,
    ),
    VariableSpec(
        "mean_sea_level_pressure", "msl", "instantaneous", "Pa", "Pressione al livello del mare"
    ),
    VariableSpec("surface_pressure", "sp", "instantaneous", "Pa", "Pressione al suolo"),
    VariableSpec(
        "10m_u_component_of_wind", "u10", "instantaneous", "m s-1", "Vento zonale a 10 m"
    ),
    VariableSpec(
        "10m_v_component_of_wind", "v10", "instantaneous", "m s-1", "Vento meridionale a 10 m"
    ),
    VariableSpec(
        "100m_u_component_of_wind", "u100", "instantaneous", "m s-1", "Vento zonale a 100 m"
    ),
    VariableSpec(
        "100m_v_component_of_wind", "v100", "instantaneous", "m s-1", "Vento meridionale a 100 m"
    ),
    VariableSpec(
        "total_cloud_cover", "tcc", "instantaneous", "0-1", "Copertura nuvolosa totale",
        non_negative=True,
    ),
    VariableSpec(
        "low_cloud_cover", "lcc", "instantaneous", "0-1", "Copertura nuvolosa bassa",
        non_negative=True,
    ),
    VariableSpec(
        "medium_cloud_cover", "mcc", "instantaneous", "0-1", "Copertura nuvolosa media",
        non_negative=True,
    ),
    VariableSpec(
        "high_cloud_cover", "hcc", "instantaneous", "0-1", "Copertura nuvolosa alta",
        non_negative=True,
    ),
    VariableSpec(
        "skin_temperature", "skt", "instantaneous", "K", "Temperatura della superficie",
        store_offset=-KELVIN_AT_ZERO_CELSIUS,
    ),
    VariableSpec(
        "soil_temperature_level_1", "stl1", "instantaneous", "K",
        "Temperatura suolo livello 1", store_offset=-KELVIN_AT_ZERO_CELSIUS,
    ),
    VariableSpec(
        "total_column_water_vapour", "tcwv", "instantaneous", "kg m-2",
        "Vapore d'acqua colonna totale", non_negative=True,
    ),
    VariableSpec(
        "convective_available_potential_energy", "cape", "instantaneous", "J kg-1",
        "Energia potenziale convettiva", non_negative=True,
    ),
    VariableSpec("snow_depth", "sd", "instantaneous", "m", "Spessore neve", non_negative=True),
    # --- cumulati sull'ora precedente ---
    VariableSpec(
        "total_precipitation", "tp", "accumulated", "m", "Precipitazione totale",
        transform="log1p", transform_scale=1000.0, non_negative=True,
    ),
    VariableSpec(
        "snowfall", "sf", "accumulated", "m of water equivalent", "Nevicata",
        transform="log1p", transform_scale=1000.0, non_negative=True,
    ),
    VariableSpec(
        "large_scale_precipitation", "lsp", "accumulated", "m",
        "Precipitazione di larga scala", transform="log1p", transform_scale=1000.0,
        non_negative=True,
    ),
    VariableSpec(
        "convective_precipitation", "cp", "accumulated", "m",
        "Precipitazione convettiva", transform="log1p", transform_scale=1000.0, non_negative=True,
    ),
    VariableSpec(
        "surface_solar_radiation_downwards", "ssrd", "accumulated", "J m-2",
        "Radiazione solare incidente", non_negative=True,
    ),
    # --- statici ---
    VariableSpec(
        "land_sea_mask", "lsm", "static", "0-1", "Frazione di terra", non_negative=True
    ),
    VariableSpec(
        "geopotential", "z", "static", "m2 s-2", "Geopotenziale della superficie (orografia)"
    ),
    # Descrittori di superficie invarianti. ERA5 li espone come campi a un solo istante:
    # descrivono come il suolo scambia energia con l'aria, cioe' esattamente il processo
    # che decide la temperatura a 2 m e che la sola quota non spiega. ERA5 non ha una
    # frazione urbana: il guadagno sulle citta' riportato da Bakketun et al. viene da un
    # campo SURFEX che qui non e' disponibile, e non va quindi atteso.
    VariableSpec("soil_type", "slt", "static", "codice", "Tipo di suolo (classi 0-7)"),
    VariableSpec(
        "high_vegetation_cover", "cvh", "static", "0-1", "Frazione di vegetazione alta",
        non_negative=True,
    ),
    VariableSpec(
        "low_vegetation_cover", "cvl", "static", "0-1", "Frazione di vegetazione bassa",
        non_negative=True,
    ),
    VariableSpec(
        "type_of_high_vegetation", "tvh", "static", "codice", "Tipo di vegetazione alta"
    ),
    VariableSpec(
        "type_of_low_vegetation", "tvl", "static", "codice", "Tipo di vegetazione bassa"
    ),
    VariableSpec(
        "lake_cover", "cl", "static", "0-1", "Frazione di acque interne", non_negative=True
    ),
    VariableSpec("lake_depth", "dl", "static", "m", "Profondita' delle acque interne"),
    VariableSpec(
        "standard_deviation_of_orography", "sdor", "static", "m",
        "Dispersione dell'orografia dentro la cella",
    ),
    VariableSpec(
        "anisotropy_of_sub_gridscale_orography", "isor", "static", "0-1",
        "Anisotropia dell'orografia sottogriglia",
    ),
    VariableSpec(
        "angle_of_sub_gridscale_orography", "anor", "static", "rad",
        "Orientamento dell'orografia sottogriglia",
    ),
    VariableSpec(
        "slope_of_sub_gridscale_orography", "slor", "static", "m m-1",
        "Pendenza dell'orografia sottogriglia",
    ),
    VariableSpec(
        "standard_deviation_of_filtered_subgrid_orography", "sdfor", "static", "m",
        "Dispersione dell'orografia filtrata",
    ),
)

BY_CDS_NAME: dict[str, VariableSpec] = {spec.cds_name: spec for spec in _SPECS}
BY_SHORT_NAME: dict[str, VariableSpec] = {spec.short_name: spec for spec in _SPECS}

if len(BY_CDS_NAME) != len(_SPECS) or len(BY_SHORT_NAME) != len(_SPECS):  # pragma: no cover
    raise RuntimeError("Registro variabili incoerente: nomi CDS o short name duplicati")


# --------------------------------------------------------------------------- #
# Variabili su livelli di pressione
# --------------------------------------------------------------------------- #

# Livelli in hPa pubblicati dalla collection `reanalysis-era5-pressure-levels`.
# Un livello fuori da questo insieme fa rifiutare l'intera richiesta dal CDS.
ERA5_PRESSURE_LEVELS: tuple[int, ...] = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 775, 800, 825,
    850, 875, 900, 925, 950, 975, 1000,
)


@dataclass(frozen=True, slots=True)
class PressureVariable:
    """Metadati comuni a tutti i livelli di una variabile su livelli di pressione.

    Il livello non fa parte di questi metadati: nei GRIB la short name e' la stessa a
    ogni livello (``t`` vale sia a 500 sia a 850 hPa) e a distinguerli e' la coordinata
    ``isobaricInhPa``.
    """

    cds_name: str
    short_name: str
    units: str
    description: str
    store_offset: float = 0.0
    non_negative: bool = False


_PRESSURE_VARIABLES: tuple[PressureVariable, ...] = (
    PressureVariable("geopotential", "z", "m2 s-2", "Geopotenziale"),
    PressureVariable(
        "temperature", "t", "K", "Temperatura", store_offset=-KELVIN_AT_ZERO_CELSIUS
    ),
    PressureVariable(
        "specific_humidity", "q", "kg kg-1", "Umidita' specifica", non_negative=True
    ),
)

PRESSURE_BY_CDS_NAME: dict[str, PressureVariable] = {
    variable.cds_name: variable for variable in _PRESSURE_VARIABLES
}


def pressure_short_name(short_name: str, level: int) -> str:
    """Nome interno univoco di una variabile a un livello: ``t`` a 850 hPa da' ``t850``.

    Serve perche' due livelli della stessa variabile condividono la short name GRIB e
    finirebbero nella stessa colonna dello store.
    """
    return f"{short_name}{level}"


def _build_pressure_spec(variable: PressureVariable, level: int) -> VariableSpec:
    return VariableSpec(
        cds_name=variable.cds_name,
        short_name=pressure_short_name(variable.short_name, level),
        kind="instantaneous",
        units=variable.units,
        description=f"{variable.description} a {level} hPa",
        non_negative=variable.non_negative,
        store_offset=variable.store_offset,
    )


PRESSURE_SPECS: dict[tuple[str, int], VariableSpec] = {
    (variable.cds_name, level): _build_pressure_spec(variable, level)
    for variable in _PRESSURE_VARIABLES
    for level in ERA5_PRESSURE_LEVELS
}

PRESSURE_BY_SHORT_NAME: dict[str, VariableSpec] = {
    spec.short_name: spec for spec in PRESSURE_SPECS.values()
}

_COLLISIONI = sorted(set(PRESSURE_BY_SHORT_NAME) & set(BY_SHORT_NAME))
if _COLLISIONI:  # pragma: no cover
    raise RuntimeError(
        f"Nomi su livelli di pressione in conflitto con le variabili di superficie: "
        f"{_COLLISIONI}"
    )


def spec_by_cds_name(cds_name: str) -> VariableSpec:
    """Restituisce lo spec di una variabile a partire dal nome CDS."""
    try:
        return BY_CDS_NAME[cds_name]
    except KeyError:
        raise KeyError(
            f"Variabile CDS non registrata: {cds_name!r}. "
            f"Aggiungerla in dwf/variables.py con la relativa short name GRIB."
        ) from None


def spec_by_short_name(short_name: str) -> VariableSpec:
    """Restituisce lo spec di una variabile dal nome interno usato nello store.

    Accetta sia le short name GRIB dei campi di superficie sia i nomi con livello dei
    campi su livelli di pressione (``t850``), che nello store sono variabili distinte.
    """
    spec = BY_SHORT_NAME.get(short_name) or PRESSURE_BY_SHORT_NAME.get(short_name)
    if spec is None:
        raise KeyError(f"Short name GRIB non registrata: {short_name!r}")
    return spec


def pressure_variable(cds_name: str) -> PressureVariable:
    """Metadati della variabile su livelli di pressione, indipendenti dal livello."""
    try:
        return PRESSURE_BY_CDS_NAME[cds_name]
    except KeyError:
        raise KeyError(
            f"Variabile su livelli di pressione non registrata: {cds_name!r}. "
            f"Disponibili: {sorted(PRESSURE_BY_CDS_NAME)}"
        ) from None


def pressure_spec(cds_name: str, level: int) -> VariableSpec:
    """Spec di una variabile a un livello di pressione, con nome interno univoco."""
    # Il nome viene validato per primo: segnalare il livello di una variabile che non
    # esiste manderebbe a cercare il problema nella parte sbagliata della richiesta.
    pressure_variable(cds_name)
    try:
        return PRESSURE_SPECS[(cds_name, level)]
    except KeyError:
        raise KeyError(
            f"Livello di pressione non pubblicato da ERA5: {level!r}. "
            f"Ammessi: {list(ERA5_PRESSURE_LEVELS)}"
        ) from None
