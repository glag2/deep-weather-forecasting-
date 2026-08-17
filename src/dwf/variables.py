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
)

BY_CDS_NAME: dict[str, VariableSpec] = {spec.cds_name: spec for spec in _SPECS}
BY_SHORT_NAME: dict[str, VariableSpec] = {spec.short_name: spec for spec in _SPECS}

if len(BY_CDS_NAME) != len(_SPECS) or len(BY_SHORT_NAME) != len(_SPECS):  # pragma: no cover
    raise RuntimeError("Registro variabili incoerente: nomi CDS o short name duplicati")


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
    """Restituisce lo spec di una variabile a partire dalla short name GRIB."""
    try:
        return BY_SHORT_NAME[short_name]
    except KeyError:
        raise KeyError(f"Short name GRIB non registrata: {short_name!r}") from None
