"""Termodinamica dell'aria umida: il calore latente nascosto nei dati gia' scaricati.

Fra le variabili ERA5 gia' presenti c'e' il **punto di rugiada**, che da solo dice poco
ma insieme alla temperatura e alla pressione determina tutto il contenuto d'acqua
dell'aria. Da li' si ricava il calore latente, cioe' l'energia immagazzinata nel vapore
e liberata quando condensa: e' il motore della precipitazione e la ragione per cui una
massa d'aria umida si raffredda molto piu' lentamente di una secca.

La rete oggi riceve temperatura e rugiada come due canali indipendenti e dovrebbe
scoprire da sola che conta la loro **differenza** e che il legame con il vapore e'
**esponenziale** (Clausius-Clapeyron: la capacita' dell'aria di contenere acqua
raddoppia ogni circa 10 gradi). E' un'inferenza costosa da imparare e gratuita da
fornire.

La grandezza piu' densa e' la temperatura potenziale equivalente, che combina calore
sensibile e latente in un solo numero e si conserva anche quando l'acqua condensa:
per questo distingue le masse d'aria meglio della temperatura.

Riferimenti: Bolton (1980), *The computation of equivalent potential temperature*;
formulazione della tensione di vapore saturo dell'IFS ECMWF, la stessa che genera ERA5.
"""

from __future__ import annotations

import numpy as np

# Rapporto fra le costanti dei gas di aria secca e vapore acqueo.
EPSILON = 0.621981

# Tensione di vapore saturo a 0 gradi, in Pa, e coefficienti di Tetens sull'acqua
# liquida nella forma usata dall'IFS ECMWF.
ES_AT_ZERO = 611.21
TETENS_A = 17.502
TETENS_B = 240.97

# Calore latente di vaporizzazione a 0 gradi e sua dipendenza dalla temperatura.
LATENT_HEAT_AT_ZERO = 2.501e6
LATENT_HEAT_SLOPE = -2370.0

# Calore specifico dell'aria secca a pressione costante, in J/(kg K).
CP_DRY_AIR = 1004.6

STANDARD_GRAVITY = 9.80665
DRY_AIR_GAS_CONSTANT = 287.058
REFERENCE_PRESSURE = 100000.0

KELVIN_AT_ZERO_CELSIUS = 273.15


def saturation_vapour_pressure(temperature_celsius: np.ndarray) -> np.ndarray:
    """Tensione di vapore saturo in Pa: quanta acqua l'aria puo' contenere.

    Cresce esponenzialmente con la temperatura, e questa non linearita' e' la ragione
    per cui fornire il campo esplicitamente aiuta piu' che passare la sola temperatura.
    """
    temperatura = np.asarray(temperature_celsius, dtype=np.float32)
    return (
        ES_AT_ZERO * np.exp(TETENS_A * temperatura / (temperatura + TETENS_B))
    ).astype(np.float32)


def relative_humidity(
    temperature_celsius: np.ndarray, dewpoint_celsius: np.ndarray
) -> np.ndarray:
    """Umidita' relativa come frazione in [0, 1].

    E' il rapporto fra il vapore presente e quello massimo possibile alla temperatura
    corrente. Il taglio a 1 assorbe la lieve sovrasaturazione che compare quando la
    rugiada supera di poco la temperatura per arrotondamento nei dati.
    """
    vapore = saturation_vapour_pressure(dewpoint_celsius)
    saturazione = saturation_vapour_pressure(temperature_celsius)
    return np.clip(vapore / saturazione, 0.0, 1.0).astype(np.float32)


def dewpoint_depression(
    temperature_celsius: np.ndarray, dewpoint_celsius: np.ndarray
) -> np.ndarray:
    """Scarto fra temperatura e punto di rugiada, in gradi.

    Vale zero quando l'aria e' satura, quindi indica quanto manca alla condensazione.
    E' il predittore piu' diretto di nebbia, nubi basse e persistenza della pioggia.
    """
    return (
        np.asarray(temperature_celsius, dtype=np.float32)
        - np.asarray(dewpoint_celsius, dtype=np.float32)
    ).astype(np.float32)


def surface_pressure_from_msl(
    mean_sea_level_pressure: np.ndarray,
    geopotential: np.ndarray,
    temperature_celsius: np.ndarray,
) -> np.ndarray:
    """Pressione alla superficie stimata dalla pressione al livello del mare.

    Fra le variabili scaricate c'e' la pressione ridotta al livello del mare, non
    quella reale al suolo: sulle Alpi differiscono di oltre il 30 %, e usare la prima
    sbaglierebbe l'umidita' specifica proprio dove il rilievo conta. Il geopotenziale
    statico da' la quota, e la formula barometrica ricostruisce la pressione locale.
    """
    quota = np.asarray(geopotential, dtype=np.float32) / STANDARD_GRAVITY
    temperatura_kelvin = (
        np.asarray(temperature_celsius, dtype=np.float32) + KELVIN_AT_ZERO_CELSIUS
    )
    altezza_scala = DRY_AIR_GAS_CONSTANT * temperatura_kelvin / STANDARD_GRAVITY
    return (
        np.asarray(mean_sea_level_pressure, dtype=np.float32)
        * np.exp(-np.clip(quota, 0.0, None) / altezza_scala)
    ).astype(np.float32)


def specific_humidity(
    dewpoint_celsius: np.ndarray, pressure_pa: np.ndarray
) -> np.ndarray:
    """Massa di vapore per massa d'aria, in kg/kg.

    A differenza dell'umidita' relativa non dipende dalla temperatura, quindi identifica
    una massa d'aria anche mentre si scalda o si raffredda.
    """
    vapore = saturation_vapour_pressure(dewpoint_celsius)
    pressione = np.asarray(pressure_pa, dtype=np.float32)
    return (EPSILON * vapore / (pressione - (1.0 - EPSILON) * vapore)).astype(np.float32)


def mixing_ratio(dewpoint_celsius: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    """Massa di vapore per massa di aria secca, in kg/kg."""
    vapore = saturation_vapour_pressure(dewpoint_celsius)
    pressione = np.asarray(pressure_pa, dtype=np.float32)
    return (EPSILON * vapore / np.clip(pressione - vapore, 1.0, None)).astype(np.float32)


def latent_heat_of_vaporization(temperature_celsius: np.ndarray) -> np.ndarray:
    """Calore latente di vaporizzazione in J/kg, con la sua debole calo con la temperatura."""
    return (
        LATENT_HEAT_AT_ZERO
        + LATENT_HEAT_SLOPE * np.asarray(temperature_celsius, dtype=np.float32)
    ).astype(np.float32)


def latent_heat_content(
    temperature_celsius: np.ndarray,
    dewpoint_celsius: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """Energia latente per massa d'aria, in J/kg.

    E' l'energia che verrebbe liberata se tutto il vapore condensasse. Espressa cosi'
    e' confrontabile con il calore sensibile e rende esplicito perche' l'aria umida
    resiste al raffreddamento notturno.
    """
    umidita = specific_humidity(dewpoint_celsius, pressure_pa)
    return (latent_heat_of_vaporization(temperature_celsius) * umidita).astype(np.float32)


def equivalent_potential_temperature(
    temperature_celsius: np.ndarray,
    dewpoint_celsius: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """Temperatura potenziale equivalente in kelvin, secondo Bolton (1980).

    Riassume in un solo campo calore sensibile, calore latente ed effetto della
    pressione, e si conserva nei processi adiabatici umidi. Per questo due punti con la
    stessa theta-e appartengono alla stessa massa d'aria anche se hanno temperature
    diverse, mentre la sola temperatura li separerebbe.
    """
    temperatura = (
        np.asarray(temperature_celsius, dtype=np.float32) + KELVIN_AT_ZERO_CELSIUS
    )
    rugiada = np.asarray(dewpoint_celsius, dtype=np.float32) + KELVIN_AT_ZERO_CELSIUS
    pressione = np.asarray(pressure_pa, dtype=np.float32)
    rapporto = mixing_ratio(dewpoint_celsius, pressure_pa)

    # Temperatura al livello di condensazione, equazione 15 di Bolton. Il clip sul
    # denominatore evita la singolarita' quando la rugiada scende sotto 56 K, che i
    # dati reali non raggiungono ma un test sintetico potrebbe.
    denominatore = 1.0 / np.clip(rugiada - 56.0, 1.0, None) + np.log(
        temperatura / rugiada
    ) / 800.0
    temperatura_lcl = 1.0 / denominatore + 56.0

    esponente = 0.2854 * (1.0 - 0.28 * rapporto)
    potenziale = temperatura * (REFERENCE_PRESSURE / pressione) ** esponente
    return (
        potenziale
        * np.exp((3376.0 / temperatura_lcl - 2.54) * rapporto * (1.0 + 0.81 * rapporto))
    ).astype(np.float32)


# Nomi dei canali termodinamici, nell'ordine in cui vengono impilati.
THERMO_CHANNEL_NAMES = ("rel_humidity", "dewpoint_depression", "latent_heat", "theta_e")

# Scale usate per portare i canali termodinamici in un intervallo confrontabile con le
# variabili gia' normalizzate. Sono divisori fissi, non statistiche stimate: i valori
# tipici sono noti dalla fisica e non serve leggerli dai dati.
THERMO_SCALES = {
    "rel_humidity": 1.0,
    "dewpoint_depression": 10.0,
    "latent_heat": 1.0e4,
    "theta_e": 300.0,
}


def thermo_fields(
    temperature_celsius: np.ndarray,
    dewpoint_celsius: np.ndarray,
    pressure_pa: np.ndarray,
) -> dict[str, np.ndarray]:
    """I quattro campi termodinamici usati come canali, gia' in scala confrontabile."""
    latente = latent_heat_content(temperature_celsius, dewpoint_celsius, pressure_pa)
    theta = equivalent_potential_temperature(
        temperature_celsius, dewpoint_celsius, pressure_pa
    )
    return {
        "rel_humidity": relative_humidity(temperature_celsius, dewpoint_celsius),
        "dewpoint_depression": (
            dewpoint_depression(temperature_celsius, dewpoint_celsius)
            / THERMO_SCALES["dewpoint_depression"]
        ).astype(np.float32),
        "latent_heat": (latente / THERMO_SCALES["latent_heat"]).astype(np.float32),
        "theta_e": (theta / THERMO_SCALES["theta_e"]).astype(np.float32),
    }


__all__ = [
    "EPSILON",
    "THERMO_CHANNEL_NAMES",
    "THERMO_SCALES",
    "dewpoint_depression",
    "equivalent_potential_temperature",
    "latent_heat_content",
    "latent_heat_of_vaporization",
    "mixing_ratio",
    "relative_humidity",
    "saturation_vapour_pressure",
    "specific_humidity",
    "surface_pressure_from_msl",
    "thermo_fields",
]
