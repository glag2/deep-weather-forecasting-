"""Geometria solare: il forzante che guida il ciclo diurno e stagionale.

La rete riceve gia' una codifica ciclica del tempo (seno e coseno dell'ora e del giorno
dell'anno), ma quella dice soltanto *quando* siamo. Non dice quanta energia arriva in
un punto: a parita' di ora, il sole a mezzogiorno e' alto in Sicilia e radente in
Lapponia, e a parita' di latitudine cambia con la stagione. Quel campo la rete
dovrebbe dedurlo combinando latitudine, ora e giorno attraverso non linearita', mentre
qui e' calcolabile **esattamente** e a costo nullo, perche' non richiede alcun dato.

E' il candidato piu' promettente per il difetto misurato del modello, che coglie la
fase del ciclo diurno ma ne sottostima l'ampiezza.

Include anche la distanza Terra-Sole, che varia del 3,3 % fra perielio e afelio e
modula l'irradianza del 6,9 %: piccola ma sistematica, e stagionalmente in
controfase con l'emisfero nord.

Formule di Spencer (1971), *Fourier series representation of the position of the sun*,
usate anche da ECMWF e NREL. Precisione sufficiente per una feature: la declinazione e'
accurata a circa 0,01 gradi.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

# Costante solare: irradianza media alla distanza media Terra-Sole, in W/m^2.
SOLAR_CONSTANT = 1361.0

# Minuti per radiante dell'angolo orario, usato dall'equazione del tempo.
MINUTES_PER_RADIAN = 229.18


def day_angle(day_of_year: np.ndarray | float) -> np.ndarray:
    """Posizione angolare della Terra sull'orbita, in radianti."""
    giorno = np.asarray(day_of_year, dtype=np.float64)
    return 2.0 * np.pi * (giorno - 1.0) / 365.0


def solar_declination(day_of_year: np.ndarray | float) -> np.ndarray:
    """Declinazione solare in radianti: l'inclinazione che crea le stagioni.

    Oscilla fra -23,44 e +23,44 gradi. E' il termine che, combinato con la latitudine,
    determina quanto e' alto il sole a mezzogiorno.
    """
    gamma = day_angle(day_of_year)
    return (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )


def earth_sun_distance_factor(day_of_year: np.ndarray | float) -> np.ndarray:
    """Fattore di correzione dell'irradianza per la distanza Terra-Sole.

    Vale il quadrato del rapporto fra distanza media e distanza istantanea, quindi
    circa 1,034 al perielio (inizio gennaio) e 0,967 all'afelio (inizio luglio). E' la
    "distanza dal sole" espressa nella forma in cui conta per l'energia ricevuta.
    """
    gamma = day_angle(day_of_year)
    return (
        1.000110
        + 0.034221 * np.cos(gamma)
        + 0.001280 * np.sin(gamma)
        + 0.000719 * np.cos(2 * gamma)
        + 0.000077 * np.sin(2 * gamma)
    )


def equation_of_time(day_of_year: np.ndarray | float) -> np.ndarray:
    """Scarto fra tempo solare vero e tempo solare medio, in minuti.

    Arriva a circa +16 e -14 minuti nell'arco dell'anno. Senza questa correzione il
    mezzogiorno solare calcolato sbaglierebbe fino a un quarto d'ora, che su uno slot
    di sei ore e' poco ma sulla posizione del sole e' visibile.
    """
    gamma = day_angle(day_of_year)
    return MINUTES_PER_RADIAN * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )


def hour_angle(
    longitudes: np.ndarray, utc_hour: float, day_of_year: float
) -> np.ndarray:
    """Angolo orario solare in radianti: 0 al mezzogiorno locale, negativo di mattina."""
    lon = np.asarray(longitudes, dtype=np.float64)
    ore_solari = utc_hour + lon / 15.0 + equation_of_time(day_of_year) / 60.0
    return np.deg2rad(15.0 * (ore_solari - 12.0))


def cos_solar_zenith(
    latitudes: np.ndarray, longitudes: np.ndarray, when: datetime
) -> np.ndarray:
    """Coseno dell'angolo zenitale solare sulla griglia, tagliato a zero di notte.

    E' proporzionale all'energia che arriva su una superficie orizzontale: 1 con il
    sole allo zenit, 0 all'orizzonte e sotto. Il taglio a zero e' fisico, non
    cosmetico: di notte non arriva radiazione, e un valore negativo farebbe credere
    alla rete che ne venga sottratta.
    """
    giorno = float(when.timetuple().tm_yday)
    ora = when.hour + when.minute / 60.0 + when.second / 3600.0

    declinazione = float(solar_declination(giorno))
    lat = np.deg2rad(np.asarray(latitudes, dtype=np.float64)).reshape(-1, 1)
    angolo = hour_angle(longitudes, ora, giorno).reshape(1, -1)

    coseno = np.sin(lat) * np.sin(declinazione) + np.cos(lat) * np.cos(declinazione) * np.cos(
        angolo
    )
    return np.clip(coseno, 0.0, 1.0).astype(np.float32)


def toa_insolation(
    latitudes: np.ndarray, longitudes: np.ndarray, when: datetime
) -> np.ndarray:
    """Irradianza al top dell'atmosfera, in W/m^2.

    E' il prodotto fra costante solare, correzione di distanza e coseno zenitale:
    l'energia disponibile prima che l'atmosfera intervenga. Cio' che la rete deve
    ancora imparare e' quanta ne arriva a terra, che dipende da nubi e umidita', e per
    quelle ha gia' i canali.
    """
    giorno = float(when.timetuple().tm_yday)
    fattore = float(earth_sun_distance_factor(giorno))
    return (SOLAR_CONSTANT * fattore * cos_solar_zenith(latitudes, longitudes, when)).astype(
        np.float32
    )


def day_length_hours(latitudes: np.ndarray, day_of_year: float) -> np.ndarray:
    """Ore di luce alla latitudine data, fra 0 (notte polare) e 24 (sole di mezzanotte).

    Non e' ricavabile dal coseno zenitale di un singolo istante, ed e' la variabile che
    governa quanta energia si accumula nell'arco della giornata: due luoghi con lo
    stesso sole a mezzogiorno ma durata del giorno diversa si scaldano diversamente.
    """
    declinazione = float(solar_declination(day_of_year))
    lat = np.deg2rad(np.asarray(latitudes, dtype=np.float64))
    # Oltre i circoli polari l'argomento esce da [-1, 1]: la saturazione produce
    # correttamente 24 ore di luce o 24 di buio.
    coseno = np.clip(-np.tan(lat) * np.tan(declinazione), -1.0, 1.0)
    return (2.0 * np.rad2deg(np.arccos(coseno)) / 15.0).astype(np.float32)


def solar_fields(
    latitudes: np.ndarray, longitudes: np.ndarray, when: datetime
) -> dict[str, np.ndarray]:
    """I tre campi solari usati come canali, gia' in scala ragionevole.

    Sono normalizzati qui e non dalle statistiche di train perche' i loro estremi sono
    noti esattamente: dividere per la costante solare e per 24 ore da' valori in [0, 1]
    senza dover stimare nulla dai dati.
    """
    n_lat = len(latitudes)
    n_lon = len(longitudes)
    giorno = float(when.timetuple().tm_yday)

    durata = day_length_hours(latitudes, giorno).reshape(-1, 1)
    return {
        "cos_zenith": cos_solar_zenith(latitudes, longitudes, when),
        "toa_insolation": (
            toa_insolation(latitudes, longitudes, when) / SOLAR_CONSTANT
        ).astype(np.float32),
        "day_length": np.broadcast_to(durata / 24.0, (n_lat, n_lon)).astype(np.float32),
    }


# Nomi dei canali solari, nell'ordine in cui vengono impilati.
SOLAR_CHANNEL_NAMES = ("cos_zenith", "toa_insolation", "day_length")


__all__ = [
    "SOLAR_CHANNEL_NAMES",
    "SOLAR_CONSTANT",
    "cos_solar_zenith",
    "day_angle",
    "day_length_hours",
    "earth_sun_distance_factor",
    "equation_of_time",
    "hour_angle",
    "solar_declination",
    "solar_fields",
    "toa_insolation",
]
