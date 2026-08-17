"""Pesi spaziali applicati alla perdita.

Due correzioni, entrambe con una ragione precisa.

**Area.** La griglia e' regolare in gradi, non in chilometri: a 70 gradi di latitudine
una cella copre circa un terzo dell'area di una cella equatoriale. Una media non pesata
darebbe quindi al nord un'importanza molto maggiore di quella che gli spetta, e la rete
spenderebbe capacita' sull'Artico a scapito delle medie latitudini. Il peso di area,
proporzionale al coseno della latitudine, e' la stessa convenzione usata dai centri
meteorologici per le verifiche.

**Fuoco locale.** Il progetto ha un luogo di interesse dichiarato, Vigo di Cadore, e
gli errori li' contano un po' di piu'. Il peso e' una campana attorno al punto, con un
guadagno volutamente contenuto: un peso troppo alto trasformerebbe un modello di
dominio in un modello locale addestrato su una manciata di celle, che generalizzerebbe
peggio ovunque, Vigo compreso.

I pesi sono **normalizzati a media unitaria** sul ritaglio, cosi' cambiarli non cambia
la scala della perdita e i pesi relativi fra le teste restano confrontabili fra
configurazioni diverse.
"""

from __future__ import annotations

import numpy as np

# Vigo di Cadore. La cella corrispondente sulla griglia a 0,25 gradi ha quota 1463 m
# contro i 951 m reali del paese: il peso indirizza la rete verso il punto, non
# pretende di risolvere quella differenza, che e' un limite di risoluzione.
VIGO_LATITUDE = 46.5031
VIGO_LONGITUDE = 12.5308

# Sotto questa latitudine assoluta il coseno resta vicino a uno; il limite inferiore
# evita che un eventuale punto polare annulli del tutto il proprio contributo.
MIN_AREA_WEIGHT = 1e-3


class WeightingError(ValueError):
    """Parametri di pesatura incoerenti."""


def latitude_area_weight(latitudes: np.ndarray, width: int) -> np.ndarray:
    """Peso proporzionale all'area della cella, replicato su tutte le longitudini."""
    if latitudes.ndim != 1:
        raise WeightingError(f"Attese latitudini monodimensionali, ricevute {latitudes.shape}")
    if width <= 0:
        raise WeightingError(f"Larghezza non valida: {width}")
    if np.any(np.abs(latitudes) > 90.0):
        raise WeightingError("Latitudini fuori da [-90, 90]")
    coseno = np.clip(np.cos(np.deg2rad(latitudes)), MIN_AREA_WEIGHT, None)
    return np.repeat(coseno[:, None], width, axis=1).astype(np.float32)


def focus_weight(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    center_lat: float = VIGO_LATITUDE,
    center_lon: float = VIGO_LONGITUDE,
    radius_deg: float = 1.5,
    gain: float = 0.5,
) -> np.ndarray:
    """Campana gaussiana attorno al punto di interesse, pari a ``1 + gain`` al centro.

    Vale uno lontano dal centro, quindi non toglie peso al resto del dominio: aggiunge
    attenzione dove serve invece di sottrarla altrove.
    """
    if radius_deg <= 0.0:
        raise WeightingError(f"Il raggio deve essere positivo, ricevuto {radius_deg}")
    if gain < 0.0:
        raise WeightingError(f"Il guadagno non puo' essere negativo, ricevuto {gain}")

    scarto_lat = latitudes[:, None] - center_lat
    # I gradi di longitudine si accorciano con la latitudine: senza questa correzione
    # la campana sarebbe molto piu' larga in est-ovest di quanto si intenda.
    scarto_lon = (longitudes[None, :] - center_lon) * np.cos(np.deg2rad(center_lat))
    distanza_quadra = scarto_lat**2 + scarto_lon**2
    campana = np.exp(-0.5 * distanza_quadra / radius_deg**2)
    return (1.0 + gain * campana).astype(np.float32)


def spatial_weight(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    use_area: bool = True,
    focus_gain: float = 0.5,
    focus_radius_deg: float = 1.5,
    center_lat: float = VIGO_LATITUDE,
    center_lon: float = VIGO_LONGITUDE,
) -> np.ndarray:
    """Peso complessivo del ritaglio, normalizzato a media unitaria."""
    altezza, larghezza = latitudes.size, longitudes.size
    peso = np.ones((altezza, larghezza), dtype=np.float32)
    if use_area:
        peso = peso * latitude_area_weight(latitudes, larghezza)
    if focus_gain > 0.0:
        peso = peso * focus_weight(
            latitudes,
            longitudes,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_deg=focus_radius_deg,
            gain=focus_gain,
        )
    media = float(peso.mean())
    if media <= 0.0:  # pragma: no cover - impossibile con i limiti imposti sopra
        raise WeightingError("Peso spaziale degenere: media nulla")
    return (peso / media).astype(np.float32)


__all__ = [
    "MIN_AREA_WEIGHT",
    "VIGO_LATITUDE",
    "VIGO_LONGITUDE",
    "WeightingError",
    "focus_weight",
    "latitude_area_weight",
    "spatial_weight",
]
