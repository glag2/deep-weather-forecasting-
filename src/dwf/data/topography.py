"""Descrittori topografici derivati dal campo di quota che il dataset contiene gia'.

Motivazione misurata: la rete guadagna solo il 5,5% sulla persistenza diurna a 24 ore
sulla temperatura, e gli errori dei modelli data-driven a questa scala si concentrano su
montagna e coste. Bakketun et al. (arXiv:2607.02824) mostrano che fornire descrittori di
superficie e indici topografici di vicinato riduce l'errore sulla temperatura a 2 metri,
con la motivazione esplicita che il decoder non collega punti di griglia vicini: il
contesto locale conviene darlo in ingresso invece di sperare che venga ricostruito.

Qui si calcola solo cio' che si ricava dai dati presenti, senza alcun download: pendenze,
rugosita', posizione relativa e rilievo del vicinato, tutto dal geopotenziale di superficie.
I descrittori sono statici, quindi si calcolano una volta per dominio.

Convenzione sulle unita': ERA5 fornisce `z` come geopotenziale in m^2/s^2; dividendo per
l'accelerazione di gravita' si ottengono metri, che rendono le pendenze leggibili come
metri per punto di griglia invece che come numeri senza significato fisico.
"""

from __future__ import annotations

import numpy as np

GRAVITA = 9.80665
"""Accelerazione di gravita' usata da ERA5 per convertire geopotenziale in metri."""

NOMI_PENDENZA = ("topo_slope_ns", "topo_slope_we")
"""Descrittori indipendenti dal raggio: le due derivate della quota."""


class TopografiaError(ValueError):
    """Errore nella richiesta o nel calcolo dei descrittori topografici."""


def nomi_descrittori(raggi: tuple[int, ...]) -> tuple[str, ...]:
    """Nomi dei canali prodotti, nell'ordine in cui vengono generati.

    L'ordine e' parte del contratto: il layout dei canali di input lo usa per sapere quale
    indice corrisponde a quale descrittore, e uno scambio silenzioso non farebbe fallire
    nulla, degraderebbe soltanto il modello.
    """
    valida_raggi(raggi)
    nomi = list(NOMI_PENDENZA)
    for raggio in raggi:
        nomi.extend(
            (f"topo_std_r{raggio}", f"topo_tpi_r{raggio}", f"topo_relief_r{raggio}")
        )
    return tuple(nomi)


def valida_raggi(raggi: tuple[int, ...]) -> None:
    """I raggi devono essere positivi, unici e crescenti."""
    if any(raggio < 1 for raggio in raggi):
        raise TopografiaError(f"I raggi devono essere >= 1, ricevuto {raggi}")
    if len(set(raggi)) != len(raggi):
        raise TopografiaError(f"I raggi contengono duplicati: {raggi}")
    if list(raggi) != sorted(raggi):
        raise TopografiaError(f"I raggi devono essere crescenti: {raggi}")


def quota_in_metri(geopotenziale: np.ndarray) -> np.ndarray:
    """Converte il geopotenziale di superficie in metri."""
    campo = np.asarray(geopotenziale, dtype=np.float32)
    if campo.ndim != 2:
        raise TopografiaError(
            f"La quota deve essere un campo (lat, lon), ricevuto {campo.shape}"
        )
    return campo / np.float32(GRAVITA)


def _finestre(campo: np.ndarray, raggio: int) -> np.ndarray:
    """Vista scorrevole (H, W, 2r+1, 2r+1) con bordo riflesso.

    Il bordo riflesso e' l'unica scelta difendibile su un dominio limitato: estendere con
    zeri inventerebbe un mare piatto oltre il confine, e ripetere il bordo inventerebbe una
    pianura. Riflettere assume che oltre il taglio il terreno somigli a quello dentro.
    """
    altezza, larghezza = campo.shape
    if raggio >= min(altezza, larghezza):
        raise TopografiaError(
            f"Il raggio {raggio} non e' minore della griglia {campo.shape}"
        )
    esteso = np.pad(campo, raggio, mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(esteso, (2 * raggio + 1,) * 2)


def descrittori(
    geopotenziale: np.ndarray, raggi: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """Calcola i descrittori topografici, standardizzati sul dominio.

    Ogni descrittore viene standardizzato con la propria media e deviazione standard
    calcolate sul dominio stesso: sono campi statici, quindi non hanno una statistica
    temporale da stimare e non passano dalle statistiche di normalizzazione delle
    variabili dinamiche. Un descrittore costante resta a zero invece di dividere per zero.
    """
    valida_raggi(raggi)
    quota = quota_in_metri(geopotenziale)

    # Differenze centrate: `np.gradient` le calcola all'interno e passa a differenze
    # laterali sul bordo, che e' esattamente il comportamento desiderato.
    pendenza_ns, pendenza_we = np.gradient(quota)
    grezzi: dict[str, np.ndarray] = {
        NOMI_PENDENZA[0]: pendenza_ns,
        NOMI_PENDENZA[1]: pendenza_we,
    }

    for raggio in raggi:
        finestre = _finestre(quota, raggio)
        assi = (-2, -1)
        grezzi[f"topo_std_r{raggio}"] = finestre.std(axis=assi)
        grezzi[f"topo_tpi_r{raggio}"] = quota - finestre.mean(axis=assi)
        grezzi[f"topo_relief_r{raggio}"] = finestre.max(axis=assi) - finestre.min(
            axis=assi
        )

    return {nome: _standardizza(valore) for nome, valore in grezzi.items()}


def _standardizza(campo: np.ndarray) -> np.ndarray:
    valori = np.asarray(campo, dtype=np.float32)
    scarto = float(valori.std())
    if scarto == 0.0:
        return np.zeros_like(valori)
    return ((valori - float(valori.mean())) / scarto).astype(np.float32)


__all__ = [
    "GRAVITA",
    "NOMI_PENDENZA",
    "TopografiaError",
    "descrittori",
    "nomi_descrittori",
    "quota_in_metri",
    "valida_raggi",
]
