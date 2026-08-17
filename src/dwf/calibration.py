"""Calibrazione delle probabilita' previste.

Una rete addestrata con la log-verosimiglianza produce probabilita' ordinate bene ma
non necessariamente **giuste**: puo' dire 0,15 dove la frequenza osservata e' 0,01 e
0,55 dove e' 0,64. L'ordinamento e' informativo, la scala no. Poiche' l'affidabilita'
e' fra le grandezze che l'utente ha chiesto, la scala va corretta.

La correzione e' una **regressione isotonica**: la mappa monotona non decrescente che
minimizza l'errore quadratico contro l'osservato. Monotona perche' non deve mai
invertire l'ordinamento appreso dalla rete, e non parametrica perche' la forma della
distorsione non e' nota a priori (una sigmoide di Platt imporrebbe una forma che i dati
non hanno).

L'algoritmo e' *pool adjacent violators*: scorre i valori ordinati e fonde i blocchi
che violano la monotonia, sostituendoli con la loro media pesata. Sono poche righe e
non richiedono una dipendenza in piu'.

**Dove si stima conta piu' di come.** Calibrare sugli stessi dati su cui si misura il
guadagno significa misurare quanto bene si e' adattato il rumore. Qui la mappa si stima
sulla validazione e si applica al test, che la rete non ha mai visto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from dwf.tables import CALIBRATION, cast_to_schema

# Sotto questa numerosita' la mappa stimata e' rumore travestito da correzione.
MIN_SAMPLES = 100


class CalibrationError(RuntimeError):
    """Errore nella stima o nell'applicazione della calibrazione."""


def pool_adjacent_violators(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Regressione isotonica su valori gia' ordinati per probabilita' crescente.

    Restituisce la sequenza non decrescente piu' vicina a `values` in errore quadratico
    pesato. Implementazione a blocchi: ogni blocco conserva somma e peso, cosi' la
    fusione e' una media pesata e il costo resta lineare.
    """
    somme: list[float] = []
    pesi: list[float] = []
    lunghezze: list[int] = []

    for valore, peso in zip(values, weights, strict=True):
        somma_corrente = float(valore) * float(peso)
        peso_corrente = float(peso)
        lunghezza_corrente = 1
        # Finche' il blocco corrente viola la monotonia rispetto al precedente, si
        # fondono. Le somme si accumulano in variabili locali: aggiornare la lista in
        # posizione mentre la si accorcia sposterebbe gli indici sotto i piedi.
        while somme and somme[-1] / pesi[-1] > somma_corrente / peso_corrente:
            somma_corrente += somme.pop()
            peso_corrente += pesi.pop()
            lunghezza_corrente += lunghezze.pop()
        somme.append(somma_corrente)
        pesi.append(peso_corrente)
        lunghezze.append(lunghezza_corrente)

    risultato = np.empty(int(sum(lunghezze)), dtype=np.float64)
    posizione = 0
    for somma, peso, lunghezza in zip(somme, pesi, lunghezze, strict=True):
        risultato[posizione : posizione + lunghezza] = somma / peso
        posizione += lunghezza
    return risultato


@dataclass(frozen=True, slots=True)
class ProbabilityCalibrator:
    """Mappa monotona da probabilita' dichiarata a probabilita' corretta."""

    knots_in: np.ndarray
    knots_out: np.ndarray
    variable: str
    fitted_on_split: str
    n_samples: int

    def apply(self, probability: np.ndarray) -> np.ndarray:
        """Applica la mappa, interpolando linearmente fra i nodi.

        Fuori dall'intervallo osservato `np.interp` satura all'estremo: e' il
        comportamento voluto, perche' estrapolare una calibrazione oltre i dati che
        l'hanno prodotta inventerebbe informazione.
        """
        corretta = np.interp(
            np.asarray(probability, dtype=np.float64), self.knots_in, self.knots_out
        )
        return np.clip(corretta, 0.0, 1.0).astype(np.float32)

    def to_table(self, *, fold: int) -> pl.DataFrame:
        return cast_to_schema(
            pl.DataFrame(
                {
                    "fold": np.full(self.knots_in.size, fold, dtype=np.int16),
                    "variable": [self.variable] * self.knots_in.size,
                    "split": [self.fitted_on_split] * self.knots_in.size,
                    "probability_in": self.knots_in,
                    "probability_out": self.knots_out,
                    "n_samples": np.full(self.knots_in.size, self.n_samples, dtype=np.int64),
                }
            ),
            CALIBRATION,
        )

    @classmethod
    def from_table(cls, table: pl.DataFrame, *, variable: str = "tp") -> ProbabilityCalibrator:
        righe = table.filter(pl.col("variable") == variable).sort("probability_in")
        if righe.height == 0:
            raise CalibrationError(f"Nessuna calibrazione registrata per {variable!r}")
        return cls(
            knots_in=righe.get_column("probability_in").to_numpy().astype(np.float64),
            knots_out=righe.get_column("probability_out").to_numpy().astype(np.float64),
            variable=variable,
            fitted_on_split=righe.get_column("split").item(0),
            n_samples=int(righe.get_column("n_samples").item(0)),
        )


def fit_calibrator(
    probability: np.ndarray,
    outcome: np.ndarray,
    *,
    variable: str = "tp",
    fitted_on_split: str = "val",
    max_knots: int = 64,
) -> ProbabilityCalibrator:
    """Stima la mappa di calibrazione da probabilita' previste e esiti osservati."""
    probabilita = np.asarray(probability, dtype=np.float64).reshape(-1)
    esiti = np.asarray(outcome, dtype=np.float64).reshape(-1)
    if probabilita.size != esiti.size:
        raise CalibrationError(
            f"Probabilita' ed esiti di lunghezza diversa: {probabilita.size} e {esiti.size}"
        )
    finiti = np.isfinite(probabilita) & np.isfinite(esiti)
    probabilita, esiti = probabilita[finiti], esiti[finiti]
    if probabilita.size < MIN_SAMPLES:
        raise CalibrationError(
            f"Servono almeno {MIN_SAMPLES} casi per calibrare, ricevuti {probabilita.size}"
        )

    ordine = np.argsort(probabilita, kind="stable")
    probabilita, esiti = probabilita[ordine], esiti[ordine]

    # I punti si raggruppano prima della regressione: con centinaia di migliaia di
    # valori la mappa a piena risoluzione sarebbe enorme e indistinguibile da questa.
    bordi = np.quantile(probabilita, np.linspace(0.0, 1.0, max_knots + 1))
    bordi = np.unique(bordi)
    if bordi.size < 3:
        raise CalibrationError(
            "Le probabilita' previste sono quasi costanti: non c'e' nulla da calibrare"
        )
    indici = np.clip(np.searchsorted(bordi[1:-1], probabilita, side="right"), 0, bordi.size - 2)

    conteggi = np.bincount(indici, minlength=bordi.size - 1).astype(np.float64)
    somma_prob = np.bincount(indici, weights=probabilita, minlength=bordi.size - 1)
    somma_esiti = np.bincount(indici, weights=esiti, minlength=bordi.size - 1)
    pieni = conteggi > 0

    medie_prob = somma_prob[pieni] / conteggi[pieni]
    medie_esiti = somma_esiti[pieni] / conteggi[pieni]
    corrette = pool_adjacent_violators(medie_esiti, conteggi[pieni])

    return ProbabilityCalibrator(
        knots_in=medie_prob,
        knots_out=corrette,
        variable=variable,
        fitted_on_split=fitted_on_split,
        n_samples=int(probabilita.size),
    )


def calibration_error(probability: np.ndarray, outcome: np.ndarray, *, n_bins: int = 10) -> float:
    """Errore di calibrazione atteso: scarto medio fra dichiarato e osservato.

    E' la sintesi numerica del diagramma di affidabilita': 0 significa che quando il
    modello dice 30 % piove nel 30 % dei casi. Pesato sulla numerosita' degli
    intervalli, perche' un intervallo con dieci casi non vale quanto uno con diecimila.
    """
    probabilita = np.asarray(probability, dtype=np.float64).reshape(-1)
    esiti = np.asarray(outcome, dtype=np.float64).reshape(-1)
    if probabilita.size == 0:
        return float("nan")
    bordi = np.linspace(0.0, 1.0, n_bins + 1)
    indici = np.clip(np.digitize(probabilita, bordi[1:-1], right=False), 0, n_bins - 1)
    totale = 0.0
    for intervallo in range(n_bins):
        selezione = indici == intervallo
        conteggio = int(selezione.sum())
        if conteggio == 0:
            continue
        scarto = abs(probabilita[selezione].mean() - esiti[selezione].mean())
        totale += scarto * conteggio
    return float(totale / probabilita.size)


__all__ = [
    "CalibrationError",
    "ProbabilityCalibrator",
    "calibration_error",
    "fit_calibrator",
    "pool_adjacent_violators",
]
