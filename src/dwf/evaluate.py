"""Valutazione delle previsioni contro linee di riferimento.

Un modello meteorologico non si giudica dall'errore assoluto ma dal confronto con
alternative banali. Due sono indispensabili:

- **persistenza**: ripetere l'ultimo stato osservato. Batterla e' il minimo sindacale
  a breve termine, ed e' sorprendentemente difficile alle prime ore.
- **climatologia**: la media storica per mese e ora del giorno. E' la previsione
  ottimale in assenza di informazione, e su orizzonti lunghi e' l'avversario vero.

Per la neve si usa il Brier Skill Score rispetto alla climatologia: la frequenza di
base cambia di ordini di grandezza fra luglio e gennaio, quindi un punteggio assoluto
direbbe soltanto in che stagione siamo. Il rapporto con la climatologia lo corregge,
e le metriche restano stratificate per mese.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch

from dwf.calibration import ProbabilityCalibrator, calibration_error
from dwf.data.dataset import (
    KEY_FEATURES,
    KEY_SLOT,
    WeatherWindowDataset,
    target_specs,
)
from dwf.data.features import NormStats
from dwf.models.heads import OutputLayout
from dwf.tables import METRICS, RELIABILITY, cast_to_schema

if TYPE_CHECKING:  # pragma: no cover - solo per i tipi
    from dwf.config import Config

# Numero di intervalli del diagramma di affidabilita'.
RELIABILITY_BINS = 10

# Sotto questo numero di casi un intervallo di affidabilita' e' rumore.
MIN_BIN_COUNT = 20


class EvaluationError(RuntimeError):
    """Errore nel calcolo delle metriche."""


# Sopra questa quota di neve sul totale, la precipitazione si considera nevosa. Serve
# una soglia perche' "nevica" e' un evento binario mentre il modello prevede una
# frazione continua; una meta' e' la lettura naturale di "nevica invece di piovere".
SNOW_FRACTION_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class Prediction:
    """Previsioni e osservazioni accumulate su uno split, in memoria compatta."""

    t2m_mean: np.ndarray
    t2m_target: np.ndarray
    tp_probability: np.ndarray
    tp_occurrence: np.ndarray
    month: np.ndarray
    lead: np.ndarray
    snow_probability: np.ndarray | None = None
    snow_occurrence: np.ndarray | None = None

    def with_calibrated_tp(self, calibrator: ProbabilityCalibrator) -> Prediction:
        """Copia con le probabilita' di pioggia corrette dalla mappa di calibrazione.

        La probabilita' di neve viene riscalata in proporzione: e' il prodotto fra la
        probabilita' che precipiti e la quota di neve, quindi correggere la prima
        senza la seconda renderebbe le due incoerenti.
        """
        corretta = calibrator.apply(self.tp_probability)
        neve = self.snow_probability
        if neve is not None:
            quota = np.divide(
                neve,
                self.tp_probability,
                out=np.zeros_like(neve),
                where=self.tp_probability > 1e-6,
            )
            neve = (corretta * quota).astype(np.float32)
        return Prediction(
            t2m_mean=self.t2m_mean,
            t2m_target=self.t2m_target,
            tp_probability=corretta,
            tp_occurrence=self.tp_occurrence,
            month=self.month,
            lead=self.lead,
            snow_probability=neve,
            snow_occurrence=self.snow_occurrence,
        )


def climatology_from_slots(
    values: np.ndarray, months: Sequence[int], slot_of_day: Sequence[int]
) -> dict[tuple[int, int], float]:
    """Media per (mese, slot del giorno), la forma piu' semplice di climatologia utile.

    Separare l'ora del giorno e' necessario: la temperatura media di gennaio alle 06
    e alle 12 differisce piu' di quanto differiscano due mesi consecutivi alla stessa
    ora.
    """
    accumulo: dict[tuple[int, int], list[float]] = {}
    for indice, valore in enumerate(values):
        chiave = (int(months[indice]), int(slot_of_day[indice]))
        accumulo.setdefault(chiave, []).append(float(valore))
    return {chiave: float(np.mean(v)) for chiave, v in accumulo.items()}


def brier_score(probability: np.ndarray, outcome: np.ndarray) -> float:
    """Errore quadratico medio della probabilita' prevista."""
    if probability.size == 0:
        return float("nan")
    return float(np.mean((probability - outcome) ** 2))


def brier_skill_score(
    probability: np.ndarray, outcome: np.ndarray, reference: np.ndarray
) -> float:
    """Miglioramento relativo del Brier rispetto a una previsione di riferimento.

    Vale 1 per una previsione perfetta, 0 se equivale al riferimento e negativo se
    fa peggio. E' la forma in cui un punteggio su eventi rari diventa interpretabile.
    """
    riferimento = brier_score(reference, outcome)
    if not np.isfinite(riferimento) or riferimento <= 0.0:
        return float("nan")
    return 1.0 - brier_score(probability, outcome) / riferimento


@dataclass(frozen=True, slots=True)
class ClassificationScore:
    """Qualita' di una decisione binaria presa da una probabilita'.

    Il Brier misura la probabilita', ma chi legge una previsione prende una decisione:
    esco senza ombrello oppure no. Queste sono le metriche di quella decisione.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    n_positive_predicted: int
    n_positive_observed: int


def classification_score(
    probability: np.ndarray, outcome: np.ndarray, *, threshold: float = 0.5
) -> ClassificationScore:
    """Precisione, richiamo e F1 della decisione "evento si" sopra la soglia."""
    previsto = probability >= threshold
    osservato = outcome > 0.5

    veri_positivi = int(np.count_nonzero(previsto & osservato))
    falsi_positivi = int(np.count_nonzero(previsto & ~osservato))
    falsi_negativi = int(np.count_nonzero(~previsto & osservato))

    precisione = veri_positivi / (veri_positivi + falsi_positivi) if previsto.any() else 0.0
    richiamo = veri_positivi / (veri_positivi + falsi_negativi) if osservato.any() else float("nan")
    if precisione + richiamo > 0 and np.isfinite(richiamo):
        f1 = 2 * precisione * richiamo / (precisione + richiamo)
    else:
        f1 = 0.0 if np.isfinite(richiamo) else float("nan")

    return ClassificationScore(
        threshold=float(threshold),
        precision=float(precisione),
        recall=float(richiamo),
        f1=float(f1),
        accuracy=float(np.mean(previsto == osservato)) if probability.size else float("nan"),
        n_positive_predicted=int(np.count_nonzero(previsto)),
        n_positive_observed=int(np.count_nonzero(osservato)),
    )


def best_f1_threshold(
    probability: np.ndarray, outcome: np.ndarray, *, n_steps: int = 99
) -> tuple[float, float]:
    """Soglia che massimizza l'F1, e il valore raggiunto.

    La soglia 0,5 e' una convenzione, non un ottimo: su un evento con frequenza di base
    diversa da un mezzo la decisione migliore cade altrove. Va scelta sulla validazione
    e poi applicata al test, non ottimizzata sul test.
    """
    if probability.size == 0:
        return 0.5, float("nan")
    migliore, punteggio = 0.5, -1.0
    for soglia in np.linspace(0.01, 0.99, n_steps):
        corrente = classification_score(probability, outcome, threshold=float(soglia)).f1
        if np.isfinite(corrente) and corrente > punteggio:
            migliore, punteggio = float(soglia), float(corrente)
    return migliore, punteggio


def reliability_table(
    probability: np.ndarray,
    outcome: np.ndarray,
    *,
    model: str,
    split: str,
    fold: int,
    variable: str,
    lead_slot: int = -1,
    n_bins: int = RELIABILITY_BINS,
) -> pl.DataFrame:
    """Frequenza osservata per intervallo di probabilita' prevista.

    E' la verifica diretta dell'affidabilita' richiesta: se il modello dice 30 % di
    pioggia, deve piovere nel 30 % dei casi in cui lo dice.
    """
    bordi = np.linspace(0.0, 1.0, n_bins + 1)
    indici = np.clip(np.digitize(probability, bordi[1:-1], right=False), 0, n_bins - 1)

    righe: list[dict[str, object]] = []
    for intervallo in range(n_bins):
        selezione = indici == intervallo
        conteggio = int(selezione.sum())
        righe.append(
            {
                "model": model,
                "split": split,
                "fold": fold,
                "variable": variable,
                "lead_slot": lead_slot,
                "bin_lower": float(bordi[intervallo]),
                "bin_upper": float(bordi[intervallo + 1]),
                "forecast_mean": (
                    float(probability[selezione].mean()) if conteggio else float("nan")
                ),
                "observed_frequency": (
                    float(outcome[selezione].mean()) if conteggio else float("nan")
                ),
                "count": conteggio,
            }
        )
    return cast_to_schema(pl.DataFrame(righe), RELIABILITY)


@torch.no_grad()
def collect_predictions(
    network: torch.nn.Module,
    dataset: WeatherWindowDataset,
    layout: OutputLayout,
    config: Config,
    *,
    max_windows: int | None = None,
    subsample: int = 64,
) -> Prediction:
    """Esegue il modello sulle finestre dello split e raccoglie previsioni e verita'.

    I punti vengono sottocampionati: conservare ogni pixel di ogni finestra
    occuperebbe gigabyte e le metriche non cambierebbero in modo significativo.
    """
    network.eval()
    generatore = np.random.default_rng(config.training.seed)

    t2m_pred: list[np.ndarray] = []
    t2m_vero: list[np.ndarray] = []
    tp_prob: list[np.ndarray] = []
    tp_occ: list[np.ndarray] = []
    neve_prob: list[np.ndarray] = []
    neve_occ: list[np.ndarray] = []
    mesi: list[np.ndarray] = []
    scadenze: list[np.ndarray] = []

    ha_neve = "sf" in layout.variables and "fraction_logit" in layout.components_of("sf")

    n_finestre = (
        len(dataset.starts) if max_windows is None else min(max_windows, len(dataset.starts))
    )
    for posizione in range(n_finestre):
        campione = dataset[posizione * dataset.crops_per_window]
        caratteristiche = campione[KEY_FEATURES].unsqueeze(0)
        previsione = network(caratteristiche)

        media = layout.select(previsione, "t2m", "mean")[0].numpy()
        probabilita = torch.sigmoid(
            layout.select(previsione, "tp", "occurrence_logit")
        )[0].numpy()

        bersaglio_t2m = campione["target_t2m"].numpy()
        bersaglio_tp = campione["target_tp_occurrence"].numpy()

        if ha_neve:
            quota = torch.sigmoid(layout.select(previsione, "sf", "fraction_logit"))[0].numpy()
            # Nevica se precipita **e** la precipitazione e' prevalentemente neve.
            probabilita_neve = probabilita * quota
            quota_vera = campione["target_sf_fraction"].numpy()
            bersaglio_neve = bersaglio_tp * (quota_vera >= SNOW_FRACTION_THRESHOLD)

        n_slot, altezza, larghezza = media.shape
        piatti = altezza * larghezza
        scelti = generatore.choice(piatti, size=min(subsample, piatti), replace=False)

        inizio = int(campione[KEY_SLOT])
        for slot in range(n_slot):
            istante = dataset.reader.valid_time(inizio + dataset.input_slots + slot)
            t2m_pred.append(media[slot].reshape(-1)[scelti])
            t2m_vero.append(bersaglio_t2m[slot].reshape(-1)[scelti])
            tp_prob.append(probabilita[slot].reshape(-1)[scelti])
            tp_occ.append(bersaglio_tp[slot].reshape(-1)[scelti])
            if ha_neve:
                neve_prob.append(probabilita_neve[slot].reshape(-1)[scelti])
                neve_occ.append(bersaglio_neve[slot].reshape(-1)[scelti])
            mesi.append(np.full(scelti.size, istante.month, dtype=np.int16))
            scadenze.append(np.full(scelti.size, slot, dtype=np.int16))

    if not t2m_pred:
        raise EvaluationError("Nessuna finestra valutata")

    return Prediction(
        t2m_mean=np.concatenate(t2m_pred),
        t2m_target=np.concatenate(t2m_vero),
        tp_probability=np.concatenate(tp_prob),
        tp_occurrence=np.concatenate(tp_occ),
        month=np.concatenate(mesi),
        lead=np.concatenate(scadenze),
        snow_probability=np.concatenate(neve_prob) if neve_prob else None,
        snow_occurrence=np.concatenate(neve_occ) if neve_occ else None,
    )


def metrics_table(
    prediction: Prediction,
    stats: NormStats,
    *,
    model: str,
    split: str,
    fold: int,
    rain_threshold: float = 0.5,
    snow_threshold: float = 0.5,
) -> pl.DataFrame:
    """Metriche per scadenza e per mese, in unita' fisiche dove ha senso.

    Le soglie di decisione sono parametri e non costanti: quella ottimale dipende dalla
    frequenza di base dell'evento e va scelta sulla validazione, mai sul test.
    """
    righe: list[dict[str, object]] = []

    # La climatologia della pioggia e' la frequenza di base osservata nello split, per
    # mese: e' il riferimento contro cui si misura l'abilita' su un evento raro.
    frequenza_base: dict[int, float] = {}
    for mese in np.unique(prediction.month):
        selezione = prediction.month == mese
        frequenza_base[int(mese)] = float(prediction.tp_occurrence[selezione].mean())

    def aggiungi(scadenza: int | None, mese: int | None, selezione: np.ndarray) -> None:
        if not selezione.any():
            return
        errore_norm = prediction.t2m_mean[selezione] - prediction.t2m_target[selezione]
        # L'errore in gradi e' quello leggibile: la deviazione standard di train
        # riporta il residuo normalizzato nelle unita' della variabile.
        scala = stats.std.get("t2m", 1.0)
        rmse = float(np.sqrt(np.mean(errore_norm**2)) * scala)
        mae = float(np.mean(np.abs(errore_norm)) * scala)

        probabilita = prediction.tp_probability[selezione]
        occorrenza = prediction.tp_occurrence[selezione]
        if mese is not None:
            riferimento = np.full_like(probabilita, frequenza_base[mese])
        else:
            riferimento = np.full_like(probabilita, float(occorrenza.mean()))

        righe.append(
            {
                "model": model,
                "split": split,
                "fold": fold,
                "variable": "t2m",
                "lead_slot": -1 if scadenza is None else scadenza,
                "month": -1 if mese is None else mese,
                "metric": "rmse_celsius",
                "value": rmse,
                "n_values": int(selezione.sum()),
            }
        )
        righe.append(
            {
                "model": model, "split": split, "fold": fold, "variable": "t2m",
                "lead_slot": -1 if scadenza is None else scadenza,
                "month": -1 if mese is None else mese,
                "metric": "mae_celsius", "value": mae, "n_values": int(selezione.sum()),
            }
        )
        righe.append(
            {
                "model": model, "split": split, "fold": fold, "variable": "tp",
                "lead_slot": -1 if scadenza is None else scadenza,
                "month": -1 if mese is None else mese,
                "metric": "brier", "value": brier_score(probabilita, occorrenza),
                "n_values": int(selezione.sum()),
            }
        )
        righe.append(
            {
                "model": model, "split": split, "fold": fold, "variable": "tp",
                "lead_slot": -1 if scadenza is None else scadenza,
                "month": -1 if mese is None else mese,
                "metric": "brier_skill_score",
                "value": brier_skill_score(probabilita, occorrenza, riferimento),
                "n_values": int(selezione.sum()),
            }
        )
        righe.append(
            {
                "model": model, "split": split, "fold": fold, "variable": "tp",
                "lead_slot": -1 if scadenza is None else scadenza,
                "month": -1 if mese is None else mese,
                "metric": "base_rate", "value": float(occorrenza.mean()),
                "n_values": int(selezione.sum()),
            }
        )

        def registra_classificazione(
            variabile: str,
            probabilita_evento: np.ndarray,
            osservato: np.ndarray,
            soglia: float,
        ) -> None:
            punteggio = classification_score(probabilita_evento, osservato, threshold=soglia)
            valori = {
                "precision": punteggio.precision,
                "recall": punteggio.recall,
                "f1": punteggio.f1,
                "accuracy": punteggio.accuracy,
                "decision_threshold": punteggio.threshold,
                "calibration_error": calibration_error(probabilita_evento, osservato),
            }
            for nome, valore in valori.items():
                righe.append(
                    {
                        "model": model, "split": split, "fold": fold, "variable": variabile,
                        "lead_slot": -1 if scadenza is None else scadenza,
                        "month": -1 if mese is None else mese,
                        "metric": nome, "value": float(valore),
                        "n_values": int(selezione.sum()),
                    }
                )

        registra_classificazione("tp", probabilita, occorrenza, rain_threshold)

        if prediction.snow_probability is not None and prediction.snow_occurrence is not None:
            neve_prob = prediction.snow_probability[selezione]
            neve_occ = prediction.snow_occurrence[selezione]
            righe.append(
                {
                    "model": model, "split": split, "fold": fold, "variable": "sf",
                    "lead_slot": -1 if scadenza is None else scadenza,
                    "month": -1 if mese is None else mese,
                    "metric": "brier", "value": brier_score(neve_prob, neve_occ),
                    "n_values": int(selezione.sum()),
                }
            )
            righe.append(
                {
                    "model": model, "split": split, "fold": fold, "variable": "sf",
                    "lead_slot": -1 if scadenza is None else scadenza,
                    "month": -1 if mese is None else mese,
                    "metric": "base_rate", "value": float(neve_occ.mean()),
                    "n_values": int(selezione.sum()),
                }
            )
            registra_classificazione("sf", neve_prob, neve_occ, snow_threshold)

    aggiungi(None, None, np.ones_like(prediction.lead, dtype=bool))
    for scadenza in np.unique(prediction.lead):
        aggiungi(int(scadenza), None, prediction.lead == scadenza)
    for mese in np.unique(prediction.month):
        aggiungi(None, int(mese), prediction.month == mese)

    return cast_to_schema(pl.DataFrame(righe), METRICS)


def persistence_baseline(
    dataset: WeatherWindowDataset, *, max_windows: int | None = None
) -> Prediction:
    """Previsione di persistenza: l'ultimo stato osservato, ripetuto su ogni scadenza."""
    specs = target_specs(dataset.config)
    nomi = {spec.name for spec in specs}
    if "t2m" not in nomi or "tp" not in nomi:
        raise EvaluationError("La linea di riferimento richiede i target t2m e tp")

    t2m_pred: list[np.ndarray] = []
    t2m_vero: list[np.ndarray] = []
    tp_prob: list[np.ndarray] = []
    tp_occ: list[np.ndarray] = []
    neve_prob: list[np.ndarray] = []
    neve_occ: list[np.ndarray] = []
    mesi: list[np.ndarray] = []
    scadenze: list[np.ndarray] = []

    n_finestre = (
        len(dataset.starts) if max_windows is None else min(max_windows, len(dataset.starts))
    )
    generatore = np.random.default_rng(dataset.config.training.seed)
    soglia = next(spec.threshold for spec in specs if spec.name == "tp")
    ha_neve = "sf" in nomi

    def occorrenza_neve(pioggia: np.ndarray, neve: np.ndarray) -> np.ndarray:
        """Nevica dove precipita in modo misurabile e la neve e' la parte prevalente."""
        precipita = pioggia > soglia
        quota = np.zeros_like(pioggia, dtype=np.float32)
        np.divide(neve, pioggia, out=quota, where=precipita)
        return (precipita & (quota >= SNOW_FRACTION_THRESHOLD)).astype(np.float32)

    for posizione in range(n_finestre):
        inizio = dataset.starts[posizione]
        totale = dataset.input_slots + dataset.output_slots
        finestra = dataset.reader.read_window(inizio, totale)

        ultimo = dataset.input_slots - 1
        ultimo_t2m = dataset.stats.normalize("t2m", finestra["t2m"][ultimo])
        ultimo_tp = (finestra["tp"][ultimo] > soglia).astype(np.float32)

        bersaglio_t2m = dataset.stats.normalize("t2m", finestra["t2m"][dataset.input_slots :])
        bersaglio_tp = (finestra["tp"][dataset.input_slots :] > soglia).astype(np.float32)

        if ha_neve:
            ultima_neve = occorrenza_neve(finestra["tp"][ultimo], finestra["sf"][ultimo])
            bersaglio_neve = occorrenza_neve(
                finestra["tp"][dataset.input_slots :], finestra["sf"][dataset.input_slots :]
            )

        piatti = ultimo_t2m.size
        scelti = generatore.choice(piatti, size=min(64, piatti), replace=False)

        for slot in range(dataset.output_slots):
            istante = dataset.reader.valid_time(inizio + dataset.input_slots + slot)
            t2m_pred.append(ultimo_t2m.reshape(-1)[scelti])
            t2m_vero.append(bersaglio_t2m[slot].reshape(-1)[scelti])
            tp_prob.append(ultimo_tp.reshape(-1)[scelti])
            tp_occ.append(bersaglio_tp[slot].reshape(-1)[scelti])
            if ha_neve:
                neve_prob.append(ultima_neve.reshape(-1)[scelti])
                neve_occ.append(bersaglio_neve[slot].reshape(-1)[scelti])
            mesi.append(np.full(scelti.size, istante.month, dtype=np.int16))
            scadenze.append(np.full(scelti.size, slot, dtype=np.int16))

    return Prediction(
        t2m_mean=np.concatenate(t2m_pred),
        t2m_target=np.concatenate(t2m_vero),
        tp_probability=np.concatenate(tp_prob),
        tp_occurrence=np.concatenate(tp_occ),
        month=np.concatenate(mesi),
        lead=np.concatenate(scadenze),
        snow_probability=np.concatenate(neve_prob) if neve_prob else None,
        snow_occurrence=np.concatenate(neve_occ) if neve_occ else None,
    )


__all__ = [
    "SNOW_FRACTION_THRESHOLD",
    "ClassificationScore",
    "EvaluationError",
    "Prediction",
    "best_f1_threshold",
    "brier_score",
    "brier_skill_score",
    "classification_score",
    "climatology_from_slots",
    "collect_predictions",
    "metrics_table",
    "persistence_baseline",
    "reliability_table",
]
