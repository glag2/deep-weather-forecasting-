"""Inferenza sul dominio intero e conversione in unita' fisiche.

La rete lavora su valori normalizzati e su parametri di distribuzioni; qui vengono
riportati a cio' che un utente legge: gradi Celsius, probabilita' di pioggia,
millimetri attesi, probabilita' che quella precipitazione sia neve.

Nota sulla natura della previsione. ERA5 pubblica con circa sei giorni di ritardo,
quindi l'ultima finestra disponibile non finisce oggi: la previsione prodotta e' una
**hindcast verificabile**, cioe' riguarda giorni gia' trascorsi e confrontabili con
l'osservato. E' un limite della sorgente, non del modello, ed e' anche cio' che rende
la valutazione onesta.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch

from dwf.calibration import ProbabilityCalibrator
from dwf.data.dataset import ZarrWindowReader
from dwf.data.features import InputLayout, NormStats, build_input_tensor
from dwf.models.heads import OutputLayout
from dwf.tables import CALIBRATION, FORECAST, cast_to_schema, read_table

if TYPE_CHECKING:  # pragma: no cover - solo per i tipi
    from dwf.config import Config


class PredictionError(RuntimeError):
    """Errore nella produzione di una previsione."""


@dataclass(frozen=True, slots=True)
class Forecast:
    """Previsione sul dominio intero, in unita' fisiche."""

    init_time: datetime
    valid_times: tuple[datetime, ...]
    latitudes: np.ndarray
    longitudes: np.ndarray
    t2m_mean: np.ndarray
    t2m_std: np.ndarray
    precip_probability: np.ndarray
    precip_amount: np.ndarray
    snow_probability: np.ndarray

    @property
    def n_lead(self) -> int:
        return len(self.valid_times)


def latest_usable_start(config: Config, usable: np.ndarray) -> int:
    """Ultima finestra completa di input disponibile nello store.

    Serve la finestra di input **piu' recente** i cui slot siano tutti utilizzabili:
    e' da li' che si estrapola. Gli slot di target possono non esistere ancora, e
    infatti nella previsione operativa non esistono.
    """
    richiesti = config.windows.input_slots
    for inizio in range(len(usable) - richiesti, -1, -1):
        if usable[inizio : inizio + richiesti].all():
            return inizio
    raise PredictionError(
        f"Nessuna finestra di {richiesti} slot consecutivi utilizzabili: "
        f"ingerire piu' mesi prima di prevedere"
    )


def load_calibrator(config: Config, fold: int) -> ProbabilityCalibrator | None:
    """Mappa di calibrazione del fold, se e' stata stimata.

    Assente non e' un errore: un modello appena addestrato non ha ancora una
    calibrazione, e le probabilita' grezze restano utilizzabili, solo meno fedeli.
    """
    percorso = config.fold_dir(fold) / CALIBRATION.filename
    if not percorso.exists():
        return None
    return ProbabilityCalibrator.from_table(read_table(CALIBRATION, percorso.parent))


@torch.no_grad()
def predict_window(
    config: Config,
    network: torch.nn.Module,
    input_layout: InputLayout,
    output_layout: OutputLayout,
    stats: NormStats,
    reader: ZarrWindowReader,
    start: int,
    calibrator: ProbabilityCalibrator | None = None,
) -> Forecast:
    """Esegue la rete su una finestra e converte l'uscita in unita' fisiche."""
    network.eval()
    n_input = config.windows.input_slots
    finestra = reader.read_window(start, n_input)

    riferimento = reader.valid_time(start + n_input - 1)
    caratteristiche = build_input_tensor(
        input_layout,
        finestra,
        stats,
        static_fields=reader.static,
        latitudes=reader.latitudes,
        reference_time=riferimento,
        slot_hours=config.time.slot_hours,
    )
    previsione = network(torch.from_numpy(caratteristiche).unsqueeze(0))

    media_norm = output_layout.select(previsione, "t2m", "mean")[0].numpy()
    log_var = output_layout.select(previsione, "t2m", "log_var")[0].numpy()
    t2m_mean = stats.denormalize("t2m", media_norm)
    # La deviazione standard e' in unita' normalizzate: basta riscalarla, perche' la
    # trasformazione della temperatura e' l'identita'.
    t2m_std = np.exp(0.5 * np.clip(log_var, -10.0, 10.0)) * stats.std["t2m"]

    probabilita = torch.sigmoid(
        output_layout.select(previsione, "tp", "occurrence_logit")
    )[0].numpy()
    if calibrator is not None:
        # La rete ordina bene ma sbaglia la scala: senza questa correzione un "40 % di
        # pioggia" non corrisponde a piovere nel 40 % dei casi in cui viene dichiarato.
        probabilita = calibrator.apply(probabilita)
    quantita = stats.denormalize(
        "tp", output_layout.select(previsione, "tp", "amount")[0].numpy()
    )
    # La quantita' e' condizionata alla pioggia: il valore atteso la pesa con la
    # probabilita' che piova davvero.
    quantita_attesa = np.clip(quantita, 0.0, None) * probabilita

    frazione = torch.sigmoid(
        output_layout.select(previsione, "sf", "fraction_logit")
    )[0].numpy()
    # Probabilita' che nevichi = probabilita' che precipiti, per la quota di neve.
    neve = probabilita * frazione

    istanti = tuple(
        reader.valid_time(start + n_input + passo)
        for passo in range(config.windows.output_slots)
    )

    return Forecast(
        init_time=riferimento,
        valid_times=istanti,
        latitudes=reader.latitudes,
        # Entrambe le coordinate vengono dallo store, non dalla configurazione: sono
        # gli assi dei dati che si stanno prevedendo.
        longitudes=reader.longitudes,
        t2m_mean=t2m_mean.astype(np.float32),
        t2m_std=t2m_std.astype(np.float32),
        precip_probability=probabilita.astype(np.float32),
        # In millimetri: i metri di ERA5 non sono leggibili.
        precip_amount=(quantita_attesa * 1000.0).astype(np.float32),
        snow_probability=neve.astype(np.float32),
    )


def forecast_to_table(forecast: Forecast, *, stride: int = 1) -> pl.DataFrame:
    """Previsione in forma lunga, un record per punto e scadenza.

    `stride` sottocampiona la griglia: la tabella completa ha quasi un milione di
    righe per previsione, utile da archiviare ma scomoda da ispezionare.
    """
    latitudini = forecast.latitudes[::stride]
    longitudini = forecast.longitudes[::stride]
    griglia_lat, griglia_lon = np.meshgrid(latitudini, longitudini, indexing="ij")
    piatta_lat = griglia_lat.reshape(-1)
    piatta_lon = griglia_lon.reshape(-1)

    blocchi: list[pl.DataFrame] = []
    for scadenza, istante in enumerate(forecast.valid_times):
        ore = int((istante - forecast.init_time).total_seconds() // 3600)
        sezione = (slice(None, None, stride), slice(None, None, stride))
        blocchi.append(
            pl.DataFrame(
                {
                    "init_time": [forecast.init_time] * piatta_lat.size,
                    "valid_time": [istante] * piatta_lat.size,
                    "lead_slot": np.full(piatta_lat.size, scadenza, dtype=np.int16),
                    "lead_hours": np.full(piatta_lat.size, ore, dtype=np.int16),
                    "latitude": piatta_lat,
                    "longitude": piatta_lon,
                    "t2m_mean": forecast.t2m_mean[scadenza][sezione].reshape(-1),
                    "t2m_std": forecast.t2m_std[scadenza][sezione].reshape(-1),
                    "precip_probability": forecast.precip_probability[scadenza][
                        sezione
                    ].reshape(-1),
                    "precip_amount": forecast.precip_amount[scadenza][sezione].reshape(-1),
                    "snow_probability": forecast.snow_probability[scadenza][
                        sezione
                    ].reshape(-1),
                }
            )
        )
    return cast_to_schema(pl.concat(blocchi), FORECAST)


def summarize(forecast: Forecast) -> pl.DataFrame:
    """Riepilogo per scadenza: quanto e' calda, piovosa e nevosa la previsione."""
    righe = []
    for scadenza, istante in enumerate(forecast.valid_times):
        righe.append(
            {
                "lead_slot": scadenza,
                "valid_time": istante,
                "t2m_mean_celsius": float(forecast.t2m_mean[scadenza].mean()),
                "t2m_std_celsius": float(forecast.t2m_std[scadenza].mean()),
                "precip_probability_mean": float(
                    forecast.precip_probability[scadenza].mean()
                ),
                "precip_amount_mm_mean": float(forecast.precip_amount[scadenza].mean()),
                "snow_probability_mean": float(forecast.snow_probability[scadenza].mean()),
            }
        )
    return pl.DataFrame(righe)


__all__ = [
    "Forecast",
    "PredictionError",
    "forecast_to_table",
    "latest_usable_start",
    "load_calibrator",
    "predict_window",
    "summarize",
]
