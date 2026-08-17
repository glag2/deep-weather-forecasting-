"""Raccolta dei dati mostrati dalla dashboard.

Questo modulo **non importa streamlit**. L'interfaccia sta in
``scripts/dashboard_app.py`` e chiama solo funzioni di qui, per due motivi: le funzioni
restano collaudabili senza avviare un server, e la logica che legge gli artefatti non
si mescola con quella che disegna i riquadri.

Ogni funzione degrada in modo esplicito quando un artefatto manca: la dashboard deve
poter essere aperta su un progetto appena clonato, dove non esiste ancora nulla, e dire
che cosa manca invece di sollevare un'eccezione.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from dwf.config import Config
from dwf.tables import DOWNLOADS, FOLDS, METRICS, SLOTS

# Tecnologie che vale la pena mostrare, con il ruolo che hanno davvero nel progetto.
# La descrizione dice a che cosa serve la tecnologia qui, non che cosa e' in generale.
TECNOLOGIE: tuple[tuple[str, str, str], ...] = (
    ("Ancoraggio diurno", "modellazione",
     "La rete prevede lo scarto dall'osservazione alla stessa ora del giorno prima, "
     "non il valore assoluto. E' l'unica scelta che nel banco esce dal rumore."),
    ("Teste probabilistiche", "modellazione",
     "Gaussiana per la temperatura, hurdle per la pioggia (occorrenza piu' quantita'), "
     "frazione per la neve. Ogni previsione porta la propria incertezza."),
    ("Attenzione a finestre", "architettura",
     "Blocco alternativo con finestre 8x8 e bias di posizione relativa."),
    ("Convoluzione spettrale", "architettura",
     "Blocco alternativo che opera sui modi di Fourier bassi: campo ricettivo globale "
     "a costo lineare."),
    ("Perdita spettrale", "ottimizzazione",
     "Confronta i moduli della trasformata di Fourier per premiare l'ampiezza corretta "
     "senza reintrodurre la doppia penalizzazione di posizione."),
    ("Pesatura spaziale", "ottimizzazione",
     "Peso di area (cos della latitudine) combinato con un fuoco gaussiano su Vigo di "
     "Cadore, entrambi normalizzati a media unitaria."),
    ("Calibrazione isotonica", "affidabilita'",
     "PAVA scritto a mano, adattato sulla validazione e misurato sul test."),
    ("Validazione a finestra mobile", "protocollo",
     "Sei fold i cui blocchi di test coprono tutti i dodici mesi."),
    ("Fisica derivata", "caratteristiche",
     "Geometria solare (Spencer 1971) e termodinamica dell'aria umida, calore latente "
     "incluso, ricavate senza scaricare nuove variabili."),
    ("Zarr piu' Polars", "dati",
     "Zarr a blocchi per il tensore, Parquet per i registri: accesso casuale a finestre "
     "senza caricare il periodo intero."),
    ("Checkpoint senza pickle", "sicurezza",
     "Pesi in .npz letti con allow_pickle=False: un file di modello non puo' eseguire "
     "codice."),
)


class DashboardError(RuntimeError):
    """Errore di raccolta dei dati per la dashboard."""


@dataclass(frozen=True, slots=True)
class Riquadro:
    """Un valore da mostrare, con l'unita' gia' incorporata nel testo."""

    etichetta: str
    valore: str
    nota: str = ""


def _leggi(spec, directory: Path) -> pl.DataFrame | None:
    percorso = directory / spec.filename
    if not percorso.exists():
        return None
    try:
        return pl.read_parquet(percorso)
    except Exception:  # il file puo' essere in scrittura proprio ora
        return None


# --------------------------------------------------------------------------- #
# Panoramica
# --------------------------------------------------------------------------- #


def panoramica(config: Config) -> list[Riquadro]:
    """Numeri d'insieme del progetto, letti dagli artefatti reali."""
    righe = config.area.n_latitudes
    colonne = config.area.n_longitudes
    riquadri = [
        Riquadro("Griglia", f"{righe} x {colonne}", f"{righe * colonne:,} punti"),
        Riquadro("Risoluzione", f"{config.area.resolution} gradi", "nativa ERA5"),
        Riquadro(
            "Finestra",
            f"{config.windows.input_slots} -> {config.windows.output_slots} slot",
            f"{config.windows.input_slots // config.time.slots_per_day} giorni di storico, "
            f"{config.windows.output_slots // config.time.slots_per_day} di previsione",
        ),
    ]

    slots = _leggi(SLOTS, config.tables_dir)
    if slots is not None and slots.height:
        primo = slots["valid_time"].min()
        ultimo = slots["valid_time"].max()
        riquadri.append(
            Riquadro(
                "Slot ingeriti",
                f"{slots.height:,}",
                f"da {primo:%Y-%m-%d} a {ultimo:%Y-%m-%d}",
            )
        )
    else:
        riquadri.append(Riquadro("Slot ingeriti", "nessuno", "eseguire scripts/ingest_era5.py"))

    scaricati = _leggi(DOWNLOADS, config.tables_dir)
    if scaricati is not None and scaricati.height:
        completati = scaricati.filter(pl.col("status") == "downloaded").height
        byte = scaricati["size_bytes"].sum() or 0
        riquadri.append(
            Riquadro("File scaricati", f"{completati}", f"{byte / 1e9:.1f} GB di GRIB")
        )

    return riquadri


def copertura_mensile(config: Config) -> pl.DataFrame | None:
    """Quanti slot esistono per ciascun mese: mostra i buchi a colpo d'occhio."""
    slots = _leggi(SLOTS, config.tables_dir)
    if slots is None or not slots.height:
        return None
    return (
        slots.with_columns(pl.col("valid_time").dt.strftime("%Y-%m").alias("mese"))
        .group_by("mese")
        .agg(pl.len().alias("slot"))
        .sort("mese")
    )


def struttura_fold(config: Config) -> pl.DataFrame | None:
    """Confini dei blocchi di ciascun fold, per vedere che test e train non si toccano."""
    return _leggi(FOLDS, config.tables_dir)


# --------------------------------------------------------------------------- #
# Modello
# --------------------------------------------------------------------------- #


def informazioni_modello(config: Config, fold: int) -> dict[str, Any]:
    """Metadati del checkpoint, senza caricare i pesi."""
    destinazione = config.fold_dir(fold)
    percorso = destinazione / "metadata.json"
    if not percorso.exists():
        return {"disponibile": False, "percorso": str(destinazione)}

    metadati = json.loads(percorso.read_text(encoding="utf-8"))
    storia_path = destinazione / "history.json"
    storia = json.loads(storia_path.read_text(encoding="utf-8")) if storia_path.exists() else []

    return {
        "disponibile": True,
        "percorso": str(destinazione),
        "metadati": metadati,
        "storia": storia,
        "variante": config.model.variant,
        "ancoraggio": config.model.anchor_diurnal,
        "canali_base": config.model.base_channels,
        "profondita": config.model.depth,
        "blocchi_per_livello": config.model.blocks_per_level,
        "aggiornato": datetime.fromtimestamp(percorso.stat().st_mtime, tz=UTC),
    }


def curva_apprendimento(informazioni: dict[str, Any]) -> pl.DataFrame | None:
    """Perdita di train e validazione per epoca."""
    storia = informazioni.get("storia") or []
    if not storia:
        return None
    return pl.DataFrame(storia)


# --------------------------------------------------------------------------- #
# Prestazioni
# --------------------------------------------------------------------------- #


def metriche(config: Config, fold: int, split: str = "test") -> pl.DataFrame | None:
    """Tabella delle metriche prodotta dalla valutazione."""
    tabella = _leggi(METRICS, config.fold_dir(fold))
    if tabella is None:
        tabella = _leggi(METRICS, config.tables_dir)
    if tabella is None or not tabella.height:
        return None
    return tabella.filter((pl.col("split") == split) & (pl.col("fold") == fold))


def riepilogo_metriche(tabella: pl.DataFrame) -> pl.DataFrame:
    """Le metriche aggregate su tutti i mesi e tutte le scadenze, per modello.

    Il filtro su ``month == -1`` e ``lead_slot == -1`` seleziona le righe gia'
    aggregate dalla valutazione, invece di rifare una media di medie che pesarebbe
    male i mesi con meno casi.
    """
    return (
        tabella.filter((pl.col("month") == -1) & (pl.col("lead_slot") == -1))
        .select("model", "variable", "metric", "value", "n_values")
        .sort("variable", "metric", "model")
    )


def per_scadenza(tabella: pl.DataFrame, variabile: str, metrica: str) -> pl.DataFrame:
    """Andamento di una metrica al crescere della scadenza."""
    return (
        tabella.filter(
            (pl.col("variable") == variabile)
            & (pl.col("metric") == metrica)
            & (pl.col("month") == -1)
            & (pl.col("lead_slot") >= 0)
        )
        .select("model", "lead_slot", "value")
        .sort("lead_slot", "model")
    )


# --------------------------------------------------------------------------- #
# Confronto visivo e mappa degli errori
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Confronto:
    """Un campo previsto e la verita' corrispondente, gia' in unita' fisiche."""

    previsto: np.ndarray
    osservato: np.ndarray
    riferimento: np.ndarray | None
    variabile: str
    scadenza: int
    istante: datetime | None

    @property
    def differenza(self) -> np.ndarray:
        return self.previsto - self.osservato

    @property
    def errore_assoluto_medio(self) -> float:
        return float(np.mean(np.abs(self.differenza)))


def confronto_visivo(
    config: Config, fold: int, *, split: str = "test", posizione: int = 0, scadenza: int = 2
) -> Confronto:
    """Esegue il modello su una finestra e restituisce previsione e verita'.

    Si usa il blocco di test, cioe' dati che il modello non ha mai visto: mostrare un
    confronto su dati di addestramento darebbe un'impressione falsamente buona.
    """
    import torch

    from dwf.data.dataset import (
        KEY_FEATURES,
        WeatherWindowDataset,
        build_reader,
        sample_starts,
        split_baselines,
    )
    from dwf.train import load_checkpoint

    rete, stats, input_layout, output_layout = load_checkpoint(config, fold)
    inizi = sample_starts(config, fold, split)
    if not inizi:
        raise DashboardError(f"Nessuna finestra nel blocco {split!r} del fold {fold}")
    posizione = max(0, min(posizione, len(inizi) - 1))

    dataset = WeatherWindowDataset(
        config, input_layout, stats, inizi, build_reader(config, input_layout),
        crop_size=None, crops_per_window=1,
    )
    campione = dataset[posizione]
    _, riferimenti = split_baselines(campione)

    rete.eval()
    with torch.no_grad():
        uscita = output_layout.apply_anchor(
            rete(campione[KEY_FEATURES].unsqueeze(0)),
            {nome: valore.unsqueeze(0) for nome, valore in riferimenti.items()},
        )
        media = output_layout.select(uscita, "t2m", "mean")[0].numpy()

    previsto = stats.denormalize("t2m", media[scadenza])
    osservato = stats.denormalize("t2m", campione["target_t2m"].numpy()[scadenza])
    riferimento = None
    if "t2m" in riferimenti:
        riferimento = stats.denormalize("t2m", riferimenti["t2m"].numpy()[scadenza])

    return Confronto(
        previsto=previsto,
        osservato=osservato,
        riferimento=riferimento,
        variabile="t2m",
        scadenza=scadenza,
        istante=_istante_di(config, int(campione["start_slot"]), scadenza),
    )


def _istante_di(config: Config, inizio: int, scadenza: int) -> datetime | None:
    slots = _leggi(SLOTS, config.tables_dir)
    if slots is None or not slots.height:
        return None
    indice = inizio + config.windows.input_slots + scadenza
    if indice >= slots.height:
        return None
    return slots["valid_time"][indice]


def mappa_errori(
    config: Config, fold: int, *, split: str = "test", n_finestre: int = 12, scadenza: int = 2
) -> tuple[np.ndarray, int]:
    """Errore quadratico medio per cella, aggregato su piu' finestre.

    Mostra **dove** il modello sbaglia, che una metrica scalare non puo' dire: un errore
    concentrato sui rilievi ha cause diverse da uno diffuso sull'oceano.
    """
    import torch

    from dwf.data.dataset import (
        KEY_FEATURES,
        WeatherWindowDataset,
        build_reader,
        sample_starts,
        split_baselines,
    )
    from dwf.train import load_checkpoint

    rete, stats, input_layout, output_layout = load_checkpoint(config, fold)
    inizi = sample_starts(config, fold, split)
    if not inizi:
        raise DashboardError(f"Nessuna finestra nel blocco {split!r} del fold {fold}")
    inizi = inizi[:n_finestre]

    dataset = WeatherWindowDataset(
        config, input_layout, stats, inizi, build_reader(config, input_layout),
        crop_size=None, crops_per_window=1,
    )

    rete.eval()
    somma: np.ndarray | None = None
    for posizione in range(len(inizi)):
        campione = dataset[posizione]
        _, riferimenti = split_baselines(campione)
        with torch.no_grad():
            uscita = output_layout.apply_anchor(
                rete(campione[KEY_FEATURES].unsqueeze(0)),
                {nome: valore.unsqueeze(0) for nome, valore in riferimenti.items()},
            )
            media = output_layout.select(uscita, "t2m", "mean")[0].numpy()

        previsto = stats.denormalize("t2m", media[scadenza])
        osservato = stats.denormalize("t2m", campione["target_t2m"].numpy()[scadenza])
        quadrato = (previsto - osservato) ** 2
        somma = quadrato if somma is None else somma + quadrato

    assert somma is not None
    return np.sqrt(somma / len(inizi)), len(inizi)


# --------------------------------------------------------------------------- #
# Risorse e avanzamento
# --------------------------------------------------------------------------- #


def risorse() -> dict[str, Any]:
    """Uso di CPU e memoria della macchina e dei processi Python del progetto."""
    import psutil

    memoria = psutil.virtual_memory()
    processi = []
    for processo in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            riga = processo.info
            comando = " ".join(riga.get("cmdline") or [])
            if "scripts" not in comando and "dwf" not in comando:
                continue
            processi.append(
                {
                    "pid": riga["pid"],
                    "comando": _accorcia(comando),
                    "cpu": processo.cpu_percent(interval=None),
                    "memoria_mb": processo.memory_info().rss / 1e6,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {
        "cpu_totale": psutil.cpu_percent(interval=0.1),
        "cpu_per_core": psutil.cpu_percent(interval=None, percpu=True),
        "core": psutil.cpu_count(logical=True),
        "memoria_usata_gb": memoria.used / 1e9,
        "memoria_totale_gb": memoria.total / 1e9,
        "memoria_percento": memoria.percent,
        "processi": sorted(processi, key=lambda r: r["memoria_mb"], reverse=True)[:8],
    }


def _accorcia(comando: str, massimo: int = 70) -> str:
    parti = [p for p in comando.split() if p.endswith(".py") or p.startswith("--")]
    testo = " ".join(parti) or comando
    return testo if len(testo) <= massimo else testo[: massimo - 3] + "..."


def spazio_dati(config: Config) -> list[Riquadro]:
    """Quanto occupano su disco le varie forme dei dati."""
    voci = [
        ("GRIB grezzi", config.raw_dir),
        ("Store Zarr", config.zarr_path),
        ("Tabelle", config.tables_dir),
        ("Modelli", config.models_dir),
    ]
    riquadri = []
    for etichetta, percorso in voci:
        if not Path(percorso).exists():
            riquadri.append(Riquadro(etichetta, "assente"))
            continue
        byte = sum(f.stat().st_size for f in Path(percorso).rglob("*") if f.is_file())
        riquadri.append(Riquadro(etichetta, f"{byte / 1e9:.2f} GB"))
    return riquadri


__all__ = [
    "TECNOLOGIE",
    "Confronto",
    "DashboardError",
    "Riquadro",
    "confronto_visivo",
    "copertura_mensile",
    "curva_apprendimento",
    "informazioni_modello",
    "mappa_errori",
    "metriche",
    "panoramica",
    "per_scadenza",
    "riepilogo_metriche",
    "risorse",
    "spazio_dati",
    "struttura_fold",
]
