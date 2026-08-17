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
    righe = config.region.n_lat
    colonne = config.region.n_lon
    riquadri = [
        Riquadro("Griglia", f"{righe} x {colonne}", f"{righe * colonne:,} punti"),
        Riquadro("Risoluzione", f"{config.region.grid_step} gradi", "nativa ERA5"),
        Riquadro(
            "Area",
            f"{config.region.south:g} - {config.region.north:g} N",
            f"{config.region.west:g} - {config.region.east:g} E",
        ),
        Riquadro(
            "Finestra",
            f"{config.windows.input_slots} -> {config.windows.output_slots} slot",
            f"{config.windows.input_slots // config.time.slots_per_day} giorni di storico, "
            f"{config.windows.output_slots // config.time.slots_per_day} di previsione",
        ),
    ]

    slots = _leggi(SLOTS, config.tables_dir)
    if slots is not None and slots.height:
        # `slots.parquet` cataloga l'intero periodo configurato, non cio' che e' stato
        # davvero ingerito: contare le righe direbbe un numero molto piu' grande del
        # vero. La colonna `usable` e' l'unica che indica la presenza reale nel tensore.
        utilizzabili = slots.filter(pl.col("usable"))
        if utilizzabili.height:
            primo = utilizzabili["valid_time"].min()
            ultimo = utilizzabili["valid_time"].max()
            nota = f"da {primo:%Y-%m-%d} a {ultimo:%Y-%m-%d}"
        else:
            nota = "nessuno presente nello store"
        riquadri.append(
            Riquadro(
                "Slot utilizzabili",
                f"{utilizzabili.height:,} / {slots.height:,}",
                nota,
            )
        )
    else:
        riquadri.append(
            Riquadro("Slot utilizzabili", "nessuno", "eseguire scripts/ingest_era5.py")
        )

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
        .agg(
            pl.len().alias("catalogati"),
            pl.col("usable").sum().alias("presenti"),
        )
        .with_columns(
            (pl.col("presenti") / pl.col("catalogati")).alias("frazione")
        )
        .sort("mese")
    )


def struttura_fold(config: Config) -> pl.DataFrame | None:
    """Confini dei blocchi di ciascun fold, per vedere che test e train non si toccano.

    La tabella su disco ha una riga per slot e per fold, quindi decine di migliaia di
    righe: mostrarla cosi' com'e' non fa vedere la struttura, la nasconde. Qui si
    riduce ai confini, che sono l'unica cosa che serve per giudicare la separazione
    fra blocchi.
    """
    tabella = _leggi(FOLDS, config.tables_dir)
    if tabella is None or not tabella.height:
        return None
    return (
        tabella.group_by("fold", "split")
        .agg(
            pl.col("slot_index").min().alias("primo_slot"),
            pl.col("slot_index").max().alias("ultimo_slot"),
            pl.len().alias("slot"),
            pl.col("is_sample_start").sum().alias("inizi_ammessi"),
        )
        .sort("fold", "split")
    )


# --------------------------------------------------------------------------- #
# Modello
# --------------------------------------------------------------------------- #


def parametri_salvati(pesi: Path) -> int | None:
    """Quanti parametri contiene davvero il file dei pesi.

    Il conteggio si legge dal file invece che dalla configurazione perche' le due cose
    possono divergere: se il checkpoint e' stato addestrato con un'altra impostazione,
    e' il file a dire com'e' fatto il modello che si sta per caricare.
    """
    if not pesi.exists():
        return None
    with np.load(pesi, allow_pickle=False) as archivio:
        return int(sum(archivio[nome].size for nome in archivio.files))


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
        "n_parametri": parametri_salvati(destinazione / "weights.npz"),
        "canali_ingresso": metadati.get("in_channels"),
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


def coerenza_artefatti(config: Config, fold: int) -> list[str]:
    """Segnala artefatti incoerenti fra loro, che altrimenti passano inosservati.

    Nasce da un guasto reale: un banco di prova che scriveva nella stessa cartella del
    modello a scala piena ne ha sovrascritto pesi, statistiche e cronologia. Il modello
    continuava a caricarsi senza errori, ma non era piu' quello addestrato. Un confronto
    fra le date dei file e fra i canali attesi lo avrebbe reso evidente subito.

    Restituisce un elenco di problemi in italiano, vuoto se non ce ne sono.
    """
    from dwf.data.features import InputLayout

    destinazione = config.fold_dir(fold)
    pesi = destinazione / "weights.npz"
    if not pesi.exists():
        return [f"Pesi assenti in {destinazione}"]

    problemi: list[str] = []
    metadati_path = destinazione / "metadata.json"
    if not metadati_path.exists():
        problemi.append("Pesi presenti ma metadata.json assente")
        return problemi

    metadati = json.loads(metadati_path.read_text(encoding="utf-8"))
    attesi = InputLayout.from_config(config).n_channels
    dichiarati = metadati.get("in_channels")
    if dichiarati is not None and dichiarati != attesi:
        problemi.append(
            f"Il checkpoint e' stato addestrato con {dichiarati} canali, la "
            f"configurazione attuale ne produce {attesi}"
        )

    quando_pesi = pesi.stat().st_mtime
    for nome in ("norm_stats.parquet", "history.json"):
        compagno = destinazione / nome
        if not compagno.exists():
            problemi.append(f"{nome} assente accanto ai pesi")
            continue
        # Una tolleranza di un minuto assorbe l'ordine di scrittura dentro la stessa
        # esecuzione; oltre, il file viene da un'altra esecuzione.
        if compagno.stat().st_mtime > quando_pesi + 60:
            problemi.append(
                f"{nome} e' piu' recente dei pesi di "
                f"{int(compagno.stat().st_mtime - quando_pesi)} s: proviene "
                "probabilmente da un'altra esecuzione"
            )

    storia_path = destinazione / "history.json"
    if storia_path.exists() and (epoca := metadati.get("epoch")) is not None:
        storia = json.loads(storia_path.read_text(encoding="utf-8"))
        if storia and epoca >= len(storia):
            problemi.append(
                f"I metadati indicano l'epoca {epoca} ma la cronologia ne contiene "
                f"{len(storia)}"
            )

    return problemi


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
    selezione = tabella.filter((pl.col("split") == split) & (pl.col("fold") == fold))
    # Una tabella esistente ma senza righe per questo blocco significa "valutazione non
    # ancora eseguita", non "nessun errore": va distinta, altrimenti la pagina mostra
    # una tabella vuota che sembra un guasto.
    return selezione if selezione.height else None


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


def _percorso_cache(config: Config, fold: int, chiave: str) -> Path | None:
    """File di cache per un calcolo pesante, valido finche' i pesi non cambiano.

    La chiave include la data dei pesi, quindi riaddestrare invalida la cache da solo:
    non serve ricordarsi di svuotarla, e non si rischia di mostrare la mappa di errore
    di un modello che non esiste piu'.
    """
    pesi = config.fold_dir(fold) / "weights.npz"
    if not pesi.exists():
        return None
    marca = int(pesi.stat().st_mtime)
    cartella = config.fold_dir(fold) / "cache"
    cartella.mkdir(parents=True, exist_ok=True)
    return cartella / f"{chiave}_{marca}.npz"


def mappa_errori(
    config: Config,
    fold: int,
    *,
    split: str = "test",
    n_finestre: int = 12,
    scadenza: int = 2,
    usa_cache: bool = True,
) -> tuple[np.ndarray, int]:
    """Errore quadratico medio per cella, aggregato su piu' finestre.

    Mostra **dove** il modello sbaglia, che una metrica scalare non puo' dire: un errore
    concentrato sui rilievi ha cause diverse da uno diffuso sull'oceano.

    Il calcolo richiede una passata della rete sull'intero dominio per ogni finestra, ed
    e' stato misurato in decine di secondi: troppo per una pagina che si ricarica a ogni
    interazione. Il risultato viene quindi conservato su disco.
    """
    import torch

    cache = (
        _percorso_cache(config, fold, f"mappa_{split}_{n_finestre}_{scadenza}")
        if usa_cache
        else None
    )
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as archivio:
            return archivio["mappa"], int(archivio["finestre"])

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
    mappa = np.sqrt(somma / len(inizi))
    if cache is not None:
        np.savez_compressed(cache, mappa=mappa, finestre=len(inizi))
    return mappa, len(inizi)


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
