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
    struttura = (
        tabella.group_by("fold", "split")
        .agg(
            pl.col("slot_index").min().alias("primo_slot"),
            pl.col("slot_index").max().alias("ultimo_slot"),
            pl.len().alias("slot"),
        )
        .sort("fold", "split")
    )
    # La colonna `is_sample_start` del file vale per la finestra in uso quando i fold
    # sono stati scritti. Mostrarla accanto a una configurazione con finestra diversa
    # darebbe un conteggio che l'addestramento non usa: si ricalcola sulla finestra
    # corrente, la stessa che vede `sample_starts`.
    finestra = config.windows.input_slots + config.windows.output_slots
    ammessi = [
        max(0, int(riga["slot"]) - finestra + 1) for riga in struttura.iter_rows(named=True)
    ]
    return struttura.with_columns(pl.Series("inizi_ammessi", ammessi, dtype=pl.UInt32))


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

    # Ingerire altri mesi sposta i confini fra addestramento, validazione e test senza
    # che nulla smetta di funzionare: il modello si carica, la valutazione gira, e
    # gira su finestre che prima stavano dall'altra parte del confine.
    from dwf.data.dataset import confronta_impronte, data_fingerprint

    # Se il catalogo non esiste non c'e' nulla da confrontare, e la sua assenza la
    # segnala gia' la pagina dei dati: ripeterla qui aggiungerebbe solo rumore.
    catalogo_presente = all(
        (config.tables_dir / f"{nome}.parquet").exists() for nome in ("slots", "folds")
    )
    if catalogo_presente:
        try:
            problemi.extend(
                confronta_impronte(metadati.get("data"), data_fingerprint(config, fold))
            )
        except Exception as errore:
            problemi.append(f"Impronta dei dati non verificabile: {errore}")

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


# L'ordine alfabetico metterebbe in cima "sf accuracy", che su una variabile presente
# nel 9 % dei casi e' la metrica piu' facile da fraintendere. Le righe vengono quindi
# ordinate per rilevanza: prima la temperatura, poi le probabilita', e dentro ciascuna
# variabile prima gli errori e per ultime le quantita' che dipendono da una soglia.
ORDINE_VARIABILI = ("t2m", "tp", "sf")
ORDINE_METRICHE = (
    "rmse_celsius",
    "mae_celsius",
    "brier",
    "brier_skill_score",
    "calibration_error",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "base_rate",
    "decision_threshold",
)
ORDINE_MODELLI = ("dwf", "persistence_diurnal", "persistence")


def _rango(valore: pl.Expr, ordine: tuple[str, ...]) -> pl.Expr:
    """Posizione nell'ordine dato; le voci non elencate finiscono in fondo."""
    rango = pl.lit(len(ordine), dtype=pl.Int32)
    for posizione, voce in reversed(list(enumerate(ordine))):
        rango = pl.when(valore == voce).then(pl.lit(posizione, dtype=pl.Int32)).otherwise(rango)
    return rango


def riepilogo_metriche(tabella: pl.DataFrame) -> pl.DataFrame:
    """Le metriche aggregate su tutti i mesi e tutte le scadenze, per modello.

    Il filtro su ``month == -1`` e ``lead_slot == -1`` seleziona le righe gia'
    aggregate dalla valutazione, invece di rifare una media di medie che pesarebbe
    male i mesi con meno casi.
    """
    return (
        tabella.filter((pl.col("month") == -1) & (pl.col("lead_slot") == -1))
        .with_columns(
            _rango(pl.col("variable"), ORDINE_VARIABILI).alias("_v"),
            _rango(pl.col("metric"), ORDINE_METRICHE).alias("_m"),
            _rango(pl.col("model"), ORDINE_MODELLI).alias("_o"),
        )
        .sort("_v", "variable", "_m", "metric", "_o", "model")
        .select("model", "variable", "metric", "value", "n_values")
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


def guadagno_su_persistenza(
    tabella: pl.DataFrame, variabile: str, metrica: str
) -> pl.DataFrame:
    """Quanto il modello guadagna, scadenza per scadenza, sul non fare nulla.

    L'errore assoluto non dice se il modello serve: a ventiquattro ore 2,40 gradi si
    ottengono ripetendo ieri alla stessa ora, e un modello che segna 2,27 sta guadagnando
    il cinque per cento, non facendo previsioni. Il confronto e' contro la **migliore**
    persistenza a quella scadenza, non contro la media delle persistenze, perche' un
    riferimento facile da battere renderebbe il guadagno adulatorio.
    """
    valori = per_scadenza(tabella, variabile, metrica)
    if valori.is_empty():
        return valori.select(
            pl.col("lead_slot"),
            pl.lit(None, dtype=pl.Float64).alias("modello"),
            pl.lit(None, dtype=pl.Float64).alias("riferimento"),
            pl.lit(None, dtype=pl.Float64).alias("guadagno_percento"),
        )

    modello = valori.filter(pl.col("model") == "dwf").select(
        "lead_slot", pl.col("value").alias("modello")
    )
    riferimento = (
        valori.filter(pl.col("model") != "dwf")
        .group_by("lead_slot")
        .agg(pl.col("value").min().alias("riferimento"))
    )
    return (
        modello.join(riferimento, on="lead_slot", how="left")
        .with_columns(
            (
                100.0
                * (pl.col("riferimento") - pl.col("modello"))
                / pl.col("riferimento")
            ).alias("guadagno_percento")
        )
        .sort("lead_slot")
    )


def accuratezze_ingannevoli(tabella: pl.DataFrame) -> pl.DataFrame:
    """Casi in cui rispondere sempre "no" batterebbe l'accuratezza dichiarata.

    Per una variabile rara l'accuratezza premia il silenzio: con frequenza di base
    0,09 un modello muto ottiene 0,91. Confrontarla con ``1 - base_rate`` e' l'unico
    modo per accorgersene leggendo la tabella.
    """
    aggregate = tabella.filter((pl.col("month") == -1) & (pl.col("lead_slot") == -1))
    accuratezza = aggregate.filter(pl.col("metric") == "accuracy").select(
        "model", "variable", pl.col("value").alias("accuratezza")
    )
    frequenza = (
        aggregate.filter(pl.col("metric") == "base_rate")
        .select("variable", pl.col("value").alias("frequenza_di_base"))
        .unique(subset=["variable"])
    )
    if not accuratezza.height or not frequenza.height:
        return accuratezza.head(0).with_columns(
            pl.lit(0.0).alias("frequenza_di_base"), pl.lit(0.0).alias("sempre_no")
        )
    return (
        accuratezza.join(frequenza, on="variable", how="inner")
        .with_columns((1.0 - pl.col("frequenza_di_base")).alias("sempre_no"))
        .filter(pl.col("accuratezza") < pl.col("sempre_no"))
        .sort("variable", "model")
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


# --------------------------------------------------------------------------- #
# Ispezione di ingressi e uscite
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CanaleIspezionato:
    """Un canale di ingresso riportato, dove ha senso, in unita' fisiche."""

    indice: int
    nome: str
    gruppo: str
    variabile: str
    ritardo: int
    campo: np.ndarray
    unita: str
    avvertenza: str | None = None


def catalogo_ingressi(config: Config) -> pl.DataFrame:
    """L'elenco dei canali di ingresso con provenienza e trattamento.

    Non richiede ne' modello ne' dati: descrive che cosa la configurazione corrente
    costruirebbe, quindi resta consultabile prima di aver addestrato qualsiasi cosa.
    """
    from dwf.data.features import InputLayout

    layout = InputLayout.from_config(config)
    return pl.DataFrame(layout.describe())


def ispeziona_ingresso(
    config: Config,
    fold: int,
    *,
    split: str = "test",
    posizione: int = 0,
    canale: int = 0,
) -> CanaleIspezionato:
    """Estrae un canale della finestra di ingresso e lo riporta in unita' leggibili.

    Il campo mostrato e' quello che la rete riceve davvero, ricostruito con la stessa
    pipeline: non e' una rilettura indipendente dello store, quindi se la costruzione
    delle feature avesse un difetto, comparirebbe anche qui. E' voluto: serve a vedere
    l'ingresso del modello, non un'idea di come dovrebbe essere.

    I canali di tendenza sono differenze fra istanti gia' normalizzati. Moltiplicarle
    per la deviazione standard le riporta alla scala della variabile, ma solo per le
    variabili a trasformazione identita' il risultato e' una differenza fisica: per la
    precipitazione, trasformata con log1p, la differenza resta nello spazio trasformato
    e viene dichiarata come tale.
    """
    from dwf.data.dataset import (
        KEY_FEATURES,
        WeatherWindowDataset,
        build_reader,
        sample_starts,
    )
    from dwf.data.features import GROUP_TENDENCY
    from dwf.train import load_checkpoint

    _, stats, input_layout, _ = load_checkpoint(config, fold)
    inizi = sample_starts(config, fold, split)
    if not inizi:
        raise DashboardError(f"Nessuna finestra nel blocco {split!r} del fold {fold}")
    posizione = max(0, min(posizione, len(inizi) - 1))
    canale = max(0, min(canale, input_layout.n_channels - 1))

    dataset = WeatherWindowDataset(
        config, input_layout, stats, inizi, build_reader(config, input_layout),
        crop_size=None, crops_per_window=1,
    )
    campione = dataset[posizione]
    valori = campione[KEY_FEATURES].numpy()[canale]

    descrizione = input_layout.describe()[canale]
    variabile = descrizione["source_variable"]
    avvertenza: str | None = None

    if not descrizione["normalized"]:
        campo, unita = valori, "adimensionale"
    elif descrizione["group"] == GROUP_TENDENCY:
        # Somma della media: sarebbe sbagliata su una differenza, si applica la sola scala.
        campo = valori * stats.std[variabile]
        trasformazione = stats.transform.get(variabile, "identity")
        unita = f"variazione di {variabile}"
        if trasformazione != "identity":
            avvertenza = (
                f"La variabile usa la trasformazione {trasformazione}: la differenza "
                "mostrata resta nello spazio trasformato, non in unita' fisiche."
            )
    else:
        campo, unita = stats.denormalize(variabile, valori), _unita_di(variabile)

    return CanaleIspezionato(
        indice=canale,
        nome=descrizione["name"],
        gruppo=descrizione["group"],
        variabile=variabile,
        ritardo=int(descrizione["lag"]),
        campo=np.asarray(campo, dtype=np.float32),
        unita=unita,
        avvertenza=avvertenza,
    )


def _unita_di(variabile: str) -> str:
    """Unita' della variabile **come la usa la pipeline**, non come sta nel GRIB.

    La specifica dichiara l'unita' di origine: la temperatura vi risulta in kelvin
    anche se lo store applica uno scostamento e tutto il progetto lavora in gradi
    Celsius. Riportare l'etichetta grezza qui vorrebbe dire scrivere "K" accanto a
    numeri intorno a zero.
    """
    from dwf.data.features import store_offset_of
    from dwf.variables import BY_SHORT_NAME

    spec = BY_SHORT_NAME.get(variabile)
    if spec is None:
        return "unita' della variabile"
    if spec.units == "K" and store_offset_of(variabile) != 0.0:
        return "degC"
    return spec.units


def ispeziona_uscite(
    config: Config,
    fold: int,
    *,
    split: str = "test",
    posizione: int = 0,
    variabile: str = "t2m",
) -> pl.DataFrame:
    """Previsto, osservato e differenza per **tutte** le scadenze di una finestra.

    Una scadenza sola nasconde il difetto piu' comune di un modello ancorato: errore
    piccolo alla prima scadenza e crescente sulle successive. La tabella le mostra tutte
    e nove, in modo che la crescita sia visibile invece che da dedurre.
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
        media = output_layout.select(uscita, variabile, "mean")[0].numpy()

    osservato_tutto = campione[f"target_{variabile}"].numpy()
    inizio = int(campione["start_slot"])
    righe = []
    for scadenza in range(media.shape[0]):
        previsto = stats.denormalize(variabile, media[scadenza])
        osservato = stats.denormalize(variabile, osservato_tutto[scadenza])
        differenza = previsto - osservato
        istante = _istante_di(config, inizio, scadenza)
        righe.append(
            {
                "scadenza": scadenza,
                "istante": istante.strftime("%Y-%m-%d %H UTC") if istante else "",
                "previsto_medio": float(np.mean(previsto)),
                "osservato_medio": float(np.mean(osservato)),
                "errore_medio": float(np.mean(differenza)),
                "errore_assoluto": float(np.mean(np.abs(differenza))),
                "radice_errore_quadratico": float(np.sqrt(np.mean(differenza**2))),
            }
        )
    return pl.DataFrame(righe)


# --------------------------------------------------------------------------- #
# Etichette leggibili
# --------------------------------------------------------------------------- #

# Le tabelle prodotte dalla valutazione usano i nomi tecnici delle colonne, che sono
# quelli giusti su disco e i peggiori possibili a schermo: `rmse_celsius` non dice
# l'unita' e `sf` non dice che si tratta di neve. Le corrispondenze stanno qui perche'
# l'interfaccia possa tradurre senza inventare.

NOMI_MODELLI: dict[str, str] = {
    "dwf": "DWF (la rete)",
    "persistence_diurnal": "Persistenza diurna",
    "persistence": "Persistenza ingenua",
}

NOMI_VARIABILI: dict[str, str] = {
    "t2m": "Temperatura a 2 m",
    "tp": "Precipitazione",
    "sf": "Neve",
}

# La stessa colonna `value` contiene gradi, probabilita' e frazioni: senza l'unita'
# accanto al nome i numeri non sono confrontabili a vista.
NOMI_METRICHE: dict[str, str] = {
    "rmse_celsius": "Errore quadratico medio (degC)",
    "mae_celsius": "Errore assoluto medio (degC)",
    "brier": "Punteggio di Brier (0 = perfetto)",
    "brier_skill_score": "Guadagno di Brier sul riferimento (1 = perfetto)",
    "calibration_error": "Errore di calibrazione (0 = perfetto)",
    "f1": "F1 (0-1, piu' alto e' meglio)",
    "precision": "Precisione (0-1)",
    "recall": "Richiamo (0-1)",
    "accuracy": "Accuratezza (0-1, da leggere con la frequenza)",
    "base_rate": "Frequenza dell'evento (0-1)",
    "decision_threshold": "Soglia di decisione",
}

NOMI_BLOCCHI: dict[str, str] = {
    "train": "Addestramento",
    "val": "Validazione",
    "test": "Test",
}

NOMI_GRUPPI_CANALI: dict[str, str] = {
    "state": "Stato della variabile",
    "tendency": "Tendenza fra due istanti",
    "static": "Campo statico",
    "time": "Coordinata temporale",
    "wind_speed": "Intensita' del vento",
    "latitude": "Latitudine",
}


def etichetta_scadenza(config: Config, scadenza: int) -> str:
    """Scadenza scritta come giorno e ora UTC invece che come numero di slot.

    Gli slot non sono equidistanti (06, 12, 18 UTC), quindi tradurli in ore di anticipo
    darebbe un passo che cambia dentro la giornata: il giorno e l'ora sono esatti.
    """
    ore = config.time.slot_hours
    if not ore:
        return f"slot {scadenza}"
    return f"giorno {scadenza // len(ore) + 1}, ore {ore[scadenza % len(ore)]:02d} UTC"


def riepilogo_leggibile(tabella: pl.DataFrame) -> pl.DataFrame:
    """Il riepilogo delle metriche con intestazioni in italiano e unita' esplicite."""
    return riepilogo_metriche(tabella).select(
        pl.col("model").replace(NOMI_MODELLI).alias("Modello"),
        pl.col("variable").replace(NOMI_VARIABILI).alias("Grandezza"),
        pl.col("metric").replace(NOMI_METRICHE).alias("Metrica"),
        pl.col("value").round(4).alias("Valore"),
        pl.col("n_values").alias("Punti confrontati"),
    )


def guadagno_leggibile(
    config: Config, tabella: pl.DataFrame, variabile: str, metrica: str
) -> pl.DataFrame:
    """Il guadagno sulla persistenza con la scadenza scritta in chiaro."""
    guadagno = guadagno_su_persistenza(tabella, variabile, metrica)
    if guadagno.is_empty():
        return pl.DataFrame(
            schema={
                "Scadenza": pl.Utf8,
                "Modello": pl.Float64,
                "Migliore persistenza": pl.Float64,
                "Guadagno (%)": pl.Float64,
            }
        )
    etichette = [
        etichetta_scadenza(config, int(valore)) for valore in guadagno["lead_slot"]
    ]
    return guadagno.select(
        pl.Series("Scadenza", etichette),
        pl.col("modello").round(3).alias("Modello"),
        pl.col("riferimento").round(3).alias("Migliore persistenza"),
        pl.col("guadagno_percento").round(1).alias("Guadagno (%)"),
    )


# --------------------------------------------------------------------------- #
# Stato d'insieme
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StatoProgetto:
    """I pochi numeri che rispondono a "come sta il progetto adesso"."""

    slot_presenti: int
    slot_catalogati: int
    ultimo_dato: datetime | None
    checkpoint: bool
    problemi: tuple[str, ...]
    scadenza: int
    errore_modello: float | None
    errore_riferimento: float | None
    guadagno_percento: float | None

    @property
    def frazione_ingerita(self) -> float:
        if not self.slot_catalogati:
            return 0.0
        return self.slot_presenti / self.slot_catalogati


def stato_progetto(
    config: Config, fold: int = 0, *, split: str = "test"
) -> StatoProgetto:
    """Stato reale del progetto: dati ingeriti, checkpoint e guadagno a ventiquattro ore.

    Il numero che conta e' il guadagno sulla **migliore** persistenza, non l'errore
    assoluto: a ventiquattro ore l'errore che si vede si ottiene gia' ripetendo
    l'osservazione di ieri alla stessa ora, quindi da solo non dice se il modello serve.

    La scadenza scelta e' l'ultimo slot del primo giorno di previsione, cioe' le
    ventiquattro ore piene rispetto all'inizio della finestra bersaglio.
    """
    slots = _leggi(SLOTS, config.tables_dir)
    presenti = 0
    catalogati = 0
    ultimo: datetime | None = None
    if slots is not None and slots.height:
        catalogati = slots.height
        utilizzabili = slots.filter(pl.col("usable"))
        presenti = utilizzabili.height
        if presenti:
            ultimo = utilizzabili["valid_time"].max()

    checkpoint = (config.fold_dir(fold) / "weights.npz").exists()
    problemi = tuple(coerenza_artefatti(config, fold)) if checkpoint else ()

    scadenza = max(len(config.time.slot_hours) - 1, 0)
    errore_modello: float | None = None
    errore_riferimento: float | None = None
    guadagno: float | None = None

    tabella = metriche(config, fold, split)
    if tabella is not None:
        righe = guadagno_su_persistenza(tabella, "t2m", "rmse_celsius").filter(
            pl.col("lead_slot") == scadenza
        )
        if righe.height:
            riga = righe.row(0, named=True)
            errore_modello = riga["modello"]
            errore_riferimento = riga["riferimento"]
            guadagno = riga["guadagno_percento"]

    return StatoProgetto(
        slot_presenti=presenti,
        slot_catalogati=catalogati,
        ultimo_dato=ultimo,
        checkpoint=checkpoint,
        problemi=problemi,
        scadenza=scadenza,
        errore_modello=errore_modello,
        errore_riferimento=errore_riferimento,
        guadagno_percento=guadagno,
    )


__all__ = [
    "NOMI_BLOCCHI",
    "NOMI_GRUPPI_CANALI",
    "NOMI_METRICHE",
    "NOMI_MODELLI",
    "NOMI_VARIABILI",
    "TECNOLOGIE",
    "CanaleIspezionato",
    "Confronto",
    "DashboardError",
    "Riquadro",
    "StatoProgetto",
    "accuratezze_ingannevoli",
    "catalogo_ingressi",
    "confronto_visivo",
    "copertura_mensile",
    "curva_apprendimento",
    "etichetta_scadenza",
    "guadagno_leggibile",
    "guadagno_su_persistenza",
    "informazioni_modello",
    "ispeziona_ingresso",
    "ispeziona_uscite",
    "mappa_errori",
    "metriche",
    "panoramica",
    "per_scadenza",
    "riepilogo_leggibile",
    "riepilogo_metriche",
    "risorse",
    "spazio_dati",
    "stato_progetto",
    "struttura_fold",
]
