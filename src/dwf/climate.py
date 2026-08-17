"""Andamenti climatici ricavati dallo store.

Il progetto ha una serie oraria su un dominio ampio, quindi il materiale per parlare di
clima c'e'. Quello che spesso manca, e che qui viene calcolato esplicitamente, e' il
diritto di trarne conclusioni: una pendenza si puo' sempre adattare a una serie, anche
di tre punti, e il risultato ha un'aria autorevole che non si merita.

Per questo ogni funzione che produce una tendenza produce anche il numero di anni su cui
e' calcolata e l'intervallo di confidenza, e ``giudizio_sulla_serie`` traduce il tutto in
una frase che dice se la conclusione regge.

Due scelte che cambiano i numeri e vanno dichiarate:

- **la media spaziale e' pesata per l'area.** Su una griglia in latitudine e longitudine
  le celle vicine al polo coprono molto meno terreno di quelle equatoriali; una media
  aritmetica darebbe al nord del dominio un peso che non ha. Il peso e' il coseno della
  latitudine.
- **il giorno ha tre osservazioni, non ventiquattro.** Gli indici che nella letteratura
  climatica si basano su minimo e massimo giornalieri qui si basano sugli slot delle 06,
  12 e 18 UTC. Sono approssimazioni, e vengono chiamate tali.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from dwf.config import Config
from dwf.tables import SLOTS, read_table
from dwf.weighting import latitude_area_weight


class ClimateError(RuntimeError):
    """Errore nel calcolo degli andamenti climatici."""


# Soglie degli indici, in gradi Celsius e in millimetri.
SOGLIA_GELO = 0.0
SOGLIA_NOTTE_TROPICALE = 20.0
SOGLIA_PIOGGIA_INTENSA = 20.0
SOGLIA_NEVE = 0.001


@dataclass(frozen=True)
class Tendenza:
    """Pendenza di una retta adattata a una serie annuale, con la sua incertezza."""

    pendenza: float
    """Variazione per anno, nelle unita' della grandezza."""

    incertezza: float
    """Errore standard della pendenza."""

    n_anni: int
    intercetta: float = 0.0

    @property
    def intervallo(self) -> tuple[float, float]:
        """Intervallo di confidenza al 95 per cento, approssimato a due errori standard."""
        margine = 1.96 * self.incertezza
        return (self.pendenza - margine, self.pendenza + margine)

    @property
    def significativa(self) -> bool:
        """Vera se l'intervallo non contiene lo zero **e** gli anni sono abbastanza.

        La seconda condizione non e' pignoleria statistica: con pochi anni la
        variabilita' naturale del tempo atmosferico produce pendenze grandi e strette
        per puro caso, e il test da solo le dichiarerebbe reali.
        """
        basso, alto = self.intervallo
        return self.n_anni >= ANNI_MINIMI_PER_TENDENZA and (basso > 0 or alto < 0)


# Sotto questa soglia una pendenza annuale non viene presentata come tendenza. Trenta
# anni e' la normale climatologica dell'Organizzazione meteorologica mondiale; qui la
# soglia e' molto piu' bassa e serve solo a distinguere "qualche indicazione" da "una
# retta tirata su due punti".
ANNI_MINIMI_PER_TENDENZA = 10


def adatta_tendenza(anni: np.ndarray, valori: np.ndarray) -> Tendenza:
    """Minimi quadrati su una serie annuale, con l'errore standard della pendenza."""
    anni = np.asarray(anni, dtype=float)
    valori = np.asarray(valori, dtype=float)
    validi = np.isfinite(valori)
    anni, valori = anni[validi], valori[validi]
    n = anni.size
    if n < 2:
        return Tendenza(pendenza=float("nan"), incertezza=float("nan"), n_anni=int(n))

    centrati = anni - anni.mean()
    varianza = float((centrati**2).sum())
    if varianza == 0:
        return Tendenza(pendenza=float("nan"), incertezza=float("nan"), n_anni=int(n))

    pendenza = float((centrati * (valori - valori.mean())).sum() / varianza)
    intercetta = float(valori.mean() - pendenza * anni.mean())

    if n == 2:
        # Due punti definiscono una retta esatta: l'errore standard non e' zero, e'
        # indefinito. Dichiararlo nullo farebbe apparire significativa qualunque cosa.
        return Tendenza(pendenza, float("nan"), int(n), intercetta)

    residui = valori - (intercetta + pendenza * anni)
    varianza_residua = float((residui**2).sum() / (n - 2))
    return Tendenza(pendenza, float(np.sqrt(varianza_residua / varianza)), int(n), intercetta)


# --------------------------------------------------------------------------- #
# Copertura
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Copertura:
    """Che cosa contiene davvero lo store, in termini utili al giudizio climatico."""

    anni: tuple[int, ...]
    anni_completi: tuple[int, ...]
    mesi_per_anno: dict[int, int]
    primo: str
    ultimo: str
    slot: int

    @property
    def n_anni_completi(self) -> int:
        return len(self.anni_completi)


def copertura(config: Config) -> Copertura | None:
    """Anni presenti e quali di essi hanno tutti i dodici mesi."""
    percorso = config.tables_dir / SLOTS.filename
    if not percorso.exists():
        return None
    tabella = read_table(SLOTS, config.tables_dir).filter(pl.col("usable"))
    if not tabella.height:
        return None

    per_anno = (
        tabella.group_by("year")
        .agg(pl.col("month").n_unique().alias("mesi"))
        .sort("year")
    )
    mesi = {int(r["year"]): int(r["mesi"]) for r in per_anno.iter_rows(named=True)}
    return Copertura(
        anni=tuple(sorted(mesi)),
        anni_completi=tuple(sorted(a for a, m in mesi.items() if m == 12)),
        mesi_per_anno=mesi,
        primo=str(tabella["valid_time"].min()),
        ultimo=str(tabella["valid_time"].max()),
        slot=int(tabella.height),
    )


def giudizio_sulla_serie(copertura_: Copertura | None) -> str:
    """Che cosa si puo' onestamente dire con i dati presenti.

    Il posto in cui si dice al lettore di non fidarsi troppo non e' una nota a pie' di
    pagina: e' la prima riga della pagina.
    """
    if copertura_ is None:
        return (
            "Nessun dato ingerito. Il catalogo degli slot non esiste ancora: "
            "eseguire prima scaricamento e ingestione."
        )

    completi = copertura_.n_anni_completi
    if completi >= ANNI_MINIMI_PER_TENDENZA:
        return (
            f"{completi} anni completi: le tendenze sono calcolabili, pur restando "
            "molto piu' brevi della normale climatologica di trent'anni."
        )
    if completi >= 2:
        return (
            f"Solo {completi} anni completi. Quello che si vede sono **differenze fra "
            "annate**, non una tendenza climatica: la variabilita' naturale del tempo "
            "atmosferico da un anno all'altro e' molto maggiore del segnale climatico, "
            "e su questa lunghezza lo copre interamente."
        )
    return (
        f"Meno di due anni completi (mesi per anno: {copertura_.mesi_per_anno}). "
        "Qualunque pendenza calcolata qui descriverebbe il caso, non il clima. "
        "La pagina mostra il ciclo stagionale e il confronto fra i mesi che si "
        "ripetono, che sono le sole letture che i dati sostengono."
    )


# --------------------------------------------------------------------------- #
# Serie aggregate
# --------------------------------------------------------------------------- #


def _pesi_area(config: Config) -> np.ndarray:
    """Peso di area sull'intera griglia, gia' replicato sulle longitudini."""
    import zarr

    gruppo = zarr.open_group(str(config.zarr_path), mode="r")
    latitudini = np.asarray(gruppo["latitude"][:], dtype=np.float64)
    larghezza = int(gruppo["longitude"].shape[0])
    return latitude_area_weight(latitudini, larghezza)


def _percorso_cache(config: Config, nome: str) -> Path:
    cartella = config.artifacts_dir / "clima"
    cartella.mkdir(parents=True, exist_ok=True)
    return cartella / f"{nome}.npz"


def medie_per_slot(
    config: Config, variabile: str = "t2m", *, usa_cache: bool = True
) -> pl.DataFrame | None:
    """Media spaziale pesata per area, uno scalare per ogni slot presente.

    Attraversare l'intero store una volta e ridurlo a una serie di scalari e' cio' che
    rende praticabile tutto il resto: le aggregazioni per mese, per stagione e per anno
    partono poi da qualche migliaio di numeri invece che da centinaia di megabyte.
    """
    import zarr

    percorso = config.tables_dir / SLOTS.filename
    if not percorso.exists() or not config.zarr_path.exists():
        return None

    catalogo = read_table(SLOTS, config.tables_dir).filter(pl.col("usable")).sort("slot_index")
    if not catalogo.height:
        return None

    indici = catalogo.get_column("slot_index").to_numpy()
    cache = _percorso_cache(config, f"medie_{variabile}_{indici.size}_{int(indici[-1])}")
    if usa_cache and cache.exists():
        with np.load(cache, allow_pickle=False) as archivio:
            medie = archivio["medie"]
    else:
        gruppo = zarr.open_group(str(config.zarr_path), mode="r")
        if variabile not in gruppo:
            raise ClimateError(f"Variabile assente dallo store: {variabile!r}")
        campo = gruppo[variabile]
        pesi = _pesi_area(config)
        totale_pesi = float(pesi.sum())

        medie = np.empty(indici.size, dtype=np.float64)
        # Si legge a blocchi per non tenere in memoria l'intero periodo: il tensore
        # completo supererebbe il mezzo gigabyte per una sola variabile.
        passo = 64
        for inizio in range(0, indici.size, passo):
            selezione = indici[inizio : inizio + passo]
            blocco = np.asarray(campo[selezione.min() : selezione.max() + 1])
            locali = selezione - selezione.min()
            pesato = (blocco[locali] * pesi[None, :, :]).sum(axis=(1, 2)) / totale_pesi
            medie[inizio : inizio + selezione.size] = pesato
        if usa_cache:
            np.savez_compressed(cache, medie=medie)

    return catalogo.select("slot_index", "valid_time", "year", "month", "day", "hour").with_columns(
        pl.Series("valore", medie)
    )


def _in_unita_di_lavoro(serie: pl.DataFrame, variabile: str) -> pl.DataFrame:
    """Applica lo scarto che il progetto usa gia' altrove per passare a gradi Celsius.

    Lo store conserva le unita' native ERA5, cioe' kelvin, e la conversione e' dichiarata
    una volta sola nel registro delle variabili. Riscoprirla qui con una soglia sui
    valori funzionerebbe finche' nessuno cambia lo store, che e' il tipo di ipotesi che
    poi si rompe in silenzio.
    """
    from dwf.data.features import store_offset_of

    scarto = store_offset_of(variabile)
    return serie if scarto == 0.0 else serie.with_columns(pl.col("valore") + scarto)


def medie_mensili(config: Config, variabile: str = "t2m") -> pl.DataFrame | None:
    """Media per anno e mese, con il numero di slot su cui e' calcolata."""
    serie = medie_per_slot(config, variabile)
    if serie is None:
        return None
    serie = _in_unita_di_lavoro(serie, variabile)
    return (
        serie.group_by("year", "month")
        .agg(pl.col("valore").mean().alias("media"), pl.len().alias("slot"))
        .sort("year", "month")
        .with_columns(
            (pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.zfill(2))
            .alias("periodo")
        )
    )


def confronto_interannuale(config: Config, variabile: str = "t2m") -> pl.DataFrame | None:
    """Lo stesso mese in anni diversi, che e' il confronto onesto su serie corte.

    Confrontare gennaio con luglio non dice nulla sul clima; confrontare il gennaio di
    anni diversi dice qualcosa, perche' toglie di mezzo il ciclo stagionale, che e' di
    gran lunga il segnale piu' forte.
    """
    mensili = medie_mensili(config, variabile)
    if mensili is None:
        return None
    ripetuti = (
        mensili.group_by("month")
        .agg(pl.col("year").n_unique().alias("anni"))
        .filter(pl.col("anni") > 1)
        .get_column("month")
        .to_list()
    )
    if not ripetuti:
        return None
    return mensili.filter(pl.col("month").is_in(ripetuti)).sort("month", "year")


def ciclo_stagionale(config: Config, variabile: str = "t2m") -> pl.DataFrame | None:
    """Media per mese dell'anno, mediando fra gli anni disponibili."""
    mensili = medie_mensili(config, variabile)
    if mensili is None:
        return None
    return (
        mensili.group_by("month")
        .agg(
            pl.col("media").mean().alias("media"),
            pl.col("media").min().alias("minimo"),
            pl.col("media").max().alias("massimo"),
            pl.col("year").n_unique().alias("anni"),
        )
        .sort("month")
    )


def medie_annuali(config: Config, variabile: str = "t2m") -> pl.DataFrame | None:
    """Media per anno, calcolata solo sugli anni con tutti i dodici mesi.

    Un anno incompleto non e' confrontabile con uno completo: se mancano i mesi freddi
    la sua media risulta piu' alta, e la differenza verrebbe letta come riscaldamento.
    """
    mensili = medie_mensili(config, variabile)
    if mensili is None:
        return None
    completi = (
        mensili.group_by("year")
        .agg(pl.len().alias("mesi"))
        .filter(pl.col("mesi") == 12)
        .get_column("year")
        .to_list()
    )
    if not completi:
        return None
    return (
        mensili.filter(pl.col("year").is_in(completi))
        .group_by("year")
        .agg(pl.col("media").mean().alias("media"), pl.len().alias("mesi"))
        .sort("year")
    )


def tendenza_annuale(config: Config, variabile: str = "t2m") -> Tendenza | None:
    """Pendenza della media annuale, se ci sono almeno due anni completi."""
    annuali = medie_annuali(config, variabile)
    if annuali is None or annuali.height < 2:
        return None
    return adatta_tendenza(
        annuali.get_column("year").to_numpy(), annuali.get_column("media").to_numpy()
    )


def mappa_differenza_mensile(
    config: Config, mese: int, anno_a: int, anno_b: int, variabile: str = "t2m"
) -> np.ndarray | None:
    """Differenza per cella fra lo stesso mese di due anni: ``anno_b`` meno ``anno_a``.

    Con una serie corta questa e' la mappa che i dati sostengono. Una mappa di tendenza
    per cella richiederebbe molti anni: adattarla a due punti produrrebbe una figura
    dettagliata e priva di significato, che e' il modo piu' efficace di ingannare chi
    guarda.
    """
    import zarr

    if not config.zarr_path.exists():
        return None
    catalogo = read_table(SLOTS, config.tables_dir).filter(pl.col("usable"))
    gruppo = zarr.open_group(str(config.zarr_path), mode="r")
    if variabile not in gruppo:
        raise ClimateError(f"Variabile assente dallo store: {variabile!r}")
    campo = gruppo[variabile]

    medie: dict[int, np.ndarray] = {}
    for anno in (anno_a, anno_b):
        indici = (
            catalogo.filter((pl.col("year") == anno) & (pl.col("month") == mese))
            .get_column("slot_index")
            .to_numpy()
        )
        if indici.size == 0:
            return None
        somma = np.zeros(campo.shape[1:], dtype=np.float64)
        for inizio in range(0, indici.size, 32):
            selezione = indici[inizio : inizio + 32]
            blocco = np.asarray(campo[selezione.min() : selezione.max() + 1])
            somma += blocco[selezione - selezione.min()].sum(axis=0)
        medie[anno] = somma / indici.size

    return (medie[anno_b] - medie[anno_a]).astype(np.float32)


def mesi_confrontabili(config: Config) -> list[tuple[int, list[int]]]:
    """Per ogni mese dell'anno, quali anni sono disponibili con dati sufficienti."""
    percorso = config.tables_dir / SLOTS.filename
    if not percorso.exists():
        return []
    catalogo = read_table(SLOTS, config.tables_dir).filter(pl.col("usable"))
    if not catalogo.height:
        return []
    conteggi = (
        catalogo.group_by("month", "year")
        .agg(pl.len().alias("slot"))
        # Meno di venti slot in un mese vuol dire meno di una settimana di dati: una
        # media calcolata li' non rappresenta il mese.
        .filter(pl.col("slot") >= 20)
        .sort("month", "year")
    )
    risultato: list[tuple[int, list[int]]] = []
    for mese in sorted(conteggi.get_column("month").unique().to_list()):
        anni = (
            conteggi.filter(pl.col("month") == mese).get_column("year").to_list()
        )
        if len(anni) > 1:
            risultato.append((int(mese), [int(a) for a in anni]))
    return risultato


# --------------------------------------------------------------------------- #
# Indici di estremi
# --------------------------------------------------------------------------- #


def indici_estremi(config: Config) -> pl.DataFrame | None:
    """Conteggi annuali di giornate caratteristiche, sulla cella di Vigo di Cadore.

    Gli indici classici usano minimo e massimo giornalieri. Qui il giorno ha tre
    osservazioni, quindi il gelo si conta sullo slot delle 06 UTC, il piu' vicino al
    minimo mattutino alle nostre longitudini, e il caldo su quello delle 12. Sono
    approssimazioni dichiarate, non definizioni standard.
    """
    import zarr

    if not config.zarr_path.exists():
        return None
    catalogo = read_table(SLOTS, config.tables_dir).filter(pl.col("usable")).sort("slot_index")
    if not catalogo.height:
        return None

    from dwf.weighting import VIGO_LATITUDE, VIGO_LONGITUDE

    gruppo = zarr.open_group(str(config.zarr_path), mode="r")
    latitudini = np.asarray(gruppo["latitude"][:])
    longitudini = np.asarray(gruppo["longitude"][:])
    riga = int(np.abs(latitudini - VIGO_LATITUDE).argmin())
    colonna = int(np.abs(longitudini - VIGO_LONGITUDE).argmin())

    indici = catalogo.get_column("slot_index").to_numpy()
    from dwf.data.features import store_offset_of

    temperatura = np.asarray(gruppo["t2m"][:, riga, colonna])[indici] + store_offset_of("t2m")
    pioggia = np.asarray(gruppo["tp"][:, riga, colonna])[indici] * 1000.0
    neve = np.asarray(gruppo["sf"][:, riga, colonna])[indici] * 1000.0

    dati = catalogo.select("year", "hour").with_columns(
        pl.Series("temperatura", temperatura),
        pl.Series("pioggia", pioggia),
        pl.Series("neve", neve),
    )

    return (
        dati.group_by("year")
        .agg(
            ((pl.col("temperatura") < SOGLIA_GELO) & (pl.col("hour") == 6))
            .sum()
            .alias("mattine_di_gelo"),
            ((pl.col("temperatura") > SOGLIA_NOTTE_TROPICALE) & (pl.col("hour") == 6))
            .sum()
            .alias("mattine_sopra_20"),
            (pl.col("pioggia") > SOGLIA_PIOGGIA_INTENSA).sum().alias("slot_pioggia_intensa"),
            (pl.col("neve") > SOGLIA_NEVE).sum().alias("slot_con_neve"),
            pl.col("temperatura").mean().alias("temperatura_media"),
            pl.len().alias("slot"),
        )
        .sort("year")
    )


__all__ = [
    "ANNI_MINIMI_PER_TENDENZA",
    "ClimateError",
    "Copertura",
    "Tendenza",
    "adatta_tendenza",
    "ciclo_stagionale",
    "confronto_interannuale",
    "copertura",
    "giudizio_sulla_serie",
    "indici_estremi",
    "mappa_differenza_mensile",
    "medie_annuali",
    "medie_mensili",
    "medie_per_slot",
    "mesi_confrontabili",
    "tendenza_annuale",
]
