"""Dashboard di ispezione del progetto.

Avvio:

    uv run streamlit run scripts/dashboard_app.py

Tutta la lettura degli artefatti sta in ``dwf.dashboard`` e ``dwf.climate``; qui c'e'
solo la presentazione. I calcoli costosi (esecuzione del modello su una finestra, mappa
degli errori) sono dietro una cache, altrimenti ogni interazione con un cursore
rieseguirebbe la rete sull'intera griglia.

Tre regole tengono insieme l'interfaccia:

* le pagine sono raggruppate per **area tematica**, non elencate in una lista piatta;
* ogni tabella arriva a schermo con intestazioni in italiano, unita' di misura e una
  didascalia che dice come si legge, oppure viene sostituita da un grafico;
* i grafici semplici si disegnano **tutti** con ``grafico_linee`` e ``grafico_barre``,
  cosi' che scala, griglia e caratteri non cambino da una pagina all'altra.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dwf import runs  # noqa: E402
from dwf.climate import (  # noqa: E402
    ANNI_MINIMI_PER_TENDENZA,
    ciclo_stagionale,
    confronto_interannuale,
    copertura,
    giudizio_sulla_serie,
    indici_estremi,
    mappa_differenza_mensile,
    medie_mensili,
    mesi_confrontabili,
    tendenza_annuale,
)
from dwf.config import Config  # noqa: E402
from dwf.dashboard import (  # noqa: E402
    NOMI_BLOCCHI,
    NOMI_GRUPPI_CANALI,
    NOMI_METRICHE,
    NOMI_MODELLI,
    NOMI_VARIABILI,
    TECNOLOGIE,
    accuratezze_ingannevoli,
    affidabilita,
    calibrazione,
    catalogo_ingressi,
    coerenza_artefatti,
    confronto_visivo,
    copertura_mensile,
    curva_apprendimento,
    effetto_calibrazione,
    etichetta_scadenza,
    guadagno_leggibile,
    informazioni_modello,
    ispeziona_ingresso,
    ispeziona_uscite,
    mappa_errori,
    metriche,
    panoramica,
    per_scadenza,
    riepilogo_leggibile,
    risorse,
    scarto_di_affidabilita,
    spazio_dati,
    stato_progetto,
    struttura_fold,
)
from dwf.models import variants  # noqa: E402
from dwf.runs import RunError  # noqa: E402

st.set_page_config(
    page_title="Deep Weather Forecasting", page_icon="🌦️", layout="wide"
)

# Un solo stile per tutte le figure: la dashboard mescolava grafici nativi di Streamlit
# e figure matplotlib con impostazioni diverse, e due grafici della stessa grandezza non
# si potevano confrontare a vista.
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
    }
)

COLORE_MODELLO = "#1f6feb"
COLORE_RIFERIMENTO = "#d1495b"
BLOCCHI = ("test", "val", "train")


# --------------------------------------------------------------------------- #
# Cache dei calcoli costosi
# --------------------------------------------------------------------------- #


@st.cache_resource
def carica_config(percorso: str) -> Config:
    return Config.load(percorso)


@st.cache_data(show_spinner="Lettura dello stato del progetto...")
def stato_in_cache(percorso: str, fold: int):
    return stato_progetto(carica_config(percorso), fold)


@st.cache_data(show_spinner="Esecuzione del modello sulla finestra...")
def confronto_in_cache(percorso: str, fold: int, posizione: int, scadenza: int):
    config = carica_config(percorso)
    confronto = confronto_visivo(config, fold, posizione=posizione, scadenza=scadenza)
    # Si restituiscono array grezzi: la dataclass non e' serializzabile dalla cache.
    return (
        confronto.previsto,
        confronto.osservato,
        confronto.riferimento,
        confronto.istante,
        confronto.errore_assoluto_medio,
    )


@st.cache_data(show_spinner="Aggregazione dell'errore su piu' finestre...")
def errori_in_cache(percorso: str, fold: int, n_finestre: int, scadenza: int):
    config = carica_config(percorso)
    return mappa_errori(config, fold, n_finestre=n_finestre, scadenza=scadenza)


@st.cache_data(show_spinner="Lettura della copertura temporale...")
def copertura_in_cache(percorso: str):
    return copertura(carica_config(percorso))


@st.cache_data(show_spinner="Media dell'intero periodo sul dominio...")
def mensili_in_cache(percorso: str):
    return medie_mensili(carica_config(percorso))


@st.cache_data(show_spinner="Conteggio delle giornate caratteristiche...")
def estremi_in_cache(percorso: str):
    return indici_estremi(carica_config(percorso))


@st.cache_data(show_spinner="Ricostruzione della finestra di ingresso...")
def ingresso_in_cache(percorso: str, fold: int, split: str, posizione: int, canale: int):
    ispezione = ispeziona_ingresso(
        carica_config(percorso), fold, split=split, posizione=posizione, canale=canale
    )
    return (
        ispezione.nome, ispezione.gruppo, ispezione.variabile, ispezione.ritardo,
        ispezione.campo, ispezione.unita, ispezione.avvertenza,
    )


@st.cache_data(show_spinner="Esecuzione del modello su tutte le scadenze...")
def uscite_in_cache(percorso: str, fold: int, split: str, posizione: int, variabile: str):
    return ispeziona_uscite(
        carica_config(percorso), fold, split=split, posizione=posizione, variabile=variabile
    )


# --------------------------------------------------------------------------- #
# Mattoni di presentazione
# --------------------------------------------------------------------------- #


def riquadro(colonna, etichetta: str, valore, nota: str = "", aiuto: str | None = None):
    """Un numero con la sua nota sotto.

    La nota non passa per il parametro ``delta`` di ``st.metric``: quel parametro
    disegna sempre una freccia in su o in giu', e su un testo come l'estensione
    dell'area la freccia suggerisce un andamento che il dato non ha.
    """
    colonna.metric(etichetta, valore, help=aiuto, border=True)
    if nota:
        colonna.caption(nota)


def griglia_di_riquadri(voci: Sequence[tuple[str, str, str]], per_riga: int = 4) -> None:
    """Dispone etichetta, valore e nota su piu' righe di larghezza costante."""
    for inizio in range(0, len(voci), per_riga):
        blocco = voci[inizio : inizio + per_riga]
        colonne = st.columns(per_riga)
        for colonna, (etichetta, valore, nota) in zip(colonne, blocco, strict=False):
            riquadro(colonna, etichetta, valore, nota)


def tabella(dati, *, didascalia: str, altezza: int | None = None) -> None:
    """Ogni tabella della dashboard passa di qui, didascalia compresa.

    La didascalia e' obbligatoria per costruzione: una tabella senza istruzioni di
    lettura era il difetto piu' diffuso della versione precedente.
    """
    # `height=None` non e' un valore ammesso da Streamlit: il parametro va omesso.
    extra = {"height": altezza} if altezza is not None else {}
    st.dataframe(dati, width="stretch", hide_index=True, **extra)
    st.caption(didascalia)


def grafico_linee(
    serie: dict[str, tuple[Sequence, Sequence]],
    *,
    asse_x: str,
    asse_y: str,
    etichette_x: Sequence[str] | None = None,
    altezza: float = 3.2,
) -> None:
    """L'unico modo di disegnare una serie semplice in tutta la dashboard."""
    figura, asse = plt.subplots(figsize=(9.5, altezza), constrained_layout=True)
    for nome, (ascisse, ordinate) in serie.items():
        asse.plot(ascisse, ordinate, marker="o", markersize=4, linewidth=1.7, label=nome)
    if len(serie) > 1:
        asse.legend()
    asse.set_xlabel(asse_x)
    asse.set_ylabel(asse_y)
    if etichette_x is not None:
        prima = next(iter(serie.values()))[0]
        asse.set_xticks(list(prima))
        asse.set_xticklabels(etichette_x, rotation=30, ha="right")
    _rifinisci(asse)
    st.pyplot(figura)
    plt.close(figura)


def grafico_barre(
    etichette: Sequence[str],
    valori: Sequence[float],
    *,
    asse_x: str,
    asse_y: str,
    colore: str = COLORE_MODELLO,
    linea_zero: bool = False,
    altezza: float = 3.2,
) -> None:
    """L'unico modo di disegnare un confronto per categorie in tutta la dashboard."""
    figura, asse = plt.subplots(figsize=(9.5, altezza), constrained_layout=True)
    posizioni = np.arange(len(etichette))
    colori = (
        [colore if v >= 0 else COLORE_RIFERIMENTO for v in valori]
        if linea_zero
        else colore
    )
    asse.bar(posizioni, valori, color=colori, width=0.68)
    if linea_zero:
        asse.axhline(0.0, color="#333333", linewidth=0.8)
    asse.set_xticks(posizioni)
    passo = max(1, len(etichette) // 24)
    asse.set_xticklabels(
        [e if i % passo == 0 else "" for i, e in enumerate(etichette)],
        rotation=45,
        ha="right",
    )
    asse.set_xlabel(asse_x)
    asse.set_ylabel(asse_y)
    _rifinisci(asse)
    st.pyplot(figura)
    plt.close(figura)


def _rifinisci(asse) -> None:
    for lato in ("top", "right"):
        asse.spines[lato].set_visible(False)
    asse.grid(axis="y", alpha=0.3, linewidth=0.6)
    asse.set_axisbelow(True)


def mappa(asse, campo: np.ndarray, config: Config, *, titolo: str, cmap: str,
          vmin=None, vmax=None):
    """Disegna un campo con gli estremi geografici corretti.

    Le proporzioni non sono quelle di una proiezione vera: cartopy non e' fra le
    dipendenze, quindi la scala nord-sud e est-ovest divergono con la latitudine.
    """
    immagine = asse.imshow(
        campo,
        origin="upper",
        extent=[config.region.west, config.region.east,
                config.region.south, config.region.north],
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
    )
    asse.set_title(titolo)
    asse.set_xlabel("longitudine (gradi est)")
    asse.set_ylabel("latitudine (gradi nord)")
    asse.grid(False)
    return immagine


def disegna_figura(figura) -> None:
    st.pyplot(figura)
    plt.close(figura)


def intestazione(titolo: str, occhiello: str, spiegazione: str = "") -> None:
    """Titolo, riga di contesto e, se serve, il paragrafo che dice a che cosa serve.

    L'uniformita' e' voluta: nella versione precedente alcune pagine spiegavano molto e
    altre niente, e non si capiva se il silenzio significasse "ovvio" o "non finito".
    """
    st.title(titolo)
    st.caption(occhiello)
    if spiegazione:
        st.markdown(spiegazione)


def avviso_senza_metriche(fold_scelto: int, split: str) -> None:
    st.warning(
        f"Nessuna metrica per il fold {fold_scelto} sul blocco «{NOMI_BLOCCHI[split]}». "
        f"Eseguire `scripts/evaluate_model.py --fold {fold_scelto} --split {split}`."
    )


# --------------------------------------------------------------------------- #
# Area: Sintesi
# --------------------------------------------------------------------------- #


def pagina_stato() -> None:
    intestazione(
        "Stato del progetto",
        "Previsione a 3 giorni (06, 12, 18 UTC) sull'area euro-atlantica, "
        "rete convoluzionale scritta da zero e addestrata su CPU.",
    )

    stato = stato_in_cache(percorso_config, int(fold))

    colonne = st.columns(4)
    riquadro(
        colonne[0],
        "Slot ingeriti",
        f"{stato.frazione_ingerita * 100:.0f} %",
        f"{stato.slot_presenti:,} presenti su {stato.slot_catalogati:,} attesi dalla "
        "configurazione",
        aiuto="Uno slot e' un'osservazione ERA5 su tutta la griglia, alle 06, 12 o 18 UTC.",
    )

    if stato.guadagno_percento is None:
        riquadro(
            colonne[1],
            "Guadagno a 24 ore",
            "ignoto",
            "serve `scripts/evaluate_model.py` sul blocco di test",
        )
    else:
        colonne[1].metric(
            "Guadagno a 24 ore",
            f"{stato.guadagno_percento:+.1f} %",
            # Su una colonna di quattro il testo del delta viene troncato: la parola in
            # meno lo tiene intero, e la didascalia sotto dice comunque di che si tratta.
            f"{stato.errore_riferimento - stato.errore_modello:+.2f} degC",
            border=True,
            help="Guadagno sulla migliore persistenza a quella scadenza, non sulla piu' "
            "facile da battere.",
        )
        colonne[1].caption(
            f"errore quadratico medio {stato.errore_modello:.2f} degC contro "
            f"{stato.errore_riferimento:.2f} degC della persistenza"
        )

    riquadro(
        colonne[2],
        "Ultimo dato",
        stato.ultimo_dato.strftime("%d/%m/%y") if stato.ultimo_dato else "nessuno",
        stato.ultimo_dato.strftime("slot delle %H UTC del %Y-%m-%d") if stato.ultimo_dato
        else "eseguire `scripts/ingest_era5.py`",
    )

    if not stato.checkpoint:
        riquadro(
            colonne[3], "Checkpoint", "assente",
            f"eseguire `scripts/train_model.py --fold {int(fold)}`",
        )
    elif stato.problemi:
        riquadro(
            colonne[3], "Checkpoint", "sospetto",
            f"fold {int(fold)}: {len(stato.problemi)} incoerenza/e fra gli artefatti",
        )
    else:
        riquadro(
            colonne[3], "Checkpoint", "coerente",
            f"fold {int(fold)}: pesi, statistiche e cronologia della stessa esecuzione",
        )

    if stato.problemi:
        st.error(
            "**Artefatti incoerenti.** Il modello si carica lo stesso, ma non e' detto "
            "che sia quello che si crede di avere:\n\n"
            + "\n".join(f"- {problema}" for problema in stato.problemi)
        )

    st.subheader("Come va letto il numero centrale")
    if stato.guadagno_percento is None:
        st.info(
            "Senza valutazione sul blocco di test non c'e' nulla di onesto da dire sulla "
            "qualita' del modello: l'unica cosa misurabile adesso e' quanti dati ci sono."
        )
    else:
        st.markdown(
            f"Alla scadenza **{etichetta_scadenza(config, stato.scadenza)}** la rete "
            f"sbaglia in media **{stato.errore_modello:.2f} degC**. Da solo questo numero "
            f"non dice se il modello serve: **{stato.errore_riferimento:.2f} degC** si "
            "ottengono ripetendo l'osservazione di ieri alla stessa ora, senza alcun "
            "modello. Il guadagno reale e' quindi "
            f"**{stato.guadagno_percento:+.1f} %**, ed e' quello il numero da guardare."
        )
        st.markdown(
            "Il riferimento e' la **migliore** persistenza a quella scadenza: sceglierne "
            "una piu' debole renderebbe il guadagno adulatorio."
        )

    st.subheader("Da dove continuare")
    colonne = st.columns(3)
    with colonne[0]:
        st.page_link(PAGINA_QUALITA)
    with colonne[1]:
        st.page_link(PAGINA_COPERTURA)
    with colonne[2]:
        st.page_link(PAGINA_CONFRONTO)


def pagina_progetto() -> None:
    intestazione(
        "Che cosa fa il progetto",
        "Ambito, scelte tecniche e occupazione su disco.",
        "Previsione a **3 giorni** (06, 12, 18 UTC) sull'area euro-atlantica a partire "
        "dai giorni precedenti di rianalisi ERA5, con una rete convoluzionale scritta da "
        "zero e addestrata **su CPU**. Prevede temperatura in gradi Celsius, "
        "precipitazione, neve e la **propria incertezza**, calibrata.",
    )

    st.subheader("Scelte tecniche")
    st.caption(
        "Ogni voce dice il ruolo che la tecnologia ha in questo progetto, non che cosa "
        "sia in generale."
    )
    for area in sorted({voce[1] for voce in TECNOLOGIE}):
        with st.expander(area.capitalize(), expanded=(area == "modellazione")):
            for nome, categoria, descrizione in TECNOLOGIE:
                if categoria == area:
                    st.markdown(f"**{nome}** - {descrizione}")

    st.subheader("Occupazione su disco")
    griglia_di_riquadri(
        [(voce.etichetta, voce.valore, voce.nota) for voce in spazio_dati(config)]
    )
    st.caption(
        "Somma delle dimensioni dei file sotto ciascuna directory della configurazione."
    )


# --------------------------------------------------------------------------- #
# Area: Dati
# --------------------------------------------------------------------------- #


def pagina_copertura() -> None:
    intestazione(
        "Copertura dei dati",
        "Dominio, finestra temporale e quali mesi sono davvero nello store.",
    )

    st.subheader("Dominio e finestra")
    # Tre per riga e non quattro: valori come «2.862 / 2.862» non stanno in un quarto di
    # larghezza e Streamlit li troncherebbe con dei puntini, perdendo la meta' che conta.
    griglia_di_riquadri(
        [(voce.etichetta, voce.valore, voce.nota) for voce in panoramica(config)],
        per_riga=3,
    )
    st.caption(
        "Uno **slot** e' un'osservazione su tutta la griglia a una delle ore "
        f"{', '.join(f'{ora:02d}' for ora in config.time.slot_hours)} UTC."
    )

    st.subheader("Slot presenti mese per mese")
    per_mese = copertura_mensile(config)
    if per_mese is None:
        st.warning("Nessun catalogo di slot: eseguire `scripts/ingest_era5.py`.")
        return

    presenti = int(per_mese["presenti"].sum())
    catalogati = int(per_mese["catalogati"].sum())
    colonne = st.columns(3)
    riquadro(colonne[0], "Slot presenti", f"{presenti:,}", "leggibili dal tensore Zarr")
    riquadro(
        colonne[1], "Slot catalogati", f"{catalogati:,}",
        "attesi dall'intero periodo configurato",
    )
    incompleti = per_mese.filter(pl.col("frazione") < 1.0)
    riquadro(
        colonne[2], "Mesi incompleti", f"{incompleti.height} / {per_mese.height}",
        "scaricati ma non ingeriti, oppure non scaricati",
    )

    grafico_barre(
        per_mese["mese"].to_list(),
        per_mese["presenti"].to_list(),
        asse_x="mese",
        asse_y="slot presenti",
    )
    st.caption(
        "Il catalogo copre l'intero periodo configurato, quindi un mese non ancora "
        "ingerito compare a zero. Il primo e l'ultimo mese possono essere piu' bassi "
        "senza essere incompleti: il periodo configurato comincia e finisce a meta' mese. "
        "I mesi davvero con dei buchi sono quelli elencati qui sotto."
    )

    if incompleti.height:
        tabella(
            incompleti.select(
                pl.col("mese").alias("Mese"),
                pl.col("presenti").alias("Slot presenti"),
                pl.col("catalogati").alias("Slot attesi"),
                (pl.col("frazione") * 100).round(1).alias("Copertura (%)"),
            ),
            didascalia="Solo i mesi che non sono completi: sugli altri non c'e' nulla da "
            "decidere. Una copertura sotto il 100 % significa che quelle finestre non "
            "possono entrare ne' nell'addestramento ne' nella valutazione.",
        )
    else:
        st.success("Tutti i mesi catalogati sono presenti nello store.")


def pagina_fold() -> None:
    intestazione(
        "Divisione in fold",
        "Dove passano i confini fra addestramento, validazione e test.",
        "Un campione entra in un blocco solo se **l'intera finestra** di ingresso piu' "
        "bersaglio ci sta dentro: nessun bersaglio di addestramento puo' comparire fra "
        "gli ingressi di validazione. Il distacco fra blocchi e' un margine aggiuntivo "
        "contro l'autocorrelazione, non la difesa principale.",
    )

    struttura = struttura_fold(config)
    if struttura is None:
        st.info(
            "Tabella dei fold non ancora prodotta: eseguire lo script che costruisce "
            "il catalogo."
        )
        return

    tabella(
        struttura.select(
            pl.col("fold").alias("Fold"),
            pl.col("split").cast(pl.Utf8).replace(NOMI_BLOCCHI).alias("Blocco"),
            pl.col("primo_slot").alias("Primo slot"),
            pl.col("ultimo_slot").alias("Ultimo slot"),
            pl.col("slot").alias("Slot nel blocco"),
            pl.col("inizi_ammessi").alias("Finestre utilizzabili"),
        ),
        didascalia="Gli slot sono numerati dall'inizio del periodo. «Finestre "
        "utilizzabili» e' quante finestre complete entrano nel blocco con la finestra "
        f"corrente di {config.windows.input_slots} + {config.windows.output_slots} slot: "
        "e' sempre minore del numero di slot, e ricalcolata sulla configurazione attuale "
        "invece che letta dal file, che potrebbe venire da una finestra diversa.",
        altezza=320,
    )

    distacchi = [
        int(inizio) - int(fine)
        for riga in (
            struttura.sort("fold", "primo_slot")
            .group_by("fold", maintain_order=True)
            .agg(pl.col("primo_slot"), pl.col("ultimo_slot"))
            .iter_rows(named=True)
        )
        for inizio, fine in zip(
            riga["primo_slot"][1:], riga["ultimo_slot"][:-1], strict=False
        )
    ]
    if distacchi:
        st.caption(
            f"Distacco minimo fra due blocchi consecutivi: **{min(distacchi)} slot**. "
            "Zero o meno significherebbe blocchi contigui o sovrapposti."
        )


def pagina_ingressi() -> None:
    catalogo = catalogo_ingressi(config)
    intestazione(
        "Canali in ingresso",
        f"{catalogo.height} campi sull'intera griglia, ricostruiti dalla configurazione "
        "corrente.",
        "Lo **stato** e' la variabile a un dato ritardo, la **tendenza** e' la differenza "
        "fra due istanti; i canali statici e temporali non dipendono dalla finestra. "
        "Questa pagina non richiede ne' modello ne' dati ingeriti.",
    )

    conteggi = (
        catalogo["group"]
        .value_counts()
        .sort("count", descending=True)
        .with_columns(pl.col("group").replace(NOMI_GRUPPI_CANALI))
    )
    grafico_barre(
        conteggi["group"].to_list(),
        conteggi["count"].to_list(),
        asse_x="gruppo di canali",
        asse_y="numero di canali",
    )
    st.caption(
        "Il conteggio per gruppo dice dove sta il grosso dell'ingresso: la maggioranza "
        "dei canali e' quasi sempre lo storico delle stesse poche variabili."
    )

    with st.expander("Elenco completo dei canali"):
        gruppi = sorted(catalogo["group"].unique().to_list())
        filtro = st.multiselect(
            "Gruppi", gruppi, default=[],
            format_func=lambda g: NOMI_GRUPPI_CANALI.get(g, g),
        )
        mostrato = catalogo.filter(catalogo["group"].is_in(filtro)) if filtro else catalogo
        tabella(
            mostrato.select(
                pl.col("channel_index").alias("Indice"),
                pl.col("name").alias("Nome del canale"),
                pl.col("group").replace(NOMI_GRUPPI_CANALI).alias("Gruppo"),
                pl.col("source_variable").alias("Variabile di origine"),
                pl.col("lag").alias("Ritardo (slot)"),
                pl.col("transform").alias("Trasformazione"),
                pl.col("normalized").alias("Normalizzato"),
            ),
            didascalia="L'indice e' la posizione del canale nel tensore che entra nella "
            "rete. Il ritardo e' in slot: 1 slot indietro puo' valere 6 o 12 ore secondo "
            "l'ora del giorno.",
            altezza=320,
        )

    st.subheader("Un canale sulla griglia")
    colonne = st.columns(3)
    split_ingresso = colonne[0].selectbox(
        "Blocco", BLOCCHI, index=0, format_func=lambda s: NOMI_BLOCCHI[s]
    )
    posizione_ingresso = colonne[1].number_input(
        "Finestra", min_value=0, max_value=200, value=0, step=1, key="finestra_ingresso"
    )
    canale_scelto = colonne[2].selectbox(
        "Canale",
        list(range(catalogo.height)),
        format_func=lambda i: f"{i} - {catalogo['name'][i]}",
    )

    # Il risultato non puo' dipendere dal valore *transitorio* del pulsante: alla prima
    # riesecuzione della pagina, provocata da qualsiasi altro widget, il pulsante torna
    # falso e il campo appena calcolato sparirebbe senza spiegazione.
    if st.button("Mostra il canale", key="mostra_canale"):
        st.session_state["canale_richiesto"] = True
    if not st.session_state.get("canale_richiesto"):
        st.caption(
            "Il campo mostrato e' quello che la rete riceve davvero, ricostruito con la "
            "stessa pipeline e riportato indietro alle unita' di partenza."
        )
        return

    try:
        nome, gruppo, variabile, ritardo, campo, unita, avvertenza = ingresso_in_cache(
            percorso_config, int(fold), split_ingresso,
            int(posizione_ingresso), int(canale_scelto),
        )
    except Exception as errore:
        st.error(f"Canale non ispezionabile: {errore}")
        return

    st.markdown(
        f"**{nome}** - gruppo *{NOMI_GRUPPI_CANALI.get(gruppo, gruppo)}*, variabile "
        f"`{variabile}`, ritardo {ritardo} slot, unita' **{unita}**"
    )
    if avvertenza:
        st.warning(avvertenza)

    figura, asse = plt.subplots(figsize=(8, 5), constrained_layout=True)
    immagine = mappa(
        asse, campo, config, titolo=f"{nome} [{unita}]",
        cmap="RdBu_r" if gruppo == "tendency" else "viridis",
    )
    figura.colorbar(immagine, ax=asse, shrink=0.85, label=unita)
    disegna_figura(figura)

    colonne = st.columns(3)
    for colonna, (etichetta, valore) in zip(
        colonne,
        (("Minimo", float(np.min(campo))), ("Medio", float(np.mean(campo))),
         ("Massimo", float(np.max(campo)))),
        strict=False,
    ):
        riquadro(colonna, etichetta, f"{valore:.2f} {unita}")
    st.caption(
        "Il campo e' quello che la rete riceve davvero, ricostruito con la stessa "
        "pipeline e riportato indietro alle unita' di partenza."
    )


# --------------------------------------------------------------------------- #
# Area: Modello
# --------------------------------------------------------------------------- #


def pagina_checkpoint() -> None:
    intestazione(
        f"Checkpoint del fold {int(fold)}",
        "Che cosa contiene il modello salvato e come e' arrivato dove sta.",
    )

    info = informazioni_modello(config, int(fold))
    if not info["disponibile"]:
        st.warning(
            f"Nessun checkpoint in `{info['percorso']}`. "
            f"Eseguire `scripts/train_model.py --fold {int(fold)}`."
        )
        return

    metadati = info["metadati"]
    problemi = coerenza_artefatti(config, int(fold))
    if problemi:
        st.error(
            "**Artefatti incoerenti.** Il modello si carica lo stesso, ma non e' detto "
            "che sia quello che si crede di avere:\n\n"
            + "\n".join(f"- {problema}" for problema in problemi)
        )

    parametri = info.get("n_parametri")
    griglia_di_riquadri(
        [
            ("Variante", str(info["variante"]), "letta dalla configurazione attuale"),
            ("Parametri", f"{parametri:,}" if parametri else "non leggibili",
             "contati nel file dei pesi"),
            ("Canali in ingresso", str(info.get("canali_ingresso") or "?"),
             "dichiarati dal checkpoint"),
            ("Ancoraggio diurno", "attivo" if info["ancoraggio"] else "spento",
             "previsione dello scarto invece del valore assoluto"),
            ("Canali base", str(info["canali_base"]), "larghezza del primo livello"),
            ("Profondita'", str(info["profondita"]), "livelli di sottocampionamento"),
            ("Blocchi per livello", str(info["blocchi_per_livello"]), ""),
            ("Aggiornato", info["aggiornato"].strftime("%Y-%m-%d %H:%M"),
             "data del file dei metadati"),
        ]
    )
    st.caption(
        "«Variante», «canali base», «profondita'» e «blocchi per livello» vengono dalla "
        "configurazione **corrente**, non dal checkpoint: se il file e' stato addestrato "
        "con impostazioni diverse, i canali in ingresso lo rivelano e la verifica di "
        "coerenza qui sopra lo segnala."
    )

    perdita = metadati.get("val_loss")
    if perdita is not None:
        st.info(
            f"Checkpoint salvato all'epoca {metadati.get('epoch', '?')} con perdita di "
            f"validazione {perdita:.4f}."
        )

    st.subheader("Architettura")
    st.markdown(
        "Encoder-decoder a U completamente convoluzionale: si addestra su ritagli e si "
        "applica alla griglia intera. Produce **un solo tensore**, la cui mappatura su "
        "(variabile, componente, scadenza) e' dichiarata in `OutputLayout` invece di "
        "essere indicizzata a mano, perche' scambiare media e log-varianza non farebbe "
        "fallire nulla: produrrebbe solo previsioni sbagliate."
    )
    st.markdown(
        "Con l'ancoraggio attivo la rete prevede lo **scarto** rispetto all'osservazione "
        "piu' recente alla stessa ora del bersaglio. Solo la media viene traslata: la "
        "log-varianza descrive l'incertezza dello scarto."
    )

    curva = curva_apprendimento(info)
    if curva is None or not curva.height:
        st.info("Nessuna cronologia di addestramento accanto al checkpoint.")
        return

    st.subheader("Curva di apprendimento")
    grafico_linee(
        {
            "addestramento": (curva["epoch"].to_list(), curva["train_loss"].to_list()),
            "validazione": (curva["epoch"].to_list(), curva["val_loss"].to_list()),
        },
        asse_x="epoca",
        asse_y="perdita (adimensionale)",
    )
    migliore = curva.sort("val_loss").head(1)
    st.caption(
        f"Migliore epoca **{int(migliore['epoch'][0])}** con perdita di validazione "
        f"{float(migliore['val_loss'][0]):.4f}. La perdita e' la somma pesata dei termini "
        "configurati, quindi il suo valore assoluto non e' confrontabile fra "
        "configurazioni con pesi diversi."
    )


def pagina_uscite() -> None:
    intestazione(
        "Uscite scadenza per scadenza",
        "Il modello eseguito su una finestra, con tutte le scadenze insieme.",
        "Una scadenza sola nasconde il difetto piu' comune di un modello ancorato: "
        "errore piccolo sulla prima e crescente sulle successive.",
    )

    colonne = st.columns(3)
    split_uscita = colonne[0].selectbox(
        "Blocco", BLOCCHI, index=0, key="split_uscita",
        format_func=lambda s: NOMI_BLOCCHI[s],
    )
    posizione_uscita = colonne[1].number_input(
        "Finestra", min_value=0, max_value=200, value=0, step=1, key="finestra_uscita"
    )
    variabile_uscita = colonne[2].selectbox(
        "Variabile", ["t2m"], index=0, format_func=lambda v: NOMI_VARIABILI.get(v, v)
    )

    if st.button("Esegui il modello", key="esegui_uscite"):
        st.session_state["uscite_richieste"] = True
    if not st.session_state.get("uscite_richieste"):
        return

    try:
        uscite = uscite_in_cache(
            percorso_config, int(fold), split_uscita,
            int(posizione_uscita), variabile_uscita,
        )
    except Exception as errore:
        st.error(f"Esecuzione non riuscita: {errore}")
        return

    grafico_linee(
        {
            "errore quadratico medio": (
                uscite["scadenza"].to_list(),
                uscite["radice_errore_quadratico"].to_list(),
            ),
            "errore assoluto medio": (
                uscite["scadenza"].to_list(), uscite["errore_assoluto"].to_list()
            ),
        },
        asse_x="scadenza",
        asse_y="degC",
        etichette_x=[
            etichetta_scadenza(config, int(s)) for s in uscite["scadenza"]
        ],
    )
    st.caption(
        "Se le due curve crescono con la scadenza il modello sta perdendo il vantaggio "
        "dell'ancoraggio man mano che si allontana dall'ultima osservazione."
    )

    tabella(
        uscite.select(
            pl.Series(
                "Scadenza",
                [etichetta_scadenza(config, int(s)) for s in uscite["scadenza"]],
            ),
            pl.col("istante").alias("Istante previsto"),
            pl.col("previsto_medio").round(2).alias("Previsto medio (degC)"),
            pl.col("osservato_medio").round(2).alias("Osservato medio (degC)"),
            pl.col("errore_medio").round(2).alias("Scarto medio (degC)"),
            pl.col("errore_assoluto").round(2).alias("Errore assoluto medio (degC)"),
            pl.col("radice_errore_quadratico").round(2).alias(
                "Errore quadratico medio (degC)"
            ),
        ),
        didascalia="Tutte le medie sono spaziali sull'intero dominio. Uno **scarto medio** "
        "vicino a zero con un errore assoluto grande significa che gli errori si "
        "compensano fra regioni, non che la previsione sia buona: sono le ultime due "
        "colonne a dire quanto si sbaglia.",
    )


# --------------------------------------------------------------------------- #
# Area: Valutazione
# --------------------------------------------------------------------------- #


def pagina_qualita() -> None:
    intestazione(
        "Quanto serve il modello",
        "Confronto con la persistenza sul blocco di dati mai visti.",
        "Il riferimento da battere e' la **persistenza diurna**, cioe' ripetere "
        "l'osservazione di ieri alla stessa ora. Su questo dominio e' un avversario molto "
        "forte, e usare quella ingenua darebbe un vantaggio illusorio.",
    )

    split = st.selectbox(
        "Blocco", ["test", "val"], index=0, format_func=lambda s: NOMI_BLOCCHI[s]
    )
    tabella_metriche = metriche(config, int(fold), split)
    if tabella_metriche is None or not tabella_metriche.height:
        avviso_senza_metriche(int(fold), split)
        return

    variabili = sorted(
        tabella_metriche["variable"].unique().to_list(),
        key=lambda v: list(NOMI_VARIABILI).index(v) if v in NOMI_VARIABILI else 99,
    )
    metriche_disponibili = sorted(tabella_metriche["metric"].unique().to_list())
    colonne = st.columns(2)
    variabile = colonne[0].selectbox(
        "Grandezza", variabili, format_func=lambda v: NOMI_VARIABILI.get(v, v)
    )
    metrica = colonne[1].selectbox(
        "Metrica",
        metriche_disponibili,
        index=metriche_disponibili.index("rmse_celsius")
        if "rmse_celsius" in metriche_disponibili
        else 0,
        format_func=lambda m: NOMI_METRICHE.get(m, m),
    )
    nome_metrica = NOMI_METRICHE.get(metrica, metrica)

    st.subheader("Guadagno sul non fare nulla")
    guadagno = guadagno_leggibile(config, tabella_metriche, variabile, metrica)
    # Senza riferimento il guadagno e' nullo, non zero: disegnarlo come una barra a zero
    # direbbe "il modello non guadagna nulla" invece di "non c'e' con che confrontarlo".
    disegnabile = guadagno.drop_nulls("Guadagno (%)")
    if not guadagno.height:
        st.info("Nessun valore per questa combinazione di grandezza e metrica.")
    else:
        if disegnabile.height:
            grafico_barre(
                disegnabile["Scadenza"].to_list(),
                disegnabile["Guadagno (%)"].to_list(),
                asse_x="scadenza",
                asse_y="guadagno sulla migliore persistenza (%)",
                linea_zero=True,
            )
            st.caption(
                "Zero significa che ripetere il passato avrebbe dato lo stesso "
                "risultato; sotto zero il modello e' peggio del non fare nulla "
                "(barra rossa)."
            )
        tabella(
            guadagno,
            didascalia=f"«Modello» e «Migliore persistenza» sono la metrica *{nome_metrica}* "
            "a quella scadenza. Il riferimento e' la persistenza piu' forte fra quelle "
            "valutate, non la media: un avversario debole gonfierebbe il guadagno.",
        )

    st.subheader("Andamento con la scadenza")
    andamento = per_scadenza(tabella_metriche, variabile, metrica)
    if not andamento.height:
        st.info("Nessun valore per questa combinazione.")
    else:
        largo = andamento.pivot(on="model", index="lead_slot", values="value").sort(
            "lead_slot"
        )
        scadenze = largo["lead_slot"].to_list()
        grafico_linee(
            {
                NOMI_MODELLI.get(colonna, colonna): (
                    scadenze, largo[colonna].to_list()
                )
                for colonna in largo.columns
                if colonna != "lead_slot"
            },
            asse_x="scadenza",
            asse_y=nome_metrica,
            etichette_x=[etichetta_scadenza(config, int(s)) for s in scadenze],
        )
        st.caption(
            "Le stesse curve del grafico precedente prima della differenza: serve a "
            "vedere se il modello peggiora perche' sbaglia di piu' o perche' il problema "
            "diventa piu' difficile per tutti."
        )

    st.subheader("Tutte le metriche aggregate")
    tabella(
        riepilogo_leggibile(tabella_metriche),
        didascalia="Valori mediati su tutti i mesi e tutte le scadenze del blocco, presi "
        "dalle righe gia' aggregate dalla valutazione. «Punti confrontati» e' il numero di "
        "coppie previsione-osservazione su cui il valore e' calcolato.",
        altezza=380,
    )

    ingannevoli = accuratezze_ingannevoli(tabella_metriche)
    if ingannevoli.height:
        righe = "\n".join(
            f"- **{NOMI_VARIABILI.get(r['variable'], r['variable'])}**, modello "
            f"`{r['model']}`: accuratezza {r['accuratezza']:.3f}, ma rispondere sempre "
            f"\"no\" darebbe {r['sempre_no']:.3f} (l'evento accade nel "
            f"{r['frequenza_di_base'] * 100:.1f} % dei casi)."
            for r in ingannevoli.iter_rows(named=True)
        )
        st.warning(
            "Nella tabella qui sopra c'e' un'accuratezza piu' bassa di quella di un "
            "modello muto. Per gli eventi rari va letta insieme alla frequenza di base, "
            "oppure sostituita dal Brier e dall'F1.\n\n" + righe
        )


def pagina_confronto() -> None:
    intestazione(
        "Previsto contro osservato",
        "Una singola finestra del blocco di test, campo per campo.",
    )

    colonne = st.columns(2)
    posizione = colonne[0].slider("Finestra di test", 0, 60, 0)
    scadenza = colonne[1].slider(
        "Scadenza", 0, config.windows.output_slots - 1, 2,
        format="%d",
    )
    st.caption(f"Scadenza scelta: **{etichetta_scadenza(config, int(scadenza))}**.")

    try:
        previsto, osservato, riferimento, istante, errore_medio = confronto_in_cache(
            percorso_config, int(fold), int(posizione), int(scadenza)
        )
    except Exception as errore:
        st.error(f"Impossibile calcolare il confronto: {errore}")
        return

    colonne = st.columns(2)
    riquadro(
        colonne[0], "Istante previsto",
        istante.strftime("%Y-%m-%d %H:%M UTC") if istante else "non ricostruibile",
        "il modello non ha mai visto questa finestra",
    )
    riquadro(
        colonne[1], "Errore assoluto medio", f"{errore_medio:.2f} degC",
        "media su tutte le celle del dominio",
    )

    limite = float(max(abs(np.nanmin(osservato)), abs(np.nanmax(osservato))))
    figura, assi = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    immagine = mappa(assi[0], previsto, config, titolo="Previsto (degC)",
                     cmap="RdYlBu_r", vmin=-limite, vmax=limite)
    mappa(assi[1], osservato, config, titolo="Osservato (degC)",
          cmap="RdYlBu_r", vmin=-limite, vmax=limite)
    figura.colorbar(immagine, ax=assi[:2], shrink=0.85, label="degC")

    differenza = previsto - osservato
    estremo = float(np.nanpercentile(np.abs(differenza), 99)) or 1.0
    immagine_diff = mappa(assi[2], differenza, config,
                          titolo="Previsto meno osservato (degC)",
                          cmap="coolwarm", vmin=-estremo, vmax=estremo)
    figura.colorbar(immagine_diff, ax=assi[2], shrink=0.85, label="degC")
    disegna_figura(figura)
    st.caption(
        "I primi due pannelli condividono la scala di colore, quindi sono confrontabili "
        "a vista. Il terzo e' centrato sullo zero e tagliato al 99esimo percentile, "
        "perche' poche celle estreme schiaccerebbero tutto il resto sul bianco."
    )

    if riferimento is None:
        return

    st.subheader("Contro la persistenza diurna")
    errore_modello = float(np.mean(np.abs(differenza)))
    errore_riferimento = float(np.mean(np.abs(riferimento - osservato)))
    colonne = st.columns(2)
    riquadro(
        colonne[0], "Errore del modello", f"{errore_modello:.2f} degC",
        "su questa finestra",
    )
    colonne[1].metric(
        "Errore della persistenza diurna",
        f"{errore_riferimento:.2f} degC",
        f"{errore_riferimento - errore_modello:+.2f} degC di scarto",
        border=True,
    )
    st.caption(
        "Lo scarto e' positivo (verde) quando il modello sbaglia meno della persistenza. "
        "Una sola finestra non e' comunque una misura di qualita': per quella serve la "
        "pagina «Quanto serve il modello», che aggrega l'intero blocco di test."
    )


def pagina_errori() -> None:
    intestazione(
        "Dove si concentra l'errore",
        "Errore quadratico medio per cella, aggregato su piu' finestre di test.",
        "Una metrica scalare non puo' dire **dove** si sbaglia: un errore concentrato sui "
        "rilievi ha cause diverse da uno diffuso sull'oceano.",
    )

    colonne = st.columns(2)
    n_finestre = colonne[0].slider("Finestre aggregate", 2, 30, 8)
    scadenza = colonne[1].slider("Scadenza", 0, config.windows.output_slots - 1, 2)
    st.caption(f"Scadenza scelta: **{etichetta_scadenza(config, int(scadenza))}**.")

    try:
        campo, usate = errori_in_cache(
            percorso_config, int(fold), int(n_finestre), int(scadenza)
        )
    except Exception as errore:
        st.error(f"Impossibile calcolare la mappa: {errore}")
        return

    figura, asse = plt.subplots(figsize=(11, 6), constrained_layout=True)
    immagine = mappa(asse, campo, config, titolo="Errore quadratico medio per cella (degC)",
                     cmap="inferno", vmin=0.0,
                     vmax=float(np.nanpercentile(campo, 99)))
    # Vigo di Cadore, il punto di interesse dichiarato del progetto.
    asse.plot(12.5, 46.5, marker="o", markersize=7, markerfacecolor="none",
              markeredgecolor="cyan", markeredgewidth=1.8)
    asse.annotate("Vigo di Cadore", (12.5, 46.5), textcoords="offset points",
                  xytext=(9, 5), color="cyan", fontsize=9)
    figura.colorbar(immagine, ax=asse, shrink=0.85, label="degC")
    disegna_figura(figura)
    st.caption(
        f"Aggregazione su **{usate}** finestre di test. La scala e' tagliata al 99esimo "
        "percentile: le celle piu' scure sono quelle in cui il modello e' piu' affidabile."
    )

    voci = [
        ("Errore medio", f"{float(np.mean(campo)):.2f} degC", "su tutte le celle"),
        ("Errore mediano", f"{float(np.median(campo)):.2f} degC",
         "meta' delle celle sta sotto"),
        ("Errore massimo", f"{float(np.max(campo)):.2f} degC", "cella peggiore"),
    ]
    riga = round((config.region.north - 46.5) / config.region.grid_step)
    colonna_griglia = round((12.5 - config.region.west) / config.region.grid_step)
    if 0 <= riga < campo.shape[0] and 0 <= colonna_griglia < campo.shape[1]:
        voci.append(
            ("Su Vigo di Cadore", f"{float(campo[riga, colonna_griglia]):.2f} degC",
             "cella piu' vicina al punto di interesse")
        )
    griglia_di_riquadri(voci)


def pagina_probabilita() -> None:
    intestazione(
        "Quanto valgono le probabilita'",
        "Se il modello dice 40 % di pioggia, piove nel 40 % dei casi?",
        "Una probabilita' e' utile solo se **mantiene la promessa**: su cento casi "
        "annunciati al 40 % la pioggia deve arrivare in quaranta. Un modello puo' avere un "
        "errore basso e probabilita' inutilizzabili, quindi le due cose si guardano "
        "separatamente.",
    )

    split = st.selectbox(
        "Blocco", ["test", "val"], index=0, format_func=lambda s: NOMI_BLOCCHI[s]
    )
    tabella_affidabilita = affidabilita(config, int(fold), split)
    if tabella_affidabilita is None:
        st.info(
            "Nessun diagramma di affidabilita' per questo blocco: si ottiene eseguendo "
            "`scripts/evaluate_model.py`, che scrive `reliability.parquet` nella cartella "
            "del fold."
        )
    else:
        variabili = sorted(tabella_affidabilita["variable"].unique().to_list())
        variabile = st.selectbox(
            "Grandezza", variabili, format_func=lambda v: NOMI_VARIABILI.get(v, v),
            key="variabile_affidabilita",
        )
        scelta = tabella_affidabilita.filter(pl.col("variable") == variabile)

        curve: dict[str, tuple[list[float], list[float]]] = {
            NOMI_MODELLI.get(modello, modello): (
                gruppo["forecast_mean"].to_list(),
                gruppo["observed_frequency"].to_list(),
            )
            for modello, gruppo in scelta.group_by("model", maintain_order=True)
        }
        # La diagonale e' il modello perfettamente onesto: senza di lei il grafico non
        # dice nulla, perche' non c'e' un riferimento rispetto a cui essere sopra o sotto.
        curve["Promessa mantenuta"] = ([0.0, 1.0], [0.0, 1.0])
        grafico_linee(
            curve,
            asse_x="probabilita' annunciata",
            asse_y="frequenza osservata",
        )
        st.caption(
            "Sopra la diagonale il modello e' **timido** (piove piu' di quanto prometta), "
            "sotto e' **troppo sicuro**. I bin senza casi non sono disegnati: una curva "
            "che scende a zero perche' un intervallo e' vuoto sembrerebbe un errore grave "
            "e sarebbe solo assenza di dati."
        )

        scarti = scarto_di_affidabilita(scelta)
        tabella(
            scarti,
            didascalia="«scarto» e' probabilita' annunciata meno frequenza osservata: "
            "positivo significa che il modello promette l'evento piu' spesso di quanto "
            "accada. «count» dice su quanti punti poggia la riga, e le righe con pochi "
            "punti non vanno lette come tendenze.",
        )
        peggiore = scarti.sort(pl.col("scarto").abs(), descending=True).head(1)
        if peggiore.height:
            riga = peggiore.row(0, named=True)
            st.caption(
                f"Scostamento massimo: nel bin {riga['bin_lower']:.1f}-"
                f"{riga['bin_upper']:.1f} il modello annuncia "
                f"{riga['forecast_mean'] * 100:.0f} % e l'evento accade nel "
                f"{riga['observed_frequency'] * 100:.0f} % dei casi "
                f"({riga['count']:,} punti)."
            )

    st.subheader("La correzione applicata")
    tabella_calibrazione = calibrazione(config, int(fold), "val")
    if tabella_calibrazione is None:
        st.info(
            "Nessuna curva di calibrazione salvata. Viene stimata sulla validazione "
            "durante la valutazione e scritta in `calibration.parquet`."
        )
        return

    st.markdown(
        "La calibrazione e' una funzione **monotona** stimata sulla validazione (algoritmo "
        "PAVA) che riscrive le probabilita' grezze. E' stimata sulla validazione e non sul "
        "test perche' usare il test per correggere il modello e poi per giudicarlo "
        "renderebbe il giudizio ottimistico."
    )
    variabili_calibrate = sorted(tabella_calibrazione["variable"].unique().to_list())
    curve_calibrazione = {
        NOMI_VARIABILI.get(nome, nome): (
            tabella_calibrazione.filter(pl.col("variable") == nome)["probability_in"].to_list(),
            tabella_calibrazione.filter(pl.col("variable") == nome)["probability_out"].to_list(),
        )
        for nome in variabili_calibrate
    }
    curve_calibrazione["Nessuna correzione"] = ([0.0, 1.0], [0.0, 1.0])
    grafico_linee(
        curve_calibrazione,
        asse_x="probabilita' grezza della rete",
        asse_y="probabilita' corretta",
    )
    st.caption(
        "Dove la curva sta sopra la diagonale la correzione alza la probabilita', dove "
        "sta sotto la abbassa. Un tratto piatto significa che la rete distingueva valori "
        "diversi a cui corrispondeva la stessa frequenza osservata."
    )
    tabella(
        effetto_calibrazione(tabella_calibrazione),
        didascalia="Quanto la correzione sposta le probabilita'. Un intervallo grezzo "
        "molto stretto (minimo e massimo vicini) e' il sintomo di una rete che dice "
        "quasi sempre la stessa cosa, e in quel caso nessuna calibrazione puo' aggiungere "
        "l'informazione che manca.",
    )


# --------------------------------------------------------------------------- #
# Area: Clima
# --------------------------------------------------------------------------- #


def pagina_clima() -> None:
    intestazione(
        "Andamenti climatici",
        "Che cosa la serie ingerita permette davvero di dire sul clima.",
        "Tutte le medie spaziali sono **pesate per l'area della cella**: su una griglia in "
        "latitudine e longitudine le celle a nord coprono molto meno terreno, e una media "
        "aritmetica darebbe loro un peso che non hanno.",
    )

    serie = copertura_in_cache(percorso_config)
    giudizio = giudizio_sulla_serie(serie)
    if serie is None or serie.n_anni_completi < 2:
        st.warning(giudizio)
    else:
        st.info(giudizio)
    if serie is None:
        return

    griglia_di_riquadri(
        [
            ("Anni presenti", str(len(serie.anni)), "anche solo parziali"),
            ("Anni completi", str(serie.n_anni_completi), "tutti e dodici i mesi"),
            ("Slot", f"{serie.slot:,}", "osservazioni sull'intera griglia"),
            ("Periodo", f"{serie.primo[:7]} .. {serie.ultimo[:7]}", "primo e ultimo mese"),
        ]
    )

    st.subheader("Media mensile sul dominio")
    mensili = mensili_in_cache(percorso_config)
    if mensili is None:
        st.warning("Nessuna serie disponibile: ingerire almeno un mese.")
    else:
        periodi = mensili["periodo"].to_list()
        grafico_linee(
            {"media sul dominio": (list(range(len(periodi))), mensili["media"].to_list())},
            asse_x="mese",
            asse_y="temperatura a 2 m (degC)",
            etichette_x=periodi,
            altezza=3.4,
        )
        st.caption(
            "L'oscillazione che domina questo grafico e' il ciclo stagionale: e' di gran "
            "lunga il segnale piu' forte e nasconde ogni differenza fra annate."
        )

        st.subheader("Lo stesso mese in anni diversi")
        st.caption(
            "E' il confronto onesto su una serie corta: toglie di mezzo il ciclo "
            "stagionale e lascia vedere la differenza fra annate."
        )
        interannuale = confronto_interannuale(config)
        if interannuale is None:
            st.info("Nessun mese si ripete ancora in due anni diversi.")
        else:
            largo = interannuale.pivot(on="year", index="month", values="media").sort(
                "month"
            )
            anni = [c for c in largo.columns if c != "month"]
            tabella(
                largo.select(
                    pl.col("month").alias("Mese"),
                    *[pl.col(anno).round(2).alias(f"{anno} (degC)") for anno in anni],
                ),
                didascalia="Temperatura media a 2 m sul dominio, per mese e per annata. "
                "Una cella vuota e' un mese che quell'anno non e' stato ingerito: le "
                "colonne con dei vuoti non sono confrontabili riga per riga.",
            )

        st.subheader("Ciclo stagionale")
        stagionale = ciclo_stagionale(config)
        if stagionale is not None:
            tabella(
                stagionale.select(
                    pl.col("month").alias("Mese"),
                    pl.col("media").round(2).alias("Media (degC)"),
                    pl.col("minimo").round(2).alias("Annata piu' fredda (degC)"),
                    pl.col("massimo").round(2).alias("Annata piu' calda (degC)"),
                    pl.col("anni").alias("Annate disponibili"),
                ),
                didascalia="Media per mese dell'anno. Dove «annate disponibili» vale 1 la "
                "riga non e' una media: e' quell'unica annata, e minimo, media e massimo "
                "coincidono.",
            )

    st.subheader("Tendenza annuale")
    tendenza = tendenza_annuale(config)
    if tendenza is None:
        st.warning(
            "Meno di due anni completi: non viene calcolata alcuna pendenza. Una retta su "
            "un punto solo non esiste, e su due punti non ha incertezza stimabile."
        )
    else:
        basso, alto = tendenza.intervallo
        griglia_di_riquadri(
            [
                ("Pendenza", f"{tendenza.pendenza:+.3f} degC/anno", "retta ai minimi quadrati"),
                (
                    "Intervallo 95 %",
                    "non stimabile" if not np.isfinite(basso) else
                    f"{basso:+.3f} .. {alto:+.3f}",
                    "degC/anno",
                ),
                ("Anni usati", str(tendenza.n_anni), "solo annate complete"),
            ],
            per_riga=3,
        )
        if not tendenza.significativa:
            st.warning(
                f"La pendenza **non** e' significativa: servono almeno "
                f"{ANNI_MINIMI_PER_TENDENZA} anni completi e un intervallo che non "
                "contenga lo zero. Il valore mostrato descrive queste annate, non il clima."
            )

    st.subheader("Differenza per cella fra due annate")
    confrontabili = mesi_confrontabili(config)
    if not confrontabili:
        st.info("Nessun mese disponibile in due anni diversi.")
    else:
        per_mese = dict(confrontabili)
        colonne = st.columns(3)
        mese = colonne[0].selectbox("Mese", sorted(per_mese), format_func=lambda m: f"{m:02d}")
        anni_disponibili = per_mese[mese]
        anno_a = colonne[1].selectbox("Annata di riferimento", anni_disponibili, index=0)
        anno_b = colonne[2].selectbox(
            "Annata da confrontare", anni_disponibili, index=len(anni_disponibili) - 1
        )
        if anno_a == anno_b:
            st.info("Scegliere due annate diverse.")
        else:
            differenza = mappa_differenza_mensile(config, mese, anno_a, anno_b)
            if differenza is None:
                st.warning("Dati insufficienti per una delle due annate.")
            else:
                estremo = float(np.nanmax(np.abs(differenza)))
                figura, asse = plt.subplots(figsize=(8, 5), constrained_layout=True)
                immagine = mappa(
                    asse, differenza, config,
                    titolo=f"{mese:02d}/{anno_b} meno {mese:02d}/{anno_a} (degC)",
                    cmap="RdBu_r", vmin=-estremo, vmax=estremo,
                )
                figura.colorbar(immagine, ax=asse, shrink=0.85, label="degC")
                disegna_figura(figura)
                st.caption(
                    f"Media sul dominio: {float(np.nanmean(differenza)):+.2f} degC. La "
                    "scala e' centrata sullo zero, cosi' il colore non suggerisce un segno "
                    "che i dati non hanno. Una sola coppia di annate mostra "
                    "**variabilita' del tempo atmosferico**, non una tendenza."
                )

    st.subheader("Giornate caratteristiche a Vigo di Cadore")
    estremi = estremi_in_cache(percorso_config)
    if estremi is None:
        st.info("Nessun dato per gli indici.")
        return
    tabella(
        estremi.select(
            pl.col("year").alias("Anno"),
            pl.col("mattine_di_gelo").alias("Mattine di gelo (slot 06 UTC)"),
            pl.col("mattine_sopra_20").alias("Mattine sopra 20 degC"),
            pl.col("slot_pioggia_intensa").alias("Slot di pioggia intensa"),
            pl.col("slot_con_neve").alias("Slot con neve"),
            pl.col("temperatura_media").round(2).alias("Temperatura media (degC)"),
            pl.col("slot").alias("Slot disponibili"),
        ),
        didascalia="Gli indici climatici standard usano minimo e massimo giornalieri; qui "
        "il giorno ha tre osservazioni, quindi il gelo si conta sullo slot delle 06 UTC, "
        "il piu' vicino al minimo mattutino. Sono approssimazioni dichiarate. «Slot "
        "disponibili» dice quanti dati ha ciascun anno: un anno parziale non e' "
        "confrontabile con uno completo.",
    )


# --------------------------------------------------------------------------- #
# Area: Addestramento
# --------------------------------------------------------------------------- #


def pagina_avvia() -> None:
    intestazione(
        "Avvia un'esecuzione",
        "Un processo separato, con la sua cartella e la sua configurazione salvata.",
        "L'esecuzione sopravvive alla chiusura di questa pagina e resta ripetibile da "
        "terminale con la configurazione che trova salvata dentro. I parametri passano "
        "per lo **stesso schema** che valida la configurazione del progetto: un valore "
        "fuori intervallo viene rifiutato qui, prima che il processo parta, non a meta' "
        "addestramento.",
    )

    # Il nome predefinito va generato **una volta**: rigenerarlo a ogni riesecuzione
    # sovrascriverebbe quello appena digitato non appena si tocca un altro campo.
    if "nome_predefinito" not in st.session_state:
        st.session_state["nome_predefinito"] = runs.nome_proposto()
    nome_esecuzione = st.text_input("Nome", st.session_state["nome_predefinito"])

    colonne = st.columns(4)
    epoche = colonne[0].number_input(
        "Epoche", min_value=1, max_value=200, value=int(config.training.epochs)
    )
    passo = colonne[1].number_input(
        "Passo di apprendimento", min_value=1e-6, max_value=1e-1,
        value=float(config.training.learning_rate), format="%.5f",
    )
    lotto = colonne[2].number_input(
        "Dimensione del lotto", min_value=1, max_value=64,
        value=int(config.training.batch_size),
    )
    ritaglio = colonne[3].number_input(
        "Lato del ritaglio (celle)", min_value=32, max_value=256,
        value=int(config.training.crop_size), step=16,
    )

    colonne = st.columns(4)
    campioni = colonne[0].number_input(
        "Campioni per epoca", min_value=8, max_value=8192,
        value=int(config.training.samples_per_epoch), step=8,
    )
    seme = colonne[1].number_input(
        "Seme", min_value=0, max_value=10**6, value=int(config.training.seed)
    )
    variante = colonne[2].selectbox(
        "Variante",
        sorted(variants.available()),
        index=sorted(variants.available()).index(config.model.variant),
    )
    ancoraggio = colonne[3].checkbox(
        "Ancoraggio diurno", value=bool(config.model.anchor_diurnal)
    )
    if not ancoraggio:
        st.warning(
            "Senza ancoraggio il banco misura 6.61 degC contro 3.65: e' la scelta con "
            "l'effetto piu' grande di tutte quelle provate."
        )

    st.markdown("**Pesi della perdita**")
    pesi_attuali = config.training.loss_weights.model_dump()
    colonne = st.columns(len(pesi_attuali))
    pesi_scelti = {
        nome_peso: colonna.number_input(
            nome_peso, min_value=0.0, max_value=20.0,
            value=float(valore), step=0.1, key=f"peso_{nome_peso}",
        )
        for colonna, (nome_peso, valore) in zip(
            colonne, pesi_attuali.items(), strict=False
        )
    }
    st.caption(
        "I pesi moltiplicano i termini della perdita: cambiarli rende il valore della "
        "perdita non confrontabile con quello delle esecuzioni precedenti."
    )

    stima = epoche * campioni / max(int(config.training.batch_size), 1)
    st.caption(
        f"Circa {int(stima)} passi di ottimizzazione. Sul portatile usato per lo sviluppo "
        "un'epoca da 512 campioni con ritaglio 96 richiede circa 7 minuti."
    )

    if not st.button("Avvia", type="primary"):
        return

    parametri = {
        "epochs": int(epoche), "learning_rate": float(passo),
        "batch_size": int(lotto), "crop_size": int(ritaglio),
        "samples_per_epoch": int(campioni), "seed": int(seme),
        "variant": variante, "anchor_diurnal": bool(ancoraggio),
        "loss_weights": pesi_scelti,
    }
    try:
        esecuzione = runs.avvia(
            config, nome=nome_esecuzione, fold=int(fold),
            parametri=parametri, project_root=PROJECT_ROOT,
        )
    except (RunError, ValueError) as errore:
        st.error(f"Non avviata: {errore}")
    else:
        st.success(
            f"Avviata **{esecuzione.nome}** (processo {esecuzione.pid}). "
            "Segui l'andamento nella pagina «Esecuzioni»."
        )


def pagina_esecuzioni() -> None:
    intestazione(
        "Esecuzioni",
        "Le esecuzioni avviate da questa dashboard e il loro andamento.",
    )

    elenco = runs.elenca(config)
    if not elenco:
        st.info("Nessuna esecuzione avviata da questa pagina.")
        return

    tabella(
        runs.confronto(config),
        didascalia="Una riga per esecuzione. «migliore» e' la perdita di validazione piu' "
        "bassa raggiunta finora: confrontabile solo fra esecuzioni con gli stessi pesi "
        "della perdita. Le altre colonne sono i parametri che distinguono l'esecuzione.",
    )

    scelta = st.selectbox("Dettaglio", [e.nome for e in elenco])
    dettaglio = runs.leggi(config, scelta)
    if dettaglio is None:
        return

    storia = runs.cronologia(dettaglio, int(dettaglio.parametri.get("fold", 0)))
    griglia_di_riquadri(
        [
            ("Stato", dettaglio.stato, ""),
            ("Avviata", dettaglio.avviata.strftime("%d/%m %H:%M"), ""),
            ("Processo", str(dettaglio.pid or "-"), "PID del sistema operativo"),
            ("Epoche concluse", str(len(storia)), "righe scritte nella cronologia"),
        ]
    )

    premuto = dettaglio.attiva and st.button("Ferma", key=f"ferma_{scelta}")
    if premuto and runs.ferma(config, scelta):
        st.warning("Richiesta di arresto inviata.")
        st.rerun()

    if storia:
        grafico_linee(
            {
                "addestramento": (
                    [r["epoch"] for r in storia], [r["train_loss"] for r in storia]
                ),
                "validazione": (
                    [r["epoch"] for r in storia], [r["val_loss"] for r in storia]
                ),
            },
            asse_x="epoca",
            asse_y="perdita (adimensionale)",
        )
        st.caption(
            "Se la curva di validazione risale mentre quella di addestramento scende, il "
            "modello sta imparando a memoria le finestre viste."
        )

    st.markdown("**Ultime righe del diario**")
    st.code(runs.coda_del_diario(dettaglio, righe=30) or "(ancora vuoto)")
    st.caption(f"Comando: `{' '.join(dettaglio.comando)}`")

    if st.button("Aggiorna"):
        st.rerun()


# --------------------------------------------------------------------------- #
# Area: Sistema
# --------------------------------------------------------------------------- #


def pagina_risorse() -> None:
    intestazione(
        "CPU e memoria",
        "Carico della macchina e processi del progetto in esecuzione.",
    )

    if st.button("Aggiorna"):
        st.cache_data.clear()
    stato = risorse()

    griglia_di_riquadri(
        [
            ("CPU totale", f"{stato['cpu_totale']:.0f} %", "media su tutti i core"),
            ("Core logici", str(stato["core"]), "compreso l'SMT"),
            (
                "Memoria",
                f"{stato['memoria_usata_gb']:.1f} / {stato['memoria_totale_gb']:.1f} GB",
                "dell'intera macchina",
            ),
            ("Memoria in uso", f"{stato['memoria_percento']:.0f} %", ""),
        ]
    )

    if stato["cpu_per_core"]:
        st.subheader("Carico per core")
        grafico_barre(
            [str(indice) for indice in range(len(stato["cpu_per_core"]))],
            stato["cpu_per_core"],
            asse_x="core logico",
            asse_y="carico (%)",
            altezza=2.6,
        )
        st.caption(
            "Un solo core al 100 % con gli altri fermi significa che l'addestramento non "
            "sta usando il parallelismo disponibile."
        )

    st.subheader("Processi del progetto")
    if not stato["processi"]:
        st.info("Nessun processo del progetto in esecuzione.")
        return
    tabella(
        pl.DataFrame(stato["processi"]).select(
            pl.col("pid").alias("PID"),
            pl.col("comando").alias("Comando"),
            pl.col("cpu").round(1).alias("CPU (%)"),
            pl.col("memoria_mb").round(0).alias("Memoria (MB)"),
        ),
        didascalia="Sono elencati solo i processi il cui comando cita il progetto: "
        "addestramento, banchi, scaricamento e ingestione. Il carico CPU e' misurato "
        "dall'ultima lettura, quindi la prima apertura della pagina puo' mostrare zero.",
    )


# --------------------------------------------------------------------------- #
# Navigazione e stato condiviso
# --------------------------------------------------------------------------- #

# La pagina predefinita non dichiara un `url_path`: Streamlit la serve comunque sulla
# radice e ignora il percorso, quindi dichiararne uno produrrebbe un indirizzo che
# sembra valido e risponde "Page not found".
PAGINA_STATO = st.Page(
    pagina_stato, title="Stato del progetto", icon=":material/home:", default=True
)
PAGINA_PROGETTO = st.Page(
    pagina_progetto, title="Che cosa fa il progetto", icon=":material/menu_book:",
    url_path="progetto",
)
PAGINA_COPERTURA = st.Page(
    pagina_copertura, title="Copertura dei dati", icon=":material/calendar_month:",
    url_path="copertura",
)
PAGINA_FOLD = st.Page(
    pagina_fold, title="Divisione in fold", icon=":material/extension:", url_path="fold"
)
PAGINA_INGRESSI = st.Page(
    pagina_ingressi, title="Canali in ingresso", icon=":material/tune:",
    url_path="ingressi",
)
PAGINA_CHECKPOINT = st.Page(
    pagina_checkpoint, title="Checkpoint e architettura", icon=":material/psychology:",
    url_path="checkpoint",
)
PAGINA_USCITE = st.Page(
    pagina_uscite, title="Uscite scadenza per scadenza", icon=":material/output:",
    url_path="uscite",
)
PAGINA_QUALITA = st.Page(
    pagina_qualita, title="Quanto serve il modello", icon=":material/insights:",
    url_path="qualita",
)
PAGINA_CONFRONTO = st.Page(
    pagina_confronto, title="Previsto contro osservato", icon=":material/map:",
    url_path="confronto",
)
PAGINA_ERRORI = st.Page(
    pagina_errori, title="Dove si concentra l'errore",
    icon=":material/local_fire_department:", url_path="errori",
)
PAGINA_PROBABILITA = st.Page(
    pagina_probabilita, title="Quanto valgono le probabilita'",
    icon=":material/percent:", url_path="probabilita",
)
PAGINA_CLIMA = st.Page(
    pagina_clima, title="Andamenti climatici", icon=":material/public:", url_path="clima"
)
PAGINA_AVVIA = st.Page(
    pagina_avvia, title="Avvia un'esecuzione", icon=":material/play_arrow:",
    url_path="avvia",
)
PAGINA_ESECUZIONI = st.Page(
    pagina_esecuzioni, title="Esecuzioni", icon=":material/list_alt:",
    url_path="esecuzioni",
)
PAGINA_RISORSE = st.Page(
    pagina_risorse, title="CPU e memoria", icon=":material/memory:", url_path="risorse"
)

# Il raggruppamento e' il punto: nella versione precedente dati, modello, previsione,
# valutazione, clima e diagnostica stavano tutti allo stesso livello in una lista piatta,
# e l'ordine non diceva nulla su che cosa dipendesse da che cosa.
navigazione = st.navigation(
    {
        "Sintesi": [PAGINA_STATO, PAGINA_PROGETTO],
        "Dati": [PAGINA_COPERTURA, PAGINA_FOLD, PAGINA_INGRESSI],
        "Modello": [PAGINA_CHECKPOINT, PAGINA_USCITE],
        "Valutazione": [
            PAGINA_QUALITA, PAGINA_PROBABILITA, PAGINA_CONFRONTO, PAGINA_ERRORI,
        ],
        "Clima": [PAGINA_CLIMA],
        "Addestramento": [PAGINA_AVVIA, PAGINA_ESECUZIONI],
        "Sistema": [PAGINA_RISORSE],
    },
    # Senza questo Streamlit nasconde le ultime aree dietro un "vedi altre": il
    # raggruppamento serve proprio a far vedere tutte le aree insieme.
    expanded=True,
)

with st.sidebar:
    st.divider()
    with st.expander("Impostazioni", expanded=False):
        percorso_config = st.text_input(
            "Configurazione", str(PROJECT_ROOT / "configs" / "default.yaml")
        )
        fold = st.number_input("Fold", min_value=0, max_value=20, value=0, step=1)

try:
    config = carica_config(percorso_config)
except Exception as errore:
    st.error(f"Configurazione non caricabile: {errore}")
    st.stop()

with st.sidebar:
    st.caption(f"Fold **{int(fold)}** - `{Path(percorso_config).name}`")
    st.caption(
        "Valutazione, confronto visivo e mappa degli errori usano il blocco di **test**, "
        "cioe' finestre che il modello non ha mai visto."
    )

navigazione.run()
