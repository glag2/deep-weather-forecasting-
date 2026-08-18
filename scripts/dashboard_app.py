"""Dashboard di ispezione del progetto.

Avvio:

    uv run streamlit run scripts/dashboard_app.py

Tutta la lettura degli artefatti sta in ``dwf.dashboard``; qui c'e' solo la
presentazione. I calcoli costosi (esecuzione del modello su una finestra, mappa degli
errori) sono dietro una cache, altrimenti ogni interazione con un cursore rieseguirebbe
la rete sull'intera griglia.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
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
    TECNOLOGIE,
    accuratezze_ingannevoli,
    catalogo_ingressi,
    coerenza_artefatti,
    confronto_visivo,
    copertura_mensile,
    curva_apprendimento,
    guadagno_su_persistenza,
    informazioni_modello,
    ispeziona_ingresso,
    ispeziona_uscite,
    mappa_errori,
    metriche,
    panoramica,
    per_scadenza,
    riepilogo_metriche,
    risorse,
    spazio_dati,
    struttura_fold,
)
from dwf.models import variants  # noqa: E402
from dwf.runs import RunError  # noqa: E402

st.set_page_config(page_title="Deep Weather Forecasting", layout="wide")


@st.cache_resource
def carica_config(percorso: str) -> Config:
    return Config.load(percorso)


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
    asse.set_title(titolo, fontsize=10)
    asse.set_xlabel("longitudine", fontsize=8)
    asse.set_ylabel("latitudine", fontsize=8)
    asse.tick_params(labelsize=7)
    return immagine


# --------------------------------------------------------------------------- #
# Barra laterale
# --------------------------------------------------------------------------- #

st.sidebar.title("Deep Weather Forecasting")
percorso_config = st.sidebar.text_input(
    "Configurazione", str(PROJECT_ROOT / "configs" / "default.yaml")
)

try:
    config = carica_config(percorso_config)
except Exception as errore:
    st.error(f"Configurazione non caricabile: {errore}")
    st.stop()

fold = st.sidebar.number_input("Fold", min_value=0, max_value=20, value=0, step=1)
sezione = st.sidebar.radio(
    "Sezione",
    ["Panoramica", "Dati", "Modello", "Prestazioni", "Confronto visivo",
     "Mappa degli errori", "Clima", "Addestramento", "Risorse"],
)
st.sidebar.caption(
    "Il confronto visivo e la mappa degli errori usano il blocco di **test**, "
    "cioe' finestre che il modello non ha mai visto."
)


# --------------------------------------------------------------------------- #
# Panoramica
# --------------------------------------------------------------------------- #

if sezione == "Panoramica":
    st.header("Panoramica")
    riquadri = panoramica(config)
    for riga in range(0, len(riquadri), 3):
        colonne = st.columns(3)
        for colonna, riquadro in zip(colonne, riquadri[riga : riga + 3], strict=False):
            colonna.metric(riquadro.etichetta, riquadro.valore, riquadro.nota,
                           delta_color="off")

    st.subheader("Che cosa fa")
    st.markdown(
        "Previsione a **3 giorni** (06, 12, 18 UTC) sull'area euro-atlantica a partire "
        "dai giorni precedenti di rianalisi ERA5, con una rete convoluzionale scritta "
        "da zero e addestrata **su CPU**. Prevede temperatura in gradi Celsius, "
        "precipitazione, neve e la **propria incertezza**, calibrata."
    )

    st.subheader("Tecnologie impiegate")
    for area in sorted({voce[1] for voce in TECNOLOGIE}):
        with st.expander(area.capitalize(), expanded=(area == "modellazione")):
            for nome, categoria, descrizione in TECNOLOGIE:
                if categoria == area:
                    st.markdown(f"**{nome}** - {descrizione}")

    st.subheader("Occupazione su disco")
    colonne = st.columns(4)
    for colonna, riquadro in zip(colonne, spazio_dati(config), strict=False):
        colonna.metric(riquadro.etichetta, riquadro.valore)


# --------------------------------------------------------------------------- #
# Dati
# --------------------------------------------------------------------------- #

elif sezione == "Dati":
    st.header("Dati disponibili")
    copertura = copertura_mensile(config)
    if copertura is None:
        st.warning("Nessun catalogo di slot: eseguire `scripts/ingest_era5.py`.")
    else:
        presenti = int(copertura["presenti"].sum())
        catalogati = int(copertura["catalogati"].sum())
        st.metric("Slot presenti nello store", f"{presenti:,} / {catalogati:,}")
        st.caption(
            "Il catalogo copre l'intero periodo configurato. La differenza sono i mesi "
            "scaricati ma non ancora ingeriti, oppure non ancora scaricati."
        )
        st.bar_chart(copertura.to_pandas().set_index("mese")["presenti"], height=260)
        with st.expander("Dettaglio per mese"):
            st.dataframe(copertura, width="stretch", hide_index=True)

    st.subheader("Struttura dei fold")
    fold_tab = struttura_fold(config)
    if fold_tab is None:
        st.info("Tabella dei fold non ancora prodotta.")
    else:
        st.dataframe(fold_tab, width="stretch", hide_index=True)
        st.caption(
            "Un campione entra in un blocco solo se **l'intera finestra** input piu' "
            "target ci sta dentro: nessun target di train puo' comparire fra gli "
            "input di validazione. Il distacco fra blocchi e' un margine aggiuntivo "
            "contro l'autocorrelazione, non la difesa principale."
        )


# --------------------------------------------------------------------------- #
# Modello
# --------------------------------------------------------------------------- #

elif sezione == "Modello":
    st.header("Modello")
    info = informazioni_modello(config, int(fold))
    if not info["disponibile"]:
        st.warning(
            f"Nessun checkpoint in `{info['percorso']}`. "
            f"Eseguire `scripts/train_model.py --fold {fold}`."
        )
    else:
        metadati = info["metadati"]

        problemi = coerenza_artefatti(config, int(fold))
        if problemi:
            st.error(
                "**Artefatti incoerenti.** Il modello si carica lo stesso, ma non e' "
                "detto che sia quello che si crede di avere:\n\n"
                + "\n".join(f"- {p}" for p in problemi)
            )

        parametri = info.get("n_parametri")
        colonne = st.columns(4)
        colonne[0].metric("Variante", info["variante"])
        colonne[1].metric("Parametri", f"{parametri:,}" if parametri else "non leggibili")
        colonne[2].metric("Canali in ingresso", info.get("canali_ingresso") or "?")
        colonne[3].metric("Ancoraggio diurno", "attivo" if info["ancoraggio"] else "spento")

        migliore = metadati.get("val_loss")
        if migliore is not None:
            st.caption(
                f"Checkpoint salvato all'epoca {metadati.get('epoch', '?')} "
                f"con perdita di validazione {migliore:.4f}."
            )

        colonne = st.columns(4)
        colonne[0].metric("Canali base", info["canali_base"])
        colonne[1].metric("Profondita'", info["profondita"])
        colonne[2].metric("Blocchi per livello", info["blocchi_per_livello"])
        colonne[3].metric("Aggiornato", info["aggiornato"].strftime("%Y-%m-%d %H:%M"))

        st.subheader("Architettura")
        st.markdown(
            "Encoder-decoder a U completamente convoluzionale: si addestra su ritagli e "
            "si applica alla griglia intera. Produce **un solo tensore**, la cui "
            "mappatura su (variabile, componente, scadenza) e' dichiarata in "
            "`OutputLayout` invece di essere indicizzata a mano, perche' scambiare media "
            "e log-varianza non farebbe fallire nulla: produrrebbe solo previsioni "
            "sbagliate."
        )
        st.markdown(
            "Con l'ancoraggio attivo la rete prevede lo **scarto** rispetto "
            "all'osservazione piu' recente alla stessa ora del bersaglio. Solo la media "
            "viene traslata: la log-varianza descrive l'incertezza dello scarto."
        )

        curva = curva_apprendimento(info)
        if curva is not None and curva.height:
            st.subheader("Curva di apprendimento")
            st.line_chart(
                curva.to_pandas().set_index("epoch")[["train_loss", "val_loss"]],
                height=300,
            )
            migliore = curva.sort("val_loss").head(1)
            st.caption(
                f"Migliore epoca {int(migliore['epoch'][0])} con perdita di validazione "
                f"{float(migliore['val_loss'][0]):.4f}."
            )


# --------------------------------------------------------------------------- #
# Prestazioni
# --------------------------------------------------------------------------- #

elif sezione == "Prestazioni":
    st.header("Prestazioni")
    split = st.selectbox("Blocco", ["test", "val"], index=0)
    tabella = metriche(config, int(fold), split)
    if tabella is None or not tabella.height:
        st.warning(
            f"Nessuna metrica per il fold {fold}. Eseguire "
            f"`scripts/evaluate_model.py --fold {fold} --split {split}`."
        )
    else:
        st.subheader("Riepilogo")
        st.caption(
            "Il riferimento da battere e' la **persistenza diurna**, cioe' ripetere "
            "l'osservazione di ieri alla stessa ora. Su questo dominio e' un avversario "
            "molto forte, e usare quella ingenua darebbe un vantaggio illusorio."
        )
        st.dataframe(riepilogo_metriche(tabella), width="stretch", hide_index=True)

        ingannevoli = accuratezze_ingannevoli(tabella)
        if ingannevoli.height:
            righe = "\n".join(
                f"- **{r['variable']}**, modello `{r['model']}`: accuratezza "
                f"{r['accuratezza']:.3f}, ma rispondere sempre \"no\" darebbe "
                f"{r['sempre_no']:.3f} (l'evento accade nel "
                f"{r['frequenza_di_base'] * 100:.1f} % dei casi)."
                for r in ingannevoli.iter_rows(named=True)
            )
            st.warning(
                "Nella tabella qui sopra c'e' un'accuratezza piu' bassa di quella di un "
                "modello muto. "
                "Per gli eventi rari va letta insieme alla frequenza di base, oppure "
                "sostituita dal Brier e dall'F1.\n\n" + righe
            )

        st.subheader("Andamento con la scadenza")
        variabili = sorted(tabella["variable"].unique().to_list())
        metriche_disponibili = sorted(tabella["metric"].unique().to_list())
        colonne = st.columns(2)
        variabile = colonne[0].selectbox("Grandezza", variabili)
        metrica = colonne[1].selectbox("Metrica", metriche_disponibili)
        andamento = per_scadenza(tabella, variabile, metrica)
        if andamento.height:
            largo = andamento.pivot(on="model", index="lead_slot", values="value")
            st.line_chart(largo.to_pandas().set_index("lead_slot"), height=320)
        else:
            st.info("Nessun valore per questa combinazione.")

        st.subheader("Guadagno sul non fare nulla")
        st.caption(
            "L'errore assoluto non dice se il modello serve. Qui c'e' la differenza "
            "percentuale rispetto alla migliore persistenza a quella scadenza: zero "
            "significa che ripetere il passato avrebbe dato lo stesso risultato."
        )
        guadagno = guadagno_su_persistenza(tabella, variabile, metrica)
        if guadagno.height:
            st.dataframe(guadagno.to_pandas(), use_container_width=True)
            st.bar_chart(
                guadagno.select("lead_slot", "guadagno_percento")
                .to_pandas()
                .set_index("lead_slot"),
                height=260,
            )
        else:
            st.info("Nessun valore per questa combinazione.")


# --------------------------------------------------------------------------- #
# Confronto visivo
# --------------------------------------------------------------------------- #

elif sezione == "Confronto visivo":
    st.header("Previsto contro osservato")
    colonne = st.columns(2)
    posizione = colonne[0].slider("Finestra di test", 0, 60, 0)
    scadenza = colonne[1].slider(
        "Scadenza (slot)", 0, config.windows.output_slots - 1, 2
    )

    try:
        previsto, osservato, riferimento, istante, errore_medio = confronto_in_cache(
            percorso_config, int(fold), int(posizione), int(scadenza)
        )
    except Exception as errore:
        st.error(f"Impossibile calcolare il confronto: {errore}")
        st.stop()

    testa = f"Istante previsto: **{istante:%Y-%m-%d %H:%M UTC}**" if istante else ""
    st.markdown(f"{testa} - errore assoluto medio **{errore_medio:.2f} degC**")

    limite = float(max(abs(np.nanmin(osservato)), abs(np.nanmax(osservato))))
    figura, assi = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
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
    st.pyplot(figura)
    plt.close(figura)

    if riferimento is not None:
        st.subheader("Contro la persistenza diurna")
        errore_modello = float(np.mean(np.abs(differenza)))
        errore_riferimento = float(np.mean(np.abs(riferimento - osservato)))
        colonne = st.columns(2)
        colonne[0].metric("Errore del modello", f"{errore_modello:.2f} degC")
        colonne[1].metric(
            "Errore della persistenza diurna",
            f"{errore_riferimento:.2f} degC",
            f"{errore_riferimento - errore_modello:+.2f} a favore del modello",
        )


# --------------------------------------------------------------------------- #
# Mappa degli errori
# --------------------------------------------------------------------------- #

elif sezione == "Mappa degli errori":
    st.header("Dove si concentra l'errore")
    colonne = st.columns(2)
    n_finestre = colonne[0].slider("Finestre aggregate", 2, 30, 8)
    scadenza = colonne[1].slider(
        "Scadenza (slot)", 0, config.windows.output_slots - 1, 2
    )

    try:
        campo, usate = errori_in_cache(
            percorso_config, int(fold), int(n_finestre), int(scadenza)
        )
    except Exception as errore:
        st.error(f"Impossibile calcolare la mappa: {errore}")
        st.stop()

    st.caption(
        f"Errore quadratico medio per cella su {usate} finestre di test. "
        f"Una metrica scalare non puo' dire **dove** si sbaglia: un errore concentrato "
        f"sui rilievi ha cause diverse da uno diffuso sull'oceano."
    )

    figura, asse = plt.subplots(figsize=(11, 6), constrained_layout=True)
    immagine = mappa(asse, campo, config, titolo="RMSE per cella (degC)",
                     cmap="inferno", vmin=0.0,
                     vmax=float(np.nanpercentile(campo, 99)))
    # Vigo di Cadore, il punto di interesse dichiarato del progetto.
    asse.plot(12.5, 46.5, marker="o", markersize=7, markerfacecolor="none",
              markeredgecolor="cyan", markeredgewidth=1.8)
    asse.annotate("Vigo di Cadore", (12.5, 46.5), textcoords="offset points",
                  xytext=(9, 5), color="cyan", fontsize=9)
    figura.colorbar(immagine, ax=asse, shrink=0.85, label="degC")
    st.pyplot(figura)
    plt.close(figura)

    colonne = st.columns(4)
    colonne[0].metric("Errore medio", f"{float(np.mean(campo)):.2f} degC")
    colonne[1].metric("Mediano", f"{float(np.median(campo)):.2f} degC")
    colonne[2].metric("Massimo", f"{float(np.max(campo)):.2f} degC")
    riga = round((config.region.north - 46.5) / config.region.grid_step)
    colonna = round((12.5 - config.region.west) / config.region.grid_step)
    if 0 <= riga < campo.shape[0] and 0 <= colonna < campo.shape[1]:
        colonne[3].metric("Su Vigo", f"{float(campo[riga, colonna]):.2f} degC")


# --------------------------------------------------------------------------- #
# Clima
# --------------------------------------------------------------------------- #

elif sezione == "Clima":
    st.header("Andamenti climatici")

    copertura_ = copertura_in_cache(percorso_config)
    giudizio = giudizio_sulla_serie(copertura_)
    if copertura_ is None or copertura_.n_anni_completi < 2:
        st.warning(giudizio)
    else:
        st.info(giudizio)

    if copertura_ is None:
        st.stop()

    colonne = st.columns(4)
    colonne[0].metric("Anni presenti", len(copertura_.anni))
    colonne[1].metric("Anni completi", copertura_.n_anni_completi)
    colonne[2].metric("Slot", f"{copertura_.slot:,}")
    colonne[3].metric("Periodo", f"{copertura_.primo[:7]} .. {copertura_.ultimo[:7]}")

    st.caption(
        "Tutte le medie spaziali sono **pesate per l'area della cella**: su una griglia "
        "in latitudine e longitudine le celle a nord coprono molto meno terreno, e una "
        "media aritmetica darebbe loro un peso che non hanno."
    )

    st.subheader("Media mensile sul dominio")
    mensili = mensili_in_cache(percorso_config)
    if mensili is None:
        st.warning("Nessuna serie disponibile: ingerire almeno un mese.")
    else:
        st.line_chart(
            mensili.select("periodo", "media").to_pandas().set_index("periodo"),
            height=280,
        )

        st.subheader("Lo stesso mese in anni diversi")
        st.caption(
            "E' il confronto onesto su una serie corta: toglie di mezzo il ciclo "
            "stagionale, che e' di gran lunga il segnale piu' forte, e lascia vedere "
            "la differenza fra annate."
        )
        interannuale = confronto_interannuale(config)
        if interannuale is None:
            st.info("Nessun mese si ripete ancora in due anni diversi.")
        else:
            largo = interannuale.pivot(on="year", index="month", values="media")
            st.dataframe(largo, width="stretch", hide_index=True)

        st.subheader("Ciclo stagionale")
        stagionale = ciclo_stagionale(config)
        if stagionale is not None:
            st.caption(
                "Media per mese dell'anno. La colonna `anni` dice su quante annate e' "
                "calcolata ciascuna riga: dove vale uno, non e' una media, e' un'annata."
            )
            st.dataframe(stagionale, width="stretch", hide_index=True)

    st.subheader("Tendenza annuale")
    tendenza = tendenza_annuale(config)
    if tendenza is None:
        st.warning(
            "Meno di due anni completi: non viene calcolata alcuna pendenza. "
            "Una retta su un punto solo non esiste, e su due punti non ha incertezza "
            "stimabile."
        )
    else:
        colonne = st.columns(3)
        colonne[0].metric("Pendenza", f"{tendenza.pendenza:+.3f} degC/anno")
        basso, alto = tendenza.intervallo
        colonne[1].metric(
            "Intervallo 95%",
            "non stimabile" if not np.isfinite(basso) else f"{basso:+.3f} .. {alto:+.3f}",
        )
        colonne[2].metric("Anni", tendenza.n_anni)
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
        nomi = {m: a for m, a in confrontabili}
        colonne = st.columns(3)
        mese = colonne[0].selectbox("Mese", sorted(nomi), format_func=lambda m: f"{m:02d}")
        anni = nomi[mese]
        anno_a = colonne[1].selectbox("Annata di riferimento", anni, index=0)
        anno_b = colonne[2].selectbox("Annata da confrontare", anni, index=len(anni) - 1)
        if anno_a == anno_b:
            st.info("Scegliere due annate diverse.")
        else:
            differenza = mappa_differenza_mensile(config, mese, anno_a, anno_b)
            if differenza is None:
                st.warning("Dati insufficienti per una delle due annate.")
            else:
                estremo = float(np.nanmax(np.abs(differenza)))
                figura, asse = plt.subplots(figsize=(7, 4.5))
                immagine = mappa(
                    asse, differenza, config,
                    titolo=f"{mese:02d}/{anno_b} meno {mese:02d}/{anno_a} (degC)",
                    cmap="RdBu_r", vmin=-estremo, vmax=estremo,
                )
                figura.colorbar(immagine, ax=asse, shrink=0.8)
                st.pyplot(figura, width="stretch")
                plt.close(figura)
                st.caption(
                    f"Media sul dominio: {float(np.nanmean(differenza)):+.2f} degC. "
                    "La scala e' centrata sullo zero, cosi' il colore non suggerisce un "
                    "segno che i dati non hanno. Una sola coppia di annate mostra "
                    "**variabilita' del tempo atmosferico**, non una tendenza."
                )

    st.subheader("Giornate caratteristiche a Vigo di Cadore")
    st.caption(
        "Gli indici climatici standard usano minimo e massimo giornalieri. Qui il giorno "
        "ha tre osservazioni, quindi il gelo si conta sullo slot delle 06 UTC, il piu' "
        "vicino al minimo mattutino. Sono approssimazioni dichiarate. La colonna `slot` "
        "dice quanti dati ha ciascun anno: gli anni parziali non sono confrontabili con "
        "quelli completi."
    )
    estremi = estremi_in_cache(percorso_config)
    if estremi is None:
        st.info("Nessun dato per gli indici.")
    else:
        st.dataframe(estremi, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Addestramento
# --------------------------------------------------------------------------- #

elif sezione == "Addestramento":
    st.header("Addestramento")
    st.caption(
        "Da qui si vede che cosa entra nella rete, che cosa ne esce, e si avviano "
        "esecuzioni con parametri diversi. Ogni esecuzione e' un **processo separato** "
        "con la sua cartella: sopravvive alla chiusura di questa pagina e resta "
        "ripetibile da terminale con la configurazione che trova salvata dentro."
    )

    ingressi, uscite, nuova, corso = st.tabs(
        ["Ingressi", "Uscite", "Nuova esecuzione", "Esecuzioni"]
    )

    # ---------------------------------------------------------------- ingressi
    with ingressi:
        catalogo = catalogo_ingressi(config)
        st.subheader(f"{catalogo.height} canali di ingresso")
        conteggi = catalogo["group"].value_counts().sort("count", descending=True)
        st.dataframe(conteggi, hide_index=True, width="stretch")
        st.caption(
            "Ogni canale e' un campo sull'intera griglia. Lo **stato** e' la variabile a "
            "un dato ritardo, la **tendenza** e' la differenza fra due istanti, i canali "
            "statici e temporali non dipendono dalla finestra."
        )

        with st.expander("Elenco completo dei canali"):
            filtro = st.multiselect(
                "Gruppi", sorted(catalogo["group"].unique().to_list()), default=[]
            )
            mostrato = catalogo.filter(catalogo["group"].is_in(filtro)) if filtro else catalogo
            st.dataframe(mostrato, hide_index=True, width="stretch", height=300)

        st.subheader("Un canale sulla griglia")
        colonne = st.columns(3)
        split_ingresso = colonne[0].selectbox("Blocco", ["test", "val", "train"], index=0)
        posizione_ingresso = colonne[1].number_input(
            "Finestra", min_value=0, max_value=200, value=0, step=1, key="finestra_ingresso"
        )
        canale_scelto = colonne[2].selectbox(
            "Canale",
            list(range(catalogo.height)),
            format_func=lambda i: f"{i} - {catalogo['name'][i]}",
        )

        # Il risultato non puo' dipendere dal valore *transitorio* del pulsante: alla
        # prima riesecuzione della pagina, provocata da qualsiasi altro widget, il
        # pulsante torna falso e il campo appena calcolato sparirebbe senza spiegazione.
        if st.button("Mostra il canale", key="mostra_canale"):
            st.session_state["canale_richiesto"] = True
        if st.session_state.get("canale_richiesto"):
            try:
                nome, gruppo, variabile, ritardo, campo, unita, avvertenza = ingresso_in_cache(
                    percorso_config, int(fold), split_ingresso,
                    int(posizione_ingresso), int(canale_scelto),
                )
            except Exception as errore:
                st.error(f"Canale non ispezionabile: {errore}")
            else:
                st.markdown(
                    f"**{nome}** - gruppo *{gruppo}*, variabile `{variabile}`, "
                    f"ritardo {ritardo} slot, unita' **{unita}**"
                )
                if avvertenza:
                    st.warning(avvertenza)
                figura, asse = plt.subplots(figsize=(7, 5))
                immagine = mappa(
                    asse, campo, config,
                    titolo=f"{nome} [{unita}]",
                    cmap="RdBu_r" if gruppo == "tendency" else "viridis",
                )
                figura.colorbar(immagine, ax=asse, shrink=0.8)
                st.pyplot(figura)
                plt.close(figura)
                riepilogo = {
                    "minimo": float(np.min(campo)), "medio": float(np.mean(campo)),
                    "massimo": float(np.max(campo)),
                }
                colonne_r = st.columns(3)
                for colonna, (etichetta, valore) in zip(
                    colonne_r, riepilogo.items(), strict=False
                ):
                    colonna.metric(etichetta, f"{valore:.2f} {unita}")
                st.caption(
                    "Il campo e' quello che la rete riceve davvero, ricostruito con la "
                    "stessa pipeline e riportato indietro alle unita' di partenza."
                )

    # ------------------------------------------------------------------ uscite
    with uscite:
        st.subheader("Tutte e nove le scadenze")
        st.caption(
            "Una scadenza sola nasconde il difetto piu' comune di un modello ancorato: "
            "errore piccolo sulla prima e crescente sulle successive."
        )
        colonne = st.columns(3)
        split_uscita = colonne[0].selectbox(
            "Blocco", ["test", "val", "train"], index=0, key="split_uscita"
        )
        posizione_uscita = colonne[1].number_input(
            "Finestra", min_value=0, max_value=200, value=0, step=1, key="finestra_uscita"
        )
        variabile_uscita = colonne[2].selectbox("Variabile", ["t2m"], index=0)

        if st.button("Esegui il modello", key="esegui_uscite"):
            st.session_state["uscite_richieste"] = True
        if st.session_state.get("uscite_richieste"):
            try:
                tabella = uscite_in_cache(
                    percorso_config, int(fold), split_uscita,
                    int(posizione_uscita), variabile_uscita,
                )
            except Exception as errore:
                st.error(f"Esecuzione non riuscita: {errore}")
            else:
                st.dataframe(tabella, hide_index=True, width="stretch")
                figura, asse = plt.subplots(figsize=(8, 3.5))
                asse.plot(
                    tabella["scadenza"], tabella["radice_errore_quadratico"],
                    marker="o", label="RMSE",
                )
                asse.plot(
                    tabella["scadenza"], tabella["errore_assoluto"],
                    marker="s", label="errore assoluto medio",
                )
                asse.axhline(0.0, color="grey", linewidth=0.6)
                asse.set_xlabel("scadenza (slot di 6 o 12 ore)")
                asse.set_ylabel("degC")
                asse.legend(fontsize=8)
                asse.grid(alpha=0.3)
                st.pyplot(figura)
                plt.close(figura)
                st.caption(
                    "Le medie sono spaziali sull'intero dominio: un errore medio vicino a "
                    "zero con un errore assoluto grande significa che gli errori si "
                    "compensano fra regioni, non che la previsione sia buona."
                )

    # -------------------------------------------------------- nuova esecuzione
    with nuova:
        st.subheader("Avvia un addestramento")
        st.caption(
            "I parametri passano per lo **stesso schema** che valida la configurazione "
            "del progetto: un valore fuori intervallo viene rifiutato qui, prima che il "
            "processo parta, non a meta' addestramento."
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
            "Lato del ritaglio", min_value=32, max_value=256,
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
                "Senza ancoraggio il banco misura 6.61 degC contro 3.65: e' la scelta "
                "con l'effetto piu' grande di tutte quelle provate."
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

        stima = epoche * campioni / max(int(config.training.batch_size), 1)
        st.caption(
            f"Circa {int(stima)} passi di ottimizzazione. Sul portatile usato per lo "
            "sviluppo un'epoca da 512 campioni con ritaglio 96 richiede circa 7 minuti."
        )

        if st.button("Avvia", type="primary"):
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
                    "Segui l'andamento nella scheda Esecuzioni."
                )

    # ------------------------------------------------------------- esecuzioni
    with corso:
        st.subheader("Esecuzioni")
        elenco = runs.elenca(config)
        if not elenco:
            st.info("Nessuna esecuzione avviata da questa pagina.")
        else:
            st.dataframe(runs.confronto(config), hide_index=True, width="stretch")

            scelta = st.selectbox("Dettaglio", [e.nome for e in elenco])
            dettaglio = runs.leggi(config, scelta)
            if dettaglio is not None:
                colonne = st.columns(4)
                colonne[0].metric("Stato", dettaglio.stato)
                colonne[1].metric("Avviata", dettaglio.avviata.strftime("%d/%m %H:%M"))
                colonne[2].metric("Processo", str(dettaglio.pid or "-"))
                storia = runs.cronologia(dettaglio, int(dettaglio.parametri.get("fold", 0)))
                colonne[3].metric("Epoche concluse", str(len(storia)))

                premuto = dettaglio.attiva and st.button("Ferma", key=f"ferma_{scelta}")
                if premuto and runs.ferma(config, scelta):
                    st.warning("Richiesta di arresto inviata.")
                    st.rerun()

                if storia:
                    figura, asse = plt.subplots(figsize=(8, 3.5))
                    asse.plot(
                        [r["epoch"] for r in storia],
                        [r["train_loss"] for r in storia], marker="o", label="train",
                    )
                    asse.plot(
                        [r["epoch"] for r in storia],
                        [r["val_loss"] for r in storia], marker="s", label="validazione",
                    )
                    asse.set_xlabel("epoca")
                    asse.set_ylabel("perdita")
                    asse.legend(fontsize=8)
                    asse.grid(alpha=0.3)
                    st.pyplot(figura)
                    plt.close(figura)

                st.markdown("**Ultime righe del diario**")
                st.code(runs.coda_del_diario(dettaglio, righe=30) or "(ancora vuoto)")
                st.caption(f"Comando: `{' '.join(dettaglio.comando)}`")

            if st.button("Aggiorna"):
                st.rerun()


# --------------------------------------------------------------------------- #
# Risorse
# --------------------------------------------------------------------------- #

elif sezione == "Risorse":
    st.header("Uso di CPU e memoria")
    if st.button("Aggiorna"):
        st.cache_data.clear()
    stato = risorse()

    colonne = st.columns(4)
    colonne[0].metric("CPU totale", f"{stato['cpu_totale']:.0f} %")
    colonne[1].metric("Core logici", stato["core"])
    colonne[2].metric(
        "Memoria",
        f"{stato['memoria_usata_gb']:.1f} / {stato['memoria_totale_gb']:.1f} GB",
    )
    colonne[3].metric("Memoria in uso", f"{stato['memoria_percento']:.0f} %")

    if stato["cpu_per_core"]:
        st.subheader("Carico per core")
        st.bar_chart(
            {"percento": stato["cpu_per_core"]},
            height=200,
        )

    st.subheader("Processi del progetto")
    if not stato["processi"]:
        st.info("Nessun processo del progetto in esecuzione.")
    else:
        st.dataframe(stato["processi"], width="stretch", hide_index=True)
        st.caption(
            "Sono elencati solo i processi il cui comando cita il progetto: "
            "addestramento, banchi, scaricamento e ingestione."
        )
