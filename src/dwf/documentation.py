"""Il testo tecnico che accompagna le mappe del report.

Sta qui e non nel generatore del PDF per due ragioni. La prima e' che cosi' e'
verificabile: un test puo' leggere queste sezioni e controllare che i numeri citati
esistano davvero, cosa impossibile con del testo sepolto in una funzione di disegno. La
seconda e' che il report deve essere **uno**: chi riceve la previsione riceve nello stesso
file com'e' fatto il modello che l'ha prodotta, come si riaddestra e come si legge il
risultato, invece di dover cercare un documento a parte che invecchia da solo.

Le sezioni sono dati, non stringhe formattate: l'impaginazione e la larghezza della
colonna appartengono a chi disegna la pagina.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# Un blocco che inizia con questo prefisso va reso a larghezza fissa e non riformattato:
# sono tabelle e comandi, dove l'allineamento e' informazione.
PREFISSO_VERBATIM = "|"


@dataclass(frozen=True, slots=True)
class DocSection:
    """Una sezione del testo tecnico: un titolo e i suoi blocchi."""

    title: str
    body: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("Una sezione senza titolo non e' impaginabile")
        if not self.body:
            raise ValueError(f"Sezione vuota: {self.title!r}")


def _verbatim(righe: list[str]) -> str:
    return "\n".join(f"{PREFISSO_VERBATIM}{riga}" for riga in righe)


ARCHITETTURA = DocSection(
    "Come e' fatto il modello",
    (
        "Rete scritta da zero, senza pesi preaddestrati. La prima versione era una rete a "
        "U convoluzionale; il suo campo recettivo efficace e' stato misurato sul modello "
        "addestrato, derivando l'uscita in un punto rispetto all'ingresso, e il risultato "
        "e' che il 50 per cento dell'influenza su una previsione arriva da un raggio di "
        "soli 130 km, il 90 per cento da 2189 km. Una struttura sinottica alle medie "
        "latitudini viaggia fra 500 e 1000 km al giorno, quindi a tre giorni l'informazione "
        "che decide la previsione parte da 1500-3000 km di distanza. La convoluzione "
        "potrebbe arrivarci impilando strati, e i pesi che ha imparato dicono che di fatto "
        "decide guardando vicino: e' il motivo dell'architettura a nucleo globale, non una "
        "preferenza estetica.",
        "Il percorso completo di un ingresso, blocco per blocco:",
        _verbatim([
            "1. stem            conv 3x3, 245 -> 48 canali, GroupNorm, GELU",
            "                   mescola le variabili prima di qualsiasi riduzione",
            "2. to_tokens       conv 8x8 passo 8: 261x401 -> 33x51 token da 192 canali",
            "                   l'attenzione costa il quadrato dei token, patch 4 costerebbe 16x",
            "3. 4 blocchi       posizione (conv depthwise 3x3) + attenzione a 4 teste su",
            "                   TUTTI i token + percorso denso sui canali, rapporto 2",
            "                   l'attenzione decide DA DOVE, il denso decide COSA FARNE",
            "4. from_tokens     192 -> 48 canali, interpolazione a risoluzione piena",
            "5. merge           conv 3x3 fra il ramo locale (solo stem) e il contesto globale",
            "                   il dettaglio fine passa dalla scorciatoia locale, non dai token",
            "6. output_conv     conv 1x1 -> 45 canali, pesi e bias inizializzati a ZERO",
        ]),
        "I 45 canali di uscita sono 9 scadenze per cinque quantita': media e logaritmo "
        "della varianza della temperatura, logit di occorrenza e quantita' della "
        "precipitazione, logit della frazione nevosa. La rete prevede quindi una "
        "distribuzione, non un numero, ed e' costretta a dichiarare quanto e' sicura.",
        "Due scelte rendono la rete indipendente dalla dimensione dell'ingresso, e vanno "
        "conservate. La posizione e' iniettata da una convoluzione e non da una tabella di "
        "posizioni assolute, che sarebbe legata al numero di token visto in addestramento. "
        "L'ultima convoluzione parte da zero, cosi' due architetture confrontate partono "
        "dalla stessa previsione neutra e la differenza misurata non comprende il punto di "
        "partenza.",
        "Con l'ancoraggio diurno attivo la testa gaussiana non prevede la temperatura ma lo "
        "scarto dalla persistenza diurna: il livello e il ciclo giorno-notte, che sono la "
        "parte facile e dominante del segnale, arrivano gratis, e la capacita' della rete "
        "resta per la parte difficile. La stessa idea applicata alla probabilita' di pioggia "
        "e' stata provata, ha funzionato su scala ridotta ed e' stata rifiutata quando su "
        "scala piena ha smesso di funzionare: la misura che l'ha scartata e' in "
        "docs/OCCURRENCE_ANCHOR.md.",
        _verbatim([
            "parametri          nucleo globale 1.945.245      rete a U 9.979.053",
            "passo, finestra intera  3000 ms                  10499 ms",
            "migliore validazione    0,9755                   1,3031",
        ]),
        "Sono presenti e spenti due rami sperimentali: un attention sink, cioe' un token in "
        "piu' fra chiavi e valori che assorbe peso di attenzione senza produrre un'uscita, e "
        "un ramo a contesto compresso che riduce i token a 9x13, li attende densamente e li "
        "reinietta con una convoluzione inizializzata a zero. Sono spenti perche' non sono "
        "stati misurati a questa scala, e l'iniezione a zero garantisce che accenderli non "
        "sposti il punto di partenza.",
    ),
)

PIPELINE = DocSection(
    "Dalla richiesta ai dati al file di previsione",
    (
        "Ogni passo legge cio' che il precedente ha scritto, e ognuno e' ripartibile: "
        "interrompere non fa danni.",
        _verbatim([
            "1. download_era5.py   un mese per richiesta e per famiglia di variabili, in GRIB.",
            "                      GRIB e non NetCDF perche' e' il formato nativo: la",
            "                      conversione lato server ha prodotto differenze non nostre.",
            "2. ingest_era5.py     GRIB -> store Zarr indicizzato per slot + indice Parquet.",
            "                      Uno slot entra solo se completo e finito; pioggia e neve",
            "                      sono deaccumulate sulla finestra di 8 ore che precede.",
            "3. campi invarianti   quota, maschera terra-mare e altri: store separato, letti",
            "                      come piani 2D. Indicizzarli per slot era un difetto reale.",
            "4. piano dei canali   nome, gruppo, variabile, ritardo, trasformazione. Salvato",
            "                      con i pesi: un ordine diverso da' numeri plausibili e falsi.",
            "5. finestre e fold    21 slot in ingresso, 9 in uscita. Sei fold a finestra",
            "                      mobile, blocchi di test su tutti i dodici mesi, guardia di",
            "                      30 slot perche' l'ingresso di un test non tocchi il train.",
            "6. normalizzazione    media e scarto dal SOLO blocco di addestramento del fold;",
            "                      il blocco di provenienza e' scritto nelle statistiche.",
            "7. train_model.py     pesi, metadati (architettura compresa) e cronologia.",
            "8. evaluate_model.py  metriche, calibrazione delle probabilita', soglie, e il",
            "                      confronto con le due persistenze.",
            "9. predict_forecast.py  9 scadenze in un passaggio, sul dominio intero.",
        ]),
        "I 245 canali di ingresso sono 189 di stato (9 variabili per 21 slot), 27 di "
        "tendenza (differenze a 1, 3 e 9 slot, cioe' la derivata temporale che distingue un "
        "fronte in arrivo da una situazione stabile), 21 di modulo del vento, 4 di tempo "
        "(seno e coseno di ora e giorno dell'anno, perche' la periodicita' va data in forma "
        "continua), 2 statici e 2 di latitudine. Altri dodici campi invarianti sono "
        "scaricati e verificati ma restano spenti finche' un confronto non dimostra che "
        "servono; fra questi la profondita' dei laghi e' inutilizzabile grezza, perche' fuori "
        "dai laghi contiene valori di riempimento.",
    ),
)

ADDESTRAMENTO = DocSection(
    "Come si riaddestra, e le manopole che contano",
    (
        _verbatim([
            "uv sync --extra notebooks",
            "uv run python scripts/check_cds_access.py",
            "uv run python scripts/download_era5.py        # ore, non minuti",
            "uv run python scripts/ingest_era5.py",
            "uv run python scripts/train_model.py --fold 0",
            "uv run python scripts/evaluate_model.py --fold 0 --split test",
            "uv run python scripts/predict_forecast.py --fold 0",
            "uv run python scripts/report_forecast.py --fold 0   # questo documento",
        ]),
        "L'addestramento scrive pesi, metadati e cronologia in models/fold_00 e si rifiuta "
        "di sovrascrivere una cartella occupata: e' una protezione aggiunta dopo aver perso "
        "due punti di controllo, non una precauzione teorica. I metadati contengono "
        "l'architettura e il piano dei canali, e il caricamento verifica che corrispondano.",
        _verbatim([
            "crop_size: null      finestra intera 261x401, non ritagli 96x96.",
            "                     Il ritaglio addestrava dentro un orizzonte artificiale:",
            "                     oltre il bordo non c'era nulla da guardare, quindi la rete",
            "                     non poteva imparare a usare informazione che in previsione",
            "                     le viene data. Costo: 3000 ms per passo contro 800 per",
            "                     quattro ritagli 96, cioe' 28,7 contro 21,7 secondi per",
            "                     milione di punti previsti. Il dominio intero e' un terzo",
            "                     MENO efficiente per punto, e si paga comunque.",
            "batch_size: 1        conseguenza della finestra intera.",
            "channels_last: true  misurato 8% piu' veloce su questa CPU. Nella stessa prova",
            "                     bfloat16 e' risultato DANNOSO, 0,50x: scritto nella",
            "                     configurazione perche' l'attesa contraria era ragionevole.",
            "optimizer: adamw     CMuon e' implementato e spento: ortogonalizzare costa",
            "                     107 ms contro 42, il passo va da 235 a 319 ms, quindi va",
            "                     confrontato a pari TEMPO, non a pari epoche.",
            "samples_per_epoch    e' la manopola da alzare: finora l'addestramento vedeva",
            "                     0,94 passate sui dati e 2,7 visite per finestra.",
        ]),
        "La validazione usa ritagli deterministici. Prima non era cosi', e la scelta "
        "dell'epoca migliore conteneva rumore quanto la differenza che si voleva misurare: "
        "due reti diverse crollavano entrambe esattamente all'epoca 9.",
    ),
)

LETTURA = DocSection(
    "Come si leggono questi risultati",
    (
        "Un errore assoluto non dice nulla. Due gradi sono un disastro in un pomeriggio "
        "stabile di luglio e un risultato eccellente durante il passaggio di un fronte. "
        "Serve un avversario, e l'avversario giusto non e' ovvio: poiche' le uscite sono "
        "alle 06, 12 e 18 UTC, le scadenze distano 6, 6 e 12 ore, quindi ripetere l'ultimo "
        "dato confronta mezzanotte con mezzogiorno ed e' un avversario finto. Il riferimento "
        "vero e' la persistenza diurna, ripetere ieri alla stessa ora: a 24 ore fa 2,40 "
        "gradi, dove quella ingenua ne fa 5,48. Chi si confronta con la seconda si convince "
        "di avere un buon modello.",
        "L'ordine di lettura, e non e' arbitrario:",
        _verbatim([
            "1. gli artefatti sono coerenti?   pesi e cronologia della stessa corsa;",
            "                                  se no, tutto il resto e' aria.",
            "2. il guadagno sulla persistenza  dove non e' positivo, il modello non serve.",
            "   diurna e' positivo a ogni",
            "   scadenza?",
            "3. la validazione era ancora in   allora il numero non e' un limite del modello",
            "   discesa alla fine?             ma del tempo di calcolo speso.",
            "4. l'incertezza dichiarata e'     dispersione diviso errore deve valere 1, e la",
            "   credibile?                     copertura al 90% deve valere 0,90. Sotto 1 il",
            "                                  modello e' troppo sicuro, ed e' il difetto",
            "                                  peggiore: induce a fidarsi quando non si deve.",
            "5. le probabilita' mantengono la  se annuncia il 40%, deve piovere in 4 casi su",
            "   promessa?                      10. Attenzione: l'accuratezza su un evento",
            "                                  raro inganna, perche' con pioggia nel 20% dei",
            "                                  casi rispondere sempre no da' l'80%. Per",
            "                                  questo si guardano Brier e F1.",
            "6. il campo previsto ha la        se e' piu' piatto, il modello sta comprando",
            "   variabilita' dell'osservato?   errore quadratico con la sfumatura: metrica",
            "                                  buona, previsione inutile.",
        ]),
        "Il notebook notebooks/03_collaudo.ipynb esegue questi sei controlli su un fold "
        "addestrato e spiega ogni grafico mentre lo produce.",
    ),
)

STATO = DocSection(
    "Che cosa vale il modello, e cosa manca",
    (
        "Il modello batte la persistenza diurna a tutte e nove le scadenze, ma il margine "
        "non e' quello che serve: a 24 ore fa 2,27 gradi contro 2,40, cioe' un guadagno del "
        "5,5 per cento dove ne servirebbe circa il 17 per stare sotto i 2 gradi richiesti. "
        "Il difetto non e' che il modello fa 2,27 gradi: e' che 2,40 gradi sono gratis. A 12 "
        "ore il guadagno e' del 23,2 per cento e a 24 crolla al 5,5, cioe' la rete sfrutta "
        "la continuita' a brevissimo termine e poi smette di aggiungere informazione.",
        "Tre cause misurate, che vanno affrontate insieme perche' nessuna spiega il "
        "risultato da sola:",
        _verbatim([
            "poco addestramento     2560 passi, 0,94 passate sui dati, 2,7 visite per finestra",
            "nessuna quota          tutti gli ingressi sono di superficie; i livelli di",
            "                       pressione sono scaricati (2,2 GB) e non ancora ingeriti",
            "portata spaziale       50% dell'influenza entro 130 km, contro i 500-1000 km al",
            "                       giorno di una struttura sinottica",
        ]),
        "Sulla pioggia il modello vince dove conta: punteggio di Brier 0,192 contro 0,274 "
        "della persistenza diurna. Sulla neve perde in F1 e conviene dirlo: prevede neve "
        "troppo spesso, recupera l'82 per cento dei casi ma solo il 35 per cento delle sue "
        "segnalazioni e' corretto. La sua accuratezza sulla neve non va letta come un "
        "risultato, perche' la neve compare nel 9 per cento dei casi e rispondere sempre no "
        "darebbe il 90,9 per cento.",
        "Difetti chiusi che hanno lasciato una protezione permanente: contaminazione fra "
        "blocchi (finestre rienumerate dai confini e guardia di 30 slot), campi invarianti "
        "indicizzati per slot, due punti di controllo sovrascritti (architettura nei metadati "
        "e verifica al caricamento), validazione decisa dal caso (ritagli deterministici), "
        "avanzamento delle scadenze che assumeva slot equidistanti. Il principio che tiene "
        "insieme tutto: i dati sono veri, e se qualcosa non torna la spiegazione va cercata "
        "prima nell'aspettativa che nel dato. La neve maggiore della pioggia era "
        "l'impacchettamento intero del formato GRIB, non un errore dell'archivio.",
    ),
)

SEZIONI_TECNICHE = (ARCHITETTURA, PIPELINE, ADDESTRAMENTO, LETTURA, STATO)


def skill_lines(metrics: pl.DataFrame, *, variable: str = "t2m") -> list[str]:
    """Tabella del guadagno per scadenza, dai numeri misurati e non da valori scritti a mano.

    Se la tabella delle metriche non contiene ne' il modello ne' una persistenza, il
    risultato lo dichiara: un report che inventa una riga vuota e' peggio di uno che
    ammette di non avere la misura.
    """
    richieste = {"model", "metric", "variable", "lead_slot", "month", "value"}
    mancanti = richieste - set(metrics.columns)
    if mancanti:
        raise ValueError(f"Tabella delle metriche incompleta: mancano {sorted(mancanti)}")

    scelta = metrics.filter(
        (pl.col("variable") == variable)
        & (pl.col("metric") == "rmse_celsius")
        & (pl.col("lead_slot") >= 0)
        & (pl.col("month") == -1)
    )
    if scelta.is_empty():
        return ["nessuna metrica per scadenza: eseguire scripts/evaluate_model.py"]

    largo = scelta.pivot(on="model", index="lead_slot", values="value").sort("lead_slot")
    riferimenti = [c for c in largo.columns if c.startswith("persistence")]
    if "dwf" not in largo.columns or not riferimenti:
        return ["confronto impossibile: manca il modello o la persistenza nella tabella"]

    righe = [f"{'scadenza':>8}  {'modello':>9}  {'ieri stessa ora':>15}  {'guadagno':>9}"]
    righe.append("-" * len(righe[0]))
    for riga in largo.iter_rows(named=True):
        modello = riga["dwf"]
        riferimento = min(
            (riga[nome] for nome in riferimenti if riga[nome] is not None), default=None
        )
        if modello is None or riferimento is None:
            continue
        guadagno = 100.0 * (riferimento - modello) / riferimento
        righe.append(
            f"{riga['lead_slot']:>8}  {modello:>9.3f}  {riferimento:>15.3f}  "
            f"{guadagno:>8.1f}%"
        )
    return righe


def measured_section(metrics: pl.DataFrame | None) -> DocSection:
    """La sezione con i numeri del fold, costruita al momento della scrittura del report."""
    if metrics is None:
        corpo = (
            "Nessuna tabella di metriche disponibile per questo fold: eseguire "
            "scripts/evaluate_model.py prima del report. Questa pagina resta volutamente "
            "senza numeri, invece di riportare quelli di un'altra corsa.",
        )
    else:
        corpo = (
            "Errore quadratico medio della temperatura in gradi Celsius, per scadenza, "
            "contro la migliore delle due persistenze. Sono i numeri della tabella prodotta "
            "dall'ultima valutazione di questo fold, letti al momento di scrivere il report.",
            _verbatim(skill_lines(metrics)),
            "Il guadagno e' minimo alle scadenze multiple di 24 ore, dove la persistenza "
            "diurna coincide con quella semplice ed e' quindi al massimo della sua forza.",
        )
    return DocSection("I numeri di questo fold", corpo)


__all__ = [
    "ADDESTRAMENTO",
    "ARCHITETTURA",
    "LETTURA",
    "PIPELINE",
    "PREFISSO_VERBATIM",
    "SEZIONI_TECNICHE",
    "STATO",
    "DocSection",
    "measured_section",
    "skill_lines",
]
