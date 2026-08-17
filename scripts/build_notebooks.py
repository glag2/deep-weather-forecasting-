"""Genera i due notebook a partire da questo sorgente.

I notebook sono artefatti: scriverli a mano in JSON e' fragile e produce diff
illeggibili. Qui il contenuto e' codice Python normale, revisionabile, e il notebook
viene rigenerato quando cambia. Le celle non vengono eseguite: gli output nascono
sulla macchina di chi li apre.

Uso:
    python scripts/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT_ROOT / "notebooks"


def markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text.strip())


def notebook(cells: list[nbformat.NotebookNode]) -> nbformat.NotebookNode:
    documento = nbformat.v4.new_notebook(cells=cells)
    documento.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    return documento


PREAMBOLO = """
import sys
from pathlib import Path

# Il notebook puo' essere aperto dalla cartella `notebooks/`: senza questo, l'import
# di `dwf` fallisce a seconda di dove e' stato avviato Jupyter.
RADICE = Path.cwd()
if not (RADICE / "src").exists():
    RADICE = RADICE.parent
sys.path.insert(0, str(RADICE / "src"))

from dwf.config import Config

config = Config.load(RADICE / "configs" / "default.yaml", project_root=RADICE)
print(f"periodo   : {config.time.start} .. {config.time.end}")
print(f"dominio   : {config.region.n_lat} x {config.region.n_lon}")
print(f"finestre  : {config.windows.input_slots} slot in ingresso -> "
      f"{config.windows.output_slots} previsti")
print(f"dati      : {config.paths.data_root}")
"""


def training_notebook() -> nbformat.NotebookNode:
    return notebook([
        markdown("""
# Addestramento

Questo notebook addestra la rete su un fold della validazione a finestra mobile e ne
misura la qualita' contro la persistenza.

Prerequisiti: i GRIB scaricati con `scripts/download_era5.py` e ingeriti con
`scripts/ingest_era5.py`. La cella di stato piu' sotto dice se ci sono dati a
sufficienza.

**Perche' a fold e non con una singola divisione.** Il periodo disponibile copre poco
piu' di due anni. Una divisione cronologica unica metterebbe nel test una sola
stagione, e il punteggio direbbe soprattutto in che mesi e' caduto il test. La
finestra mobile produce sei fold i cui blocchi di test coprono tutti i dodici mesi.
"""),
        code(PREAMBOLO),
        markdown("""
## 1. Stato dei dati

Ogni fold ha bisogno di finestre ammesse sia in train sia in validazione. Una finestra
e' ammessa solo se tutti i suoi 30 slot sono stati ingeriti e non contengono valori non
finiti: bastano pochi mesi mancanti per lasciare un fold senza dati.
"""),
        code("""
import polars as pl

from dwf.data.dataset import sample_starts

righe = []
for indice in range(len(config.build_folds())):
    righe.append({
        "fold": indice,
        "train": len(sample_starts(config, indice, "train")),
        "val": len(sample_starts(config, indice, "val")),
        "test": len(sample_starts(config, indice, "test")),
    })
disponibilita = pl.DataFrame(righe).with_columns(
    addestrabile=(pl.col("train") > 0) & (pl.col("val") > 0)
)
disponibilita
"""),
        code("""
addestrabili = disponibilita.filter("addestrabile").get_column("fold").to_list()
if not addestrabili:
    raise RuntimeError(
        "Nessun fold ha insieme finestre di train e di validazione: "
        "scaricare e ingerire altri mesi prima di proseguire."
    )
FOLD = addestrabili[0]
print(f"fold usato in questo notebook: {FOLD}")
"""),
        markdown("""
## 2. Che cosa entra nella rete

L'ordine dei canali e' un contratto: scambiarne due non fa fallire nulla e produce solo
associazioni sbagliate. E' dichiarato una volta in `dwf.data.features` e tutto il resto
lo legge da li'.
"""),
        code("""
from dwf.data.features import InputLayout
from dwf.models.heads import OutputLayout

input_layout = InputLayout.from_config(config)
output_layout = OutputLayout.from_targets(config.targets, config.windows.output_slots)

print(f"canali in ingresso: {input_layout.n_channels}")
print(f"canali in uscita  : {output_layout.total_channels}")
input_layout.to_table().group_by("group").len().sort("group")
"""),
        code("""
# Le teste probabilistiche: la rete non prevede un numero ma una distribuzione, ed e'
# questo che rende misurabile l'affidabilita'.
pl.DataFrame(output_layout.describe())
"""),
        markdown("""
## 3. Addestramento

La normalizzazione viene calcolata **solo** sugli slot di train del fold. Calcolarla su
tutti i dati farebbe entrare nel modello informazione dal futuro e la validazione
perderebbe significato.

Il costo misurato su questa macchina e' di circa 0,6 s per campione su ritagli 96x96.
Conviene partire con poche epoche per verificare che la perdita scenda, e allungare poi.
"""),
        code("""
from dwf.train import train_fold

EPOCHE = 5  # alzare dopo aver verificato che la perdita scende

esito = train_fold(config, FOLD, epochs=EPOCHE)
print(f"\\nmigliore epoca: {esito.best_epoch}, validazione {esito.best_val_loss:.4f}")
print(f"checkpoint: {esito.checkpoint}")
"""),
        code("""
import matplotlib.pyplot as plt

epoche = [record.epoch for record in esito.history]
figura, assi = plt.subplots(1, 2, figsize=(12, 4))

assi[0].plot(epoche, [r.train_loss for r in esito.history], label="train")
assi[0].plot(epoche, [r.val_loss for r in esito.history], label="validazione")
assi[0].set_xlabel("epoca"); assi[0].set_ylabel("perdita"); assi[0].legend()
assi[0].set_title("Perdita totale")

# Le componenti separate dicono quale testa non sta imparando: senza, un miglioramento
# della temperatura potrebbe mascherare una precipitazione ferma.
for nome in esito.history[-1].components:
    assi[1].plot(epoche, [r.components.get(nome, float("nan")) for r in esito.history],
                 label=nome)
assi[1].set_xlabel("epoca"); assi[1].legend(); assi[1].set_title("Componenti (train)")
plt.tight_layout(); plt.show()
"""),
        markdown("""
## 4. Il modello batte una previsione banale?

Un errore assoluto non dice nulla da solo: 2 K possono essere ottimi a tre giorni e
pessimi a sei ore. Il confronto con la persistenza, cioe' "domani come oggi", e' il
minimo sindacale.
"""),
        code("""
from dwf.data.dataset import WeatherWindowDataset, build_reader
from dwf.evaluate import collect_predictions, metrics_table, persistence_baseline
from dwf.train import load_checkpoint

rete, stats, input_layout, output_layout = load_checkpoint(config, FOLD)
lettore = build_reader(config, input_layout)
finestre_val = sample_starts(config, FOLD, "val")

dataset_val = WeatherWindowDataset(
    config, input_layout, stats, finestre_val, lettore,
    crop_size=None, crops_per_window=1, seed=config.training.seed,
)

MAX_FINESTRE = 8  # alzare per una stima piu' stabile, al costo di tempo
previsioni = collect_predictions(rete, dataset_val, output_layout, config,
                                 max_windows=MAX_FINESTRE)
riferimento = persistence_baseline(dataset_val, max_windows=MAX_FINESTRE)

metriche = pl.concat([
    metrics_table(previsioni, stats, model="dwf", split="val", fold=FOLD),
    metrics_table(riferimento, stats, model="persistence", split="val", fold=FOLD),
])
confronto = (
    metriche.filter((pl.col("lead_slot") >= 0) & (pl.col("month") == -1))
    .pivot(on="model", index=["variable", "metric", "lead_slot"], values="value")
    .sort("variable", "metric", "lead_slot")
)
confronto
"""),
        code("""
rmse = confronto.filter((pl.col("variable") == "t2m") & (pl.col("metric") == "rmse_kelvin"))
plt.figure(figsize=(7, 4))
plt.plot(rmse["lead_slot"], rmse["dwf"], marker="o", label="modello")
plt.plot(rmse["lead_slot"], rmse["persistence"], marker="s", label="persistenza")
plt.xlabel("scadenza (slot di 6 ore)"); plt.ylabel("RMSE [K]")
plt.title("Temperatura a 2 m: modello contro persistenza")
plt.legend(); plt.grid(alpha=0.3); plt.show()
"""),
        markdown("""
## 5. Affidabilita'

Se il modello dice 30 % di pioggia, deve piovere nel 30 % dei casi in cui lo dice. Il
diagramma di affidabilita' verifica esattamente questo: i punti sulla diagonale sono
previsioni calibrate, sotto la diagonale sono eccessi di sicurezza.
"""),
        code("""
from dwf.evaluate import reliability_table

affidabilita = reliability_table(
    previsioni.tp_probability, previsioni.tp_occurrence,
    model="dwf", split="val", fold=FOLD, variable="tp",
)
validi = affidabilita.filter(pl.col("count") > 20)

plt.figure(figsize=(5.5, 5.5))
plt.plot([0, 1], [0, 1], "k--", label="calibrazione perfetta")
plt.plot(validi["forecast_mean"], validi["observed_frequency"], marker="o", label="modello")
plt.xlabel("probabilita' prevista"); plt.ylabel("frequenza osservata")
plt.title("Affidabilita' della probabilita' di pioggia")
plt.legend(); plt.grid(alpha=0.3); plt.show()
validi
"""),
        markdown("""
## 6. Prossimi passi

- Alzare `EPOCHE` finche' la validazione smette di migliorare.
- Addestrare gli altri fold: `for fold in addestrabili: train_fold(config, fold)`.
- Ingerire altri mesi per riempire i fold ancora vuoti e coprire tutte le stagioni.
"""),
    ])


def inference_notebook() -> nbformat.NotebookNode:
    return notebook([
        markdown("""
# Previsione

Carica un modello addestrato, verifica che i dati siano aggiornati, e produce la
previsione a tre giorni sull'intero dominio.

**Che tipo di previsione e'.** ERA5 pubblica con circa sei giorni di ritardo, quindi
l'ultima finestra disponibile non finisce oggi. Cio' che si ottiene e' una previsione
su giorni gia' trascorsi, confrontabile con l'osservato: un limite della sorgente, non
del modello, ed e' anche cio' che permette di verificarne l'onesta'.
"""),
        code(PREAMBOLO),
        markdown("""
## 1. I dati sono aggiornati?

Prima di prevedere bisogna sapere fino a quando arrivano i dati. La frontiera non si
indovina: il catalogo del CDS la dichiara, e la si legge da li'. Se il catalogo non
risponde si ricade su una stima prudente, dichiarata come tale.

**Il mese in corso e' sempre parziale.** ERA5 e' una rianalisi, non una previsione:
esce con alcuni giorni di ritardo, quindi l'ultimo mese si ferma a meta'. Un file che
copre mezzo mese non e' rotto, e non va riscaricato a ogni esecuzione: va riscaricato
solo quando ERA5 ha pubblicato altri giorni. Il confronto fra i giorni gia' presenti e
quelli ora disponibili e' esattamente cio' che decide.
"""),
        code("""
from dwf.data.freshness import query_availability, summarize_freshness

disponibilita = query_availability()
print(disponibilita.describe())

stato = summarize_freshness(config, disponibilita)
da_aggiornare = stato.filter("needs_download")
print(f"\\nmesi attesi: {stato.height} | da aggiornare: {da_aggiornare.height} | "
      f"parziali: {int(stato.get_column('partial').sum())}")
da_aggiornare if da_aggiornare.height else stato.tail(5)
"""),
        markdown("""
### Aggiornamento

La cella seguente scarica e ingerisce cio' che manca. E' l'operazione lunga del
notebook: circa nove minuti e 390 MB per mese, quindi conviene tenere `MAX_MESI` basso
la prima volta e rilanciare, invece di avviare ore di scaricamento alla cieca.

Serve la credenziale CDS nel file `.env`. Se manca, la cella si ferma con un messaggio
esplicito e il resto del notebook funziona comunque sui dati gia' presenti.
"""),
        code("""
from dwf.data.refresh import refresh_data

AGGIORNA = False   # portare a True per scaricare davvero
MAX_MESI = 1       # quanti mesi aggiornare per esecuzione

rapporto = refresh_data(
    config,
    availability=disponibilita,
    download=AGGIORNA,
    max_months=MAX_MESI,
)
"""),
        code("""
import numpy as np

from dwf.tables import SLOTS, read_table

catalogo = read_table(SLOTS, config.tables_dir).sort("slot_index")
utilizzabili = catalogo.get_column("usable").to_numpy()
ultimo = catalogo.filter("usable").tail(1)
print(f"slot utilizzabili: {int(utilizzabili.sum())} su {len(utilizzabili)}")
print(f"ultimo istante disponibile: {ultimo.get_column('valid_time').item()}")
"""),
        markdown("""
## 2. Il modello

Il checkpoint contiene i pesi e il numero di canali attesi; le statistiche di
normalizzazione stanno accanto. Se la configurazione e' cambiata dopo l'addestramento,
il caricamento fallisce invece di produrre previsioni senza senso.
"""),
        code("""
from dwf.train import load_checkpoint

FOLD = 0
rete, stats, input_layout, output_layout = load_checkpoint(config, FOLD)
print(f"canali attesi: {input_layout.n_channels}")
print(f"parametri    : {rete.n_parameters:,}")
print(f"normalizzazione calcolata su: {stats.computed_on_split}")
"""),
        markdown("""
## 3. La previsione

Un solo passaggio della rete produce tutti e nove gli slot previsti: non c'e'
ricorsione, quindi non c'e' accumulo di errore da un passo al successivo.
"""),
        code("""
from dwf.data.dataset import build_reader
from dwf.predict import latest_usable_start, predict_window, summarize

lettore = build_reader(config, input_layout)
inizio = latest_usable_start(config, utilizzabili)

previsione = predict_window(
    config, rete, input_layout, output_layout, stats, lettore, inizio
)
print(f"ultimo istante osservato: {previsione.init_time}")
summarize(previsione)
"""),
        markdown("""
## 4. Le mappe

Tre grandezze per ciascuna scadenza: temperatura prevista, probabilita' di
precipitazione e probabilita' che sia neve. L'incertezza sulla temperatura e' una
previsione a sua volta, e va guardata insieme alla media.
"""),
        code("""
import matplotlib.pyplot as plt

estensione = [config.region.west, config.region.east, config.region.south, config.region.north]

def mappa(asse, campo, titolo, cmap, vmin=None, vmax=None):
    immagine = asse.imshow(campo, extent=estensione, origin="upper", cmap=cmap,
                           vmin=vmin, vmax=vmax, aspect="auto")
    asse.set_title(titolo, fontsize=10)
    plt.colorbar(immagine, ax=asse, fraction=0.03)

SCADENZE = [0, 4, 8]  # primo giorno, secondo, terzo
figura, assi = plt.subplots(len(SCADENZE), 3, figsize=(15, 4 * len(SCADENZE)))
for riga, scadenza in enumerate(SCADENZE):
    istante = previsione.valid_times[scadenza]
    mappa(assi[riga, 0], previsione.t2m_mean[scadenza] - 273.15,
          f"Temperatura [C] - {istante:%d/%m %H UTC}", "RdBu_r")
    mappa(assi[riga, 1], previsione.precip_probability[scadenza],
          f"Probabilita' di pioggia - {istante:%d/%m %H UTC}", "Blues", 0, 1)
    mappa(assi[riga, 2], previsione.snow_probability[scadenza],
          f"Probabilita' di neve - {istante:%d/%m %H UTC}", "PuBu", 0, 1)
plt.tight_layout(); plt.show()
"""),
        code("""
# L'incertezza dichiarata dal modello: dove e' alta, la previsione va presa con cautela.
figura, assi = plt.subplots(1, 3, figsize=(15, 4))
for colonna, scadenza in enumerate(SCADENZE):
    mappa(assi[colonna], previsione.t2m_std[scadenza],
          f"Incertezza [K] - +{scadenza} slot", "magma")
plt.tight_layout(); plt.show()
"""),
        markdown("""
## 5. Previsione in un punto

Utile per leggere il risultato come lo leggerebbe una persona: che tempo fa in un
luogo, nei prossimi tre giorni.
"""),
        code("""
LATITUDINE, LONGITUDINE = 45.07, 7.69  # Torino

riga = int(np.abs(previsione.latitudes - LATITUDINE).argmin())
colonna = int(np.abs(previsione.longitudes - LONGITUDINE).argmin())
print(f"punto di griglia: {previsione.latitudes[riga]:.2f} N, "
      f"{previsione.longitudes[colonna]:.2f} E")

import polars as pl

pl.DataFrame({
    "istante": list(previsione.valid_times),
    "temperatura_C": [float(previsione.t2m_mean[s, riga, colonna] - 273.15)
                      for s in range(previsione.n_lead)],
    "incertezza_K": [float(previsione.t2m_std[s, riga, colonna])
                     for s in range(previsione.n_lead)],
    "prob_pioggia": [float(previsione.precip_probability[s, riga, colonna])
                     for s in range(previsione.n_lead)],
    "pioggia_mm": [float(previsione.precip_amount[s, riga, colonna])
                   for s in range(previsione.n_lead)],
    "prob_neve": [float(previsione.snow_probability[s, riga, colonna])
                  for s in range(previsione.n_lead)],
})
"""),
        markdown("""
## 6. Verifica contro l'osservato

Poiche' la previsione riguarda giorni gia' trascorsi, l'osservato puo' esistere: quando
c'e', si misura l'errore davvero commesso invece di limitarsi a guardare le mappe.

Attenzione pero': la previsione qui sopra parte dalla finestra **piu' recente**, e per
costruzione i suoi tre giorni previsti cadono oltre l'ultimo slot ingerito. Per
verificare serve una finestra piu' arretrata, che abbia anche i target.
"""),
        code("""
n_input, n_output = config.windows.input_slots, config.windows.output_slots

def finestra_verificabile(utilizzabili):
    \"\"\"Ultima finestra che ha sia gli input sia i target: solo li' la verifica ha senso.\"\"\"
    for candidato in range(len(utilizzabili) - n_input - n_output, -1, -1):
        if utilizzabili[candidato : candidato + n_input + n_output].all():
            return candidato
    return None

inizio_verifica = finestra_verificabile(utilizzabili)
if inizio_verifica is None:
    print("Nessuna finestra ha insieme input e osservato: ingerire altri mesi.")
elif inizio_verifica == inizio:
    verifica = previsione
    print("La previsione piu' recente e' gia' verificabile.")
else:
    verifica = predict_window(config, rete, input_layout, output_layout, stats,
                              lettore, inizio_verifica)
    print(f"Verifica su una finestra arretrata, inizializzata al {verifica.init_time} "
          f"(la piu' recente non ha ancora l'osservato).")
"""),
        code("""
if inizio_verifica is not None:
    osservato = lettore.read_window(inizio_verifica + n_input, n_output)
    errore = verifica.t2m_mean - osservato["t2m"]
    riepilogo = pl.DataFrame({
        "scadenza": list(range(verifica.n_lead)),
        "istante": list(verifica.valid_times),
        "errore_medio_K": [float(errore[s].mean()) for s in range(verifica.n_lead)],
        "rmse_K": [float(np.sqrt((errore[s] ** 2).mean())) for s in range(verifica.n_lead)],
        "incertezza_dichiarata_K": [float(verifica.t2m_std[s].mean())
                                    for s in range(verifica.n_lead)],
    })
else:
    riepilogo = None
riepilogo
"""),
        markdown("""
Se l'incertezza dichiarata e' molto piu' piccola dell'RMSE effettivo, il modello e'
troppo sicuro di se'; se e' molto piu' grande, e' troppo prudente. Le due colonne
dovrebbero avvicinarsi con l'addestramento.

## 7. Salvare la previsione
"""),
        code("""
from dwf.predict import forecast_to_table
from dwf.tables import FORECAST, write_table

tabella = forecast_to_table(previsione, stride=4)
destinazione = config.artifacts_dir / f"fold_{FOLD:02d}"
destinazione.mkdir(parents=True, exist_ok=True)
print(write_table(tabella, FORECAST, destinazione), f"({tabella.height:,} righe)")
"""),
    ])


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    for nome, costruttore in (
        ("01_training.ipynb", training_notebook),
        ("02_inference.ipynb", inference_notebook),
    ):
        percorso = NOTEBOOKS / nome
        documento = costruttore()
        nbformat.validate(documento)
        nbformat.write(documento, percorso)
        print(f"{percorso} ({len(documento.cells)} celle)")


if __name__ == "__main__":
    main()
