# Previsione meteo con deep learning su rianalisi ERA5

Previsione dei **3 giorni successivi** (mattina, mezzogiorno, sera) sull'area
euro-atlantica, a partire dai **7 giorni precedenti** di rianalisi ERA5, con una rete
convoluzionale scritta da zero.

Variabili previste: **temperatura**, **precipitazione**, **neve**, ciascuna con la
propria **incertezza** calibrata.

![Esempio di campo ERA5: temperatura a 2 m](image/README/1730719737389.png)

## Dominio

| | |
|---|---|
| Area | lat 10 N - 75 N, lon 40 W - 60 E |
| Risoluzione | 0.25 gradi (nativa ERA5) |
| Griglia | **261 x 401 = 104.661 punti** |
| Slot giornalieri | 06, 12, 18 UTC |
| Input | 21 slot (7 giorni) |
| Output | 9 slot (3 giorni) |

Dataset: [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)

## Requisiti

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) per la gestione dell'ambiente
- Un account Copernicus CDS (gratuito)
- Nessuna GPU necessaria: il training e' pensato per CPU

## Installazione

```bash
uv sync --extra notebooks
```

Su connessioni lente il download dei wheel piu' grossi (`torch`, `polars`, `scipy`)
puo' superare il timeout di rete predefinito di `uv`, che e' di 30 secondi. In quel
caso:

```bash
# Linux / macOS
UV_HTTP_TIMEOUT=600 uv sync --extra notebooks
```

```powershell
# Windows PowerShell
$env:UV_HTTP_TIMEOUT=600; uv sync --extra notebooks
```

## Credenziali CDS

Il progetto legge le credenziali da variabili d'ambiente, con lo stesso ordine di
precedenza usato da `cdsapi`: prima l'ambiente, poi `~/.cdsapirc`.

**Variabili richieste:**

| Variabile | Valore |
|---|---|
| `CDSAPI_URL` | `https://cds.climate.copernicus.eu/api` |
| `CDSAPI_KEY` | il proprio Personal Access Token |

**Procedura:**

1. Registrarsi su <https://cds.climate.copernicus.eu> e autenticarsi.
2. Aprire <https://cds.climate.copernicus.eu/how-to-api>: la pagina mostra il proprio
   Personal Access Token.
3. Accettare i *Terms of Use* del dataset. Passaggio separato e facile da dimenticare:
   senza di esso ogni richiesta API fallisce anche con un token valido. Si trova in
   fondo al form nella scheda *Download* di
   <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>,
   oppure si puo' accettare via API con `--accept-licences` (vedi sotto).
4. Creare nella radice del progetto un file `.env` con le due variabili:

   ```
   CDSAPI_URL=https://cds.climate.copernicus.eu/api
   CDSAPI_KEY=il-proprio-token
   ```

`.env` e' escluso dal versioning. Non va committato, ne' incollato in chat, log o
notebook: se una chiave esce dal proprio archivio va considerata compromessa e
rigenerata dal profilo CDS.

> Su Windows, Notepad e `Set-Content -Encoding utf8` di PowerShell 5.1 scrivono un
> BOM UTF-8 in testa al file. Il progetto lo gestisce leggendo con `utf-8-sig`, ma
> molti altri strumenti no.

### Verifica dell'accesso

Prima di accodare decine di richieste conviene controllare che tutto sia a posto. Lo
script distingue i tre motivi per cui un download fallisce, che altrimenti si
confondono in un unico errore HTTP:

```bash
uv run python scripts/check_cds_access.py
uv run python scripts/check_cds_access.py --accept-licences
```

Riporta anche **l'ultima data ERA5 effettivamente disponibile**, cercandola a ritroso
da oggi invece di assumere una latenza fissa.

## Configurazione

Tutto e' dichiarato in [`configs/default.yaml`](configs/default.yaml) e validato a
runtime: area allineata alla griglia, variabili di tipo coerente, target presenti tra
le variabili scaricate, percorsi confinati sotto la cartella dati.

### Validazione a finestra mobile

Il modello non viene valutato su un unico blocco finale. Un test contiguo cadrebbe
tutto nella coda del periodo, che e' estiva: la neve non sarebbe misurabile e la
temperatura verrebbe valutata su un solo regime meteorologico.

La suddivisione usa quindi la **rolling origin validation**: l'origine avanza nel
tempo e ogni fold ha il proprio train, validation e test, sempre in quest'ordine
cronologico. Con la configurazione di riferimento entrano 6 fold i cui blocchi di test
coprono **tutti i dodici mesi**. Il costo e' che il training va ripetuto per ogni
fold; `split.n_folds` permette di limitarli durante lo sviluppo, e
`split.mode: chronological` torna allo split a blocco unico.

## Struttura

```
configs/default.yaml       configurazione di riferimento
src/dwf/
  variables.py             registro variabili ERA5 (nome CDS <-> short name GRIB)
  slots.py                 slot temporali, finestre di accumulo, split senza leakage
  config.py                configurazione validata
  credentials.py           credenziali CDS, senza mai esporne il valore
  tables.py                layer Polars/Parquet con schemi verificati
  data/                    scarico, ingestione, feature, dataset
  models/                  rete convoluzionale e teste probabilistiche
scripts/
  check_cds_access.py      diagnosi di accesso al CDS
  download_era5.py         scarico del periodo configurato, ripartibile
  benchmark_model.py       costo del modello su CPU
tests/                     suite pytest
datasets/                  (ignorata da git) GRIB, Zarr, tabelle, artefatti
```

La radice dei dati e' `datasets/` e non `data/`: su Windows `data` verrebbe risolto
nella cartella `Data/` gia' presente nel repository, mescolando decine di GB generati
ai file tracciati, mentre su Linux e in Docker resterebbe una cartella distinta. Il
nome e' configurabile con `paths.data_root`.

## Scarico dei dati

```bash
uv run python scripts/download_era5.py --dry-run   # mostra il piano, non invia nulla
uv run python scripts/download_era5.py --limit 2   # un solo mese, per misurare
uv run python scripts/download_era5.py             # tutto il periodo
```

Il download e' **ripartibile**: i file gia' presenti e non vuoti vengono saltati, e
ogni richiesta scrive su un file `.partial` rinominato solo a scaricamento completato,
cosi' un'interruzione non lascia un GRIB troncato che sembrerebbe valido. Le richieste
sono sequenziali perche' il CDS limita quelle concorrenti per utente.

I dati sono organizzati su due livelli: **Zarr** per i tensori numerici, su cui il
training fa accesso casuale a finestre spaziotemporali, e **Polars/Parquet** come
registro dei dati puliti (catalogo degli slot, controlli qualita', statistiche di
normalizzazione, metriche, calibrazione, previsione finale).

## Come funziona la pipeline

Ogni passo legge quello che il precedente ha scritto. Si possono eseguire singolarmente.

| # | Comando | Cosa fa |
|---|---|---|
| 1 | `scripts/check_cds_access.py` | Verifica token, licenze e ultima data ERA5 disponibile. |
| 2 | `scripts/download_era5.py` | Scarica i GRIB mese per mese in `datasets/raw/`. Ripartibile. |
| 3 | `scripts/ingest_era5.py` | Converte i GRIB in un unico store Zarr `(slot, 261, 401)` e registra gli slot in Parquet. Deaccumula pioggia e neve. |
| 4 | `scripts/analyze_data.py` | Analisi esplorativa dello store: copertura, distribuzioni, prevedibilita'. Scrive `docs/DATA_ANALYSIS.md`. |
| 5 | `scripts/screen_features.py` | Misura quali famiglie di canali aiutano a prevedere il **cambiamento**. Scrive `docs/FEATURES.md`. |
| 6 | `scripts/screen_input_days.py` | Confronta 3, 7, 10, 14 giorni di storico. Scrive `docs/INPUT_DAYS.md`. |
| 7 | `scripts/compare_variants.py` | Confronta le cinque architetture a parita' di protocollo. Scrive `docs/VARIANTS.md`. |
| 8 | `scripts/train_model.py --fold 0` | Addestra un fold. Salva pesi e statistiche in `models/fold_00/`. |
| 9 | `scripts/evaluate_model.py --fold 0 --split test` | Metriche sul test, calibrazione delle probabilita', scelta delle soglie, confronto con le persistenze. |
| 10 | `scripts/predict_forecast.py --fold 0` | Previsione a 3 giorni sull'intera griglia, in Parquet. |
| 11 | `scripts/report_forecast.py --fold 0` | Report PDF di 8 pagine con le mappe. |
| 12 | `scripts/refresh_data.py` | Scarica e ingerisce solo cio' che manca, per aggiornare senza rifare tutto. |

In mezzo, i dati passano da queste forme:

```
GRIB mensili -> store Zarr (slot x 261 x 401) -> finestra di 21 slot
   -> 245 canali (stato, tendenze, vento, sole, termodinamica, statici)
   -> rete -> 45 canali di uscita -> 9 slot previsti x 4 grandezze + incertezza
   -> calibrazione -> Parquet -> PDF
```

## Prestazioni

Modello valutato sul blocco di **test** del fold 0, mai usato ne' per addestrare ne'
per scegliere le soglie. Il riferimento e' la **persistenza diurna** (ripetere ieri alla
stessa ora), che su questo dominio e' un avversario molto forte.

| Grandezza | Metrica | Modello | Persistenza diurna | Persistenza ingenua |
|---|---|---:|---:|---:|
| Temperatura | RMSE (degC) | **2.92** | 3.16 | 4.73 |
| Pioggia si/no | F1 | **0.635** | 0.609 | 0.624 |
| Pioggia si/no | Accuratezza | 0.709 | 0.726 | **0.739** |
| Neve si/no | F1 | 0.493 | 0.561 | **0.590** |
| Neve si/no | Accuratezza | 0.847 | 0.919 | **0.925** |
| Pioggia e neve | F1 macro | 0.564 | 0.585 | **0.607** |

241 finestre di test, 9 scadenze ciascuna, dominio intero. Ottenuti con
`scripts/evaluate_model.py --fold 0 --split test`, che scrive `metrics.parquet`.

**Come leggerla.** Sulla temperatura il modello batte la persistenza diurna a **tutte e
nove le scadenze**, e il vantaggio non e' concentrato sulle prime: 2.02 contro 2.43 degC
a sei ore, 3.40 contro 3.68 a tre giorni.

Sulla neve **perde**, e conviene dire perche' invece di nasconderlo. Il modello prevede
neve troppo spesso: recupera l'82 % dei casi contro il 60 % della persistenza, ma solo
il 35 % delle sue segnalazioni e' corretto contro il 55 %. La soglia di decisione e'
scelta sulla validazione, dove rende F1 0.568; sul test scende a 0.493. Cambiarla
guardando il test sposterebbe il compromesso, ma sarebbe barare.

**L'accuratezza sulla neve non va letta come un risultato.** La neve compare nel 9 % dei
casi, quindi rispondere sempre "no" darebbe 90.9 %: entrambe le persistenze, al 92.5 %,
superano di poco quella soglia banale. E' la ragione per cui la tabella riporta anche
F1, che una risposta costante non puo' gonfiare.

Sulla **qualita' della probabilita'**, che e' cio' che serve per decidere, il modello
vince ovunque, neve compresa: punteggio di Brier 0.181 contro 0.274 sulla pioggia e
0.066 contro 0.081 sulla neve, con errore di calibrazione 0.053 contro 0.274.

**F1 macro** e' la media dei due F1 binari (pioggia e neve). Un F1 unico su tutto il
modello non avrebbe senso: la temperatura e' continua e non ha una nozione di
"positivo".

## Notebook

Generati da `scripts/build_notebooks.py`, quindi non vanno modificati a mano.

| Notebook | A cosa risponde |
|---|---|
| `notebooks/01_training.ipynb` | Addestra un fold e mostra perche' la validazione e' a finestra mobile. |
| `notebooks/02_inference.ipynb` | Produce una previsione e la legge sulle mappe. |
| `notebooks/03_collaudo.ipynb` | **Collauda un modello addestrato**: coerenza degli artefatti, guadagno per scadenza, controllo dell'obiettivo dei 2 gradi a 24 ore, taratura dell'incertezza, affidabilita' delle probabilita', mappa dell'errore, e il controllo della sfumatura. Ogni sezione spiega come si legge, comprese le trappole. |

## Sviluppo

```bash
uv run pytest tests -q
uv run ruff check src tests scripts
uv run python scripts/build_notebooks.py
```

## Documentazione

Le relazioni sono in `docs/`, e tutte tranne la prima sono **rigenerate da uno
script**: non vanno modificate a mano, perche' la prossima esecuzione sovrascrive.

| File | Cosa contiene | Chi lo produce |
|---|---|---|
| [`docs/PIANO.md`](docs/PIANO.md) | Piano di lavoro, con lo stato veritiero di ogni voce | a mano |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Stato dei lavori, decisioni e problemi aperti | a mano |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | Stato dell'arte letto e cosa se ne e' preso | a mano |
| [`docs/INGESTION.md`](docs/INGESTION.md) | Come i GRIB diventano Zarr, e le anomalie trovate | a mano |
| [`docs/DATA_ANALYSIS.md`](docs/DATA_ANALYSIS.md) | Analisi del dataset ingerito | `scripts/analyze_data.py` |
| [`docs/VARIANTS.md`](docs/VARIANTS.md) | Confronto fra varianti di rete | `scripts/compare_variants.py` |
| [`docs/INPUT_DAYS.md`](docs/INPUT_DAYS.md) | Quanti giorni di storico in ingresso | `scripts/screen_input_days.py` |
| [`docs/OCCURRENCE_ANCHOR.md`](docs/OCCURRENCE_ANCHOR.md) | Ancoraggio della probabilita' di pioggia | `scripts/screen_occurrence_anchor.py` |
| [`docs/FEATURES.md`](docs/FEATURES.md) | Quali famiglie di canali aiutano a prevedere il cambiamento | `scripts/screen_features.py` |
| [`docs/NEARTIME.md`](docs/NEARTIME.md) | Latenza reale di ERA5 e sorgenti per il quasi tempo reale | a mano |

## Quanto vale il modello, oggi

L'errore assoluto non dice se un modello serve. Il confronto che lo dice e' contro la
persistenza diurna, cioe' l'ipotesi "domani come ieri alla stessa ora", che non costa
nulla. Sullo split di test, temperatura a 2 metri:

| ore avanti | modello | ieri stessa ora | guadagno |
|---|---|---|---|
| +12 | 1,87 C | 2,43 C | +23 % |
| +24 | 2,27 C | 2,40 C | **+5,5 %** |
| +48 | 2,99 C | 3,24 C | +7,6 % |
| +72 | 3,32 C | 3,68 C | +9,8 % |

A ventiquattro ore il modello guadagna il cinque per cento sul non fare nulla. Cio' che
ha imparato e' il ciclo giornaliero, che gli era gia' dato dall'ancoraggio, piu' un
lisciamento locale; la dinamica non c'e'. Le cause misurate sono in
[docs/PROGRESS.md](docs/PROGRESS.md) sezione 14: campo recettivo efficace di 130 km,
nessuna variabile in quota, 2,7 visite per finestra in tutto l'addestramento.

## Limiti noti

- **ERA5 ha 5-6 giorni di latenza.** Una previsione avviata dagli ultimi dati
  disponibili riguarda quindi giorni gia' trascorsi: e' un hindcast verificabile,
  utile per validare, non una previsione operativa. Per il tempo reale servirebbe una
  sorgente diversa, come <https://data.ecmwf.int>.
- **Il modello e' poco addestrato, e si sa di quanto.** Con le impostazioni usate finora
  l'addestramento vedeva 2560 passi, cioe' 0,94 passate sui dati e 2,7 visite per
  finestra. Un modello che ha guardato ogni esempio meno di tre volte non ha ancora un
  limite proprio: ha un limite di tempo di calcolo.
- **L'addestramento e' sulla finestra intera, non su ritagli.** Il ritaglio 96x96 rendeva
  il training molto piu' economico, ma insegnava alla rete a decidere dentro un orizzonte
  che in previsione non esiste. Costo misurato del cambio: 3000 ms per passo sul dominio
  intero contro 800 ms per quattro ritagli, cioe' 28,7 contro 21,7 secondi per milione di
  punti previsti.
- **L'incertezza dichiarata e' misurabile solo dalle valutazioni recenti.** Le tabelle
  prodotte prima dell'introduzione delle metriche `spread_skill_ratio` e `coverage_90` non
  le contengono; basta rieseguire `scripts/evaluate_model.py`.

## Licenza

Vedi [LICENSE](LICENSE). I dati ERA5 sono soggetti alla licenza Copernicus.
