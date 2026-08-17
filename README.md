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
| 4 | `scripts/analyze_data.py` | Analisi esplorativa dello store: copertura, distribuzioni, prevedibilita'. Scrive `DATA_ANALYSIS.md`. |
| 5 | `scripts/screen_features.py` | Misura quali famiglie di canali aiutano a prevedere il **cambiamento**. Scrive `FEATURES.md`. |
| 6 | `scripts/screen_input_days.py` | Confronta 3, 7, 10, 14 giorni di storico. Scrive `INPUT_DAYS.md`. |
| 7 | `scripts/compare_variants.py` | Confronta le cinque architetture a parita' di protocollo. Scrive `VARIANTS.md`. |
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

| Grandezza | Metrica | Modello | Persistenza diurna |
|---|---|---:|---:|
| Temperatura | RMSE (degC) | _da valutazione_ | _da valutazione_ |
| Pioggia si/no | Accuratezza | _da valutazione_ | _da valutazione_ |
| Pioggia si/no | F1 | _da valutazione_ | _da valutazione_ |
| Neve si/no | F1 | _da valutazione_ | _da valutazione_ |
| Pioggia e neve | F1 macro | _da valutazione_ | _da valutazione_ |

I valori si ottengono con `scripts/evaluate_model.py --fold 0 --split test`, che scrive
`metrics.parquet`. La tabella e' volutamente vuota finche' non e' stata eseguita una
valutazione sul modello corrente: riportare numeri di una versione precedente sarebbe
peggio che non riportarne.

**F1 macro** e' la media dei due F1 binari (pioggia e neve). Un F1 unico su tutto il
modello non avrebbe senso: la temperatura e' continua e non ha una nozione di
"positivo".

## Sviluppo

```bash
uv run pytest tests -q
uv run ruff check src tests scripts
```

Lo stato dei lavori, le decisioni prese con le relative motivazioni e i problemi
aperti sono in [`PROGRESS.md`](PROGRESS.md).

## Limiti noti

- **ERA5 ha 5-6 giorni di latenza.** Una previsione avviata dagli ultimi dati
  disponibili riguarda quindi giorni gia' trascorsi: e' un hindcast verificabile,
  utile per validare, non una previsione operativa. Per il tempo reale servirebbe una
  sorgente diversa, come <https://data.ecmwf.int>.
- Il training su CPU e' possibile solo grazie all'addestramento su crop spaziali: la
  rete e' completamente convoluzionale e viene poi applicata al dominio intero.

## Licenza

Vedi [LICENSE](LICENSE). I dati ERA5 sono soggetti alla licenza Copernicus.
