# Stato dello sviluppo

Documento di tracciamento, aggiornato a ogni fase completata. Registra decisioni
prese, verifiche effettivamente eseguite e problemi aperti.

- **Branch di lavoro**: `feature/era5-forecasting-pipeline`
- **Ultimo aggiornamento**: 2026-08-17
- **Fase corrente**: costruzione della pipeline dati (nessun dato reale ancora scaricato)
- **Stato verifiche**: 275 test superati in 9 s, `ruff` senza rilievi
- **Accesso CDS**: verificato end-to-end (vedi 5.1)

### Ambiente misurato

| | |
|---|---|
| CPU | 4 core fisici / 8 logici (torch usa 4 thread) |
| RAM | 15,8 GB |
| GPU | assente |
| Python | 3.12.10, venv gestito da `uv` 0.12.5 |
| torch | 2.13.0**+cpu** (nessun binario CUDA) |

---

## 1. Obiettivo

Prevedere i **3 giorni successivi** (slot mattina / mezzogiorno / sera) sull'intera
area euro-atlantica, a partire dai **7 giorni precedenti** di rianalisi ERA5, con un
modello convoluzionale **scritto da zero** (nessun modello preaddestrato).

Variabili di interesse finale: **temperatura**, **precipitazione**, **neve** e
**affidabilita'** della previsione (quest'ultima come uscita probabilistica calibrata,
non come dichiarazione qualitativa).

## 2. Decisioni prese

| Tema | Decisione | Motivazione |
|---|---|---|
| Area | lat 10-75 N, lon 40 W-60 E, 0.25 gradi, **261 x 401** | Ricavata dagli output salvati in `EDA.ipynb`, coincide con `numberOfPoints: 104661` |
| Risoluzione | nativa 0.25 gradi, `coarsen` configurabile | Richiesta esplicita dell'utente; su CPU e' sostenibile solo con training a crop |
| Fonte dati | ERA5 via CDS API (GRIB) | Continuita' con `Data/era5-request.py` |
| Periodo | **2024-01-01 .. 2026-08-11** (32 mesi, 2862 slot) | Richiesta dell'utente; `end` non e' scelto a mano ma e' `end_datetime` dei metadati della collection CDS |
| Validazione | **finestra mobile (rolling origin), 6 fold** | Uno split contiguo lascia al test solo la coda estiva; vedi 2.1 |
| Slot giornalieri | 06, 12, 18 UTC | Corrispondono a mattina / mezzogiorno / sera |
| Cumulate | finestra di 8 h centrata sullo slot | Copre 20 h su 24 con 4 conteggi ridondanti; nessuna finestra sconfina dal giorno, quindi l'ingestione resta mensile |
| Storage numerico | Zarr + dask, **solo layer di ingestione** | Accesso casuale a finestre spaziotemporali senza caricare tutto in RAM |
| Storage tabellare | **Polars / Parquet**, layer dati principale | Richiesta dell'utente; `slots.parquet` e' il registro autorevole degli slot |
| Config | pydantic con validazione runtime | Impedisce che download, training e inferenza divergano sui parametri condivisi |
| Training | CPU, patch-based su crop 96 x 96 | Nessuna GPU disponibile; la rete e' completamente convoluzionale e in inferenza si applica a 261 x 401 |
| Packaging | `uv` + `pyproject.toml` + Docker | Richiesta dell'utente |

### 2.1 Perche' la validazione e' a finestra mobile

I dati contengono **tre inverni** (gen-mar 2024, dic 2024-mar 2025, gen-mar 2026). Il
problema non era la mancanza di stagione fredda ma la **geometria dello split**: tre
blocchi contigui assorbono tutti gli inverni in train e validation e lasciano al test
la sola coda del periodo, misurata come aprile-agosto 2026, cioe' **0 slot in mesi
nevosi**. La testa neve non sarebbe stata valutabile e la temperatura sarebbe stata
misurata su un solo regime.

Adottata la **rolling origin validation**: l'origine avanza di `step_days` a ogni
fold, quindi i blocchi di test scorrono nel tempo, e dentro ogni fold l'ordine
train -> val -> test resta rispettato (nessuna valutazione su dati precedenti
all'addestramento). Con `initial_train_days=330`, `val_days=60`, `test_days=90`,
`step_days=90` entrano **6 fold** e i test coprono **tutti i 12 mesi**, mesi nevosi
inclusi. Verificato in `tests/test_config.py`.

Costo: il training va ripetuto per ogni fold. Con `expanding: true` il train cresce e
usa tutta la storia disponibile; `mode: chronological` resta selezionabile da config
per una valutazione a blocco unico.

### 2.2 Strategia per la neve

Tre scelte, tutte pensate per il fatto che la neve e' un evento raro e stagionale:

1. **`snow_depth` come predittore** (non target): rappresenta lo stato del manto
   nevoso, che condiziona la temperatura tramite albedo e fusione e determina la
   persistenza della neve al suolo. Segnale utile anche fuori dai mesi nevosi in
   quota.
2. **Testa `sf` come `fraction_of tp`**: "nevica invece di piovere". Combinata con la
   probabilita' di precipitazione della testa `hurdle` da' "neve si/no" come
   probabilita', non come soglia arbitraria.
3. **Metriche come Brier Skill Score contro la climatologia**, con la frequenza di
   base riportata accanto e stratificazione per mese. Un Brier grezzo su un evento
   raro premia il modello che prevede sempre "no neve": senza confronto con la
   climatologia il numero non e' interpretabile.

### Perche' non Polars per il tensore

L'utente preferiva Polars come storage principale. Misurato il costo: il tensore
`(tempo, lat, lon, variabile)` in forma tabellare sono **229 milioni di righe**. Il
problema non e' la dimensione (~8-10 GB in Parquet) ma il pattern di accesso del
training: ogni campione richiede un crop 96 x 96 su 30 slot consecutivi, che in
Parquet costa una scansione di 3,1 M righe per tenerne 276k (**~11x di
amplificazione**), piu' un `pivot` e un `reshape` per campione, con la correttezza
dipendente dall'ordinamento delle righe. E' lo stesso approccio di
`ERA5_grib_to_csv` in `EDA.ipynb`, che non e' mai arrivato a termine.

Compromesso adottato: **Zarr per i tensori, Polars per tutto il resto** (catalogo
slot, QC, schema canali, statistiche di normalizzazione, metriche, calibrazione,
previsione finale).

### Teste probabilistiche

| Target | Testa | Perche' |
|---|---|---|
| `t2m` | `gaussian` (media + log-varianza) | Variabile continua e quasi simmetrica; la varianza prevista da' l'incertezza |
| `tp` | `hurdle` (P(> 0.1 mm) + quantita' condizionata) | La precipitazione ha una massa di probabilita' esattamente in zero, che una gaussiana non puo' rappresentare |
| `sf` | `fraction_of` rispetto a `tp` | Modella "nevica invece di piovere" senza prevedere due volte la quantita' totale |

## 3. Struttura del repository

```
pyproject.toml            pacchetto uv, indice torch CPU forzato
uv.lock                   144 pacchetti risolti
configs/default.yaml       configurazione di riferimento
src/dwf/
  variables.py            registro variabili ERA5 (nome CDS <-> short name GRIB)
  slots.py                algebra slot temporali, finestre accumulo, split
  config.py               configurazione validata con pydantic
  credentials.py          credenziali CDS, senza mai esporne il valore
  tables.py               layer Polars/Parquet con schemi verificati
  data/download.py        richieste CDS per mese e famiglia di variabili
  data/ingest.py          GRIB -> Zarr + catalogo Parquet
  models/                 rete convoluzionale e teste probabilistiche
scripts/
  check_cds_access.py     diagnosi di accesso al CDS
  benchmark_model.py      costo del modello su CPU
  download_era5.py        scarica i GRIB, con ripresa e manifest
  ingest_era5.py          ingerisce i mesi disponibili
tests/                    305 test
datasets/                 (ignorato da git) raw GRIB, Zarr, tables, artifacts
INGESTION.md              spiegazione dettagliata della pipeline dati
```

## 4. Cronologia

### Fase 0 - Analisi del repository esistente (completata)

Stato trovato: 8 file tracciati, nessun modello, nessun test, nessun packaging.

Difetti individuati in `EDA.ipynb`:

1. `ERA5_grib_to_csv` itera in Python su `time x lat x lon x variable` con
   `pd.concat` dentro il loop: per un mese globale a 0.25 gradi sono ~7,7 x 10^8
   righe, irrealizzabile per ordini di grandezza.
2. Cella 13 dimensiona l'array con `len(grib.variables)`, che include le coordinate
   (`number`, `step`, `valid_time`), non `data_vars`.
3. Cella 17 usa `lista_di_liste`, mai definita.
4. `check_conversion` costruisce il path CSV da una cartella diversa da quella di
   scrittura, quindi riporta sempre file mancanti.
5. `xr.open_dataset` su GRIB con 39 variabili miste solleva
   `DatasetBuildError: key present and new value is different: key='time'`, visibile
   negli output salvati: i campi *mean rate* hanno un asse temporale diverso dai
   campi di analisi. Il notebook lo aggira solo nei blocchi commentati.

`Data/2023/febbraio_2023.csv` e' vuoto (2 byte): la conversione non e' mai riuscita.

### Fase 1 - Scaffolding e fondamenta (completata)

- `pyproject.toml` con `uv`, indice `pytorch-cpu` esplicito per evitare i wheel CUDA
  da ~2,5 GB nell'immagine Docker.
- `variables.py`: 24 variabili registrate (17 istantanee, 5 cumulate, 2 statiche).
- `slots.py`: aritmetica delle finestre di accumulo, sequenze di slot, rilevamento
  buchi, split temporali, codifica ciclica del tempo. Non importa `config` per
  evitare un ciclo: e' `config` a dipendere da qui per validarsi.
- `config.py`: configurazione pydantic con validazione semantica (area allineata
  alla griglia, target presenti tra le variabili scaricate, riferimenti delle teste
  coerenti, percorsi confinati sotto `data_root`).
- `tables.py`: schemi Parquet dichiarati e verificati in scrittura e rilettura.

### Fase 2 - Ingestione dati (in corso)

Completato:

- `credentials.py`: risoluzione delle credenziali senza mai esporne il valore,
  lettura in `utf-8-sig` per il BOM di Windows.
- `scripts/check_cds_access.py`: diagnosi separata di credenziali, token, licenze e
  disponibilita' temporale. Accesso verificato, vedi 5.1.
- `tables.py`: schemi Parquet dichiarati e verificati a runtime.
- `data/download.py`: una richiesta per mese e famiglia di variabili, ritaglio
  all'area, ripresa e scrittura atomica. Collaudato con un client finto: nessun test
  contatta il CDS.

- `data/ingest.py` + `scripts/ingest_era5.py`: GRIB -> Zarr, catalogo Parquet, tabella
  dei fold. Documentato in dettaglio in `INGESTION.md`.

Da fare: `features.py`, `dataset.py`.

Nota sui test: il backoff di produzione e' 30 s e i test sui riprovi lo azzerano via
configurazione. Lasciandolo attivo la suite passava da 9 a 338 secondi.

### 4.1 Fatti verificati sull'ingestione

Struttura reale dei GRIB, ispezionata prima di scrivere il codice:

| Famiglia | Struttura | Note |
|---|---|---|
| istantanee | `(time, latitude, longitude)` | asse temporale piatto, 7 variabili |
| cumulate | `(time, step, latitude, longitude)` | corse di previsione, `valid_time` **bidimensionale** |
| statiche | `(latitude, longitude)` | nessun asse temporale |

Le cumulate sono il punto delicato: due corse al giorno (base 06 e 18 UTC) con step
orari. Appiattendole si ottengono 756 istanti validi per un mese di 31 giorni, **zero
duplicati**, tutte le 24 ore presenti.

Verifiche sull'ingestione di gennaio 2024:

| Cosa | Esito |
|---|---|
| Slot ingeriti | 93 su 93, 9 variabili, **0 NaN**, 17,7 s |
| Plausibilita' fisica | tutte le 9 variabili in range (t2m 217-313 K, msl 939-1048 hPa) |
| Somma della finestra di accumulo | ricalcolata dal GRIB in modo indipendente: **differenza 0,0** |
| Chunk scritti | 12 per variabile invece di 358: Zarr salta i chunk interamente NaN |

### 4.2 Rumore di quantizzazione su `sf` e `tp`

Fisicamente la neve in equivalente d'acqua non puo' superare la precipitazione totale.
Nei dati accade nel **12,26 %** dei punti.

Non e' un difetto dell'ingestione. `tp` e `sf` sono impacchettati in GRIB come interi
scalati **in modo indipendente**, quindi quando nevica puro (`sf` ~ `tp`)
l'arrotondamento puo' far superare `tp`. Misure: violazione massima **0,0055 mm**,
rapporto `sf/tp` massimo **1,043**, e dove `tp = 0` il valore di `sf` non supera
0,005 mm.

Decisione: l'ingestione **non corregge** il dato, per restare fedele alla sorgente. La
correzione appartiene alla costruzione del target, dove il rapporto va limitato a [0, 1]
e definito solo sopra la soglia di 0,1 mm. La testa `fraction_of` resta appropriata.

## 5. Verifiche eseguite

| Cosa | Come | Esito |
|---|---|---|
| Registro variabili | import ed estrazione spec, nome sconosciuto | superato: 24 spec, errore esplicito |
| Finestre di accumulo | slot 06 -> ore 3..10, slot 18 -> 15..22 | superato, coerente con la convenzione ERA5 `(H-1, H]` |
| Copertura giornaliera | `accumulation_coverage([6,12,18], 8)` | 20 ore su 24, 4 conteggi ridondanti (come dichiarato) |
| Vincolo mezzanotte | slot notturni con finestra 8 h | superato: solleva `ValueError` |
| Rilevamento buchi | serie completa e serie bucata | superato |
| Periodo a granularita' di giorno | mesi parziali agli estremi | superato: 2026-08 troncato a 11 giorni, nessuna richiesta oltre il limite pubblicato |
| Fold a finestra mobile | copertura stagionale dei test | superato: 6 fold, tutti i 12 mesi coperti, mesi nevosi inclusi |
| Codifica temporale | ciclicita' giornaliera e stagionale | superato |
| Ambiente runtime | `tests/test_environment.py` | superato: backend `cfgrib` registrato in xarray, binari eccodes raggiungibili, torch senza CUDA, round-trip Parquet |
| Configurazione | `tests/test_config.py` | superato: 30 casi, incluse tutte le incoerenze semantiche |
| Layout canali e rete | `tests/test_models.py` | superato: 45 casi |
| Rete su dominio reale | forward 261 x 401 (non divisibile per 8) | superato: forma preservata grazie al padding riflesso |
| Costo su CPU | `scripts/benchmark_model.py` | misurato, vedi sotto |
| Credenziali | `tests/test_credentials.py` | superato: 18 casi, incluso il BOM e la non esposizione del valore |
| Layer tabellare | `tests/test_tables.py` | superato: 20 casi, incluso un Parquet con schema vecchio |
| Downloader | `tests/test_download.py` | superato: 27 casi con client finto (ripresa, riprovi, scrittura atomica) |
| Ingestione | `tests/test_ingest.py` | superato: 27 casi, dataset sintetici + integrazione sui GRIB reali |
| Ingestione su dati reali | gennaio 2024 | superato: 93 slot, 0 NaN, accumulo ricalcolato con differenza 0,0 |
| Lint | `ruff check src tests scripts` | nessun rilievo |
| Suite completa | `pytest tests` | 305 superati |

### 5.1 Accesso CDS verificato

`scripts/check_cds_access.py`, eseguito il 2026-08-17:

| Controllo | Esito |
|---|---|
| Credenziali lette da `.env` | url e key presenti |
| Classe client effettiva | `ecmwf.datastores.legacy_client.LegacyClient` (dispatch confermato) |
| Autenticazione | OK |
| `licence-to-use-copernicus-products`, `terms-of-use-cds` | gia' accettate |
| Estensione dataset | **1940-01-01 .. 2026-08-11** (latenza 6 giorni) |
| Download reale su 2024-01-01 | OK, 116 byte |
| Download reale su 2026-08-11 | OK, 116 byte |

Due difetti del mio script corretti durante la verifica:

1. Il primo tentativo passava `dataset=` a `get_licences`, che accetta **solo**
   `scope`. Il fallback elencava tutte le 48 licenze del portale e le segnalava come
   da accettare: `--accept-licences` ne avrebbe accettate 44 **estranee** a nome
   dell'utente. Rimossa l'accettazione in blocco; ora l'unico test della licenza e' il
   download reale e si accetta solo una licenza indicata esplicitamente.
2. La disponibilita' veniva sondata provando date a caso. Il campo `end_datetime` dei
   metadati della collection e' la fonte autorevole ed evita richieste inutili.

Nota: la collection espone `licences: null`, quindi **l'API non permette di sapere
quali licenze richiede un singolo dataset**.

### Costo misurato su CPU (non stimato)

Rete con 221 canali di input, 45 di uscita, `base_channels=48`, `depth=3`:

| | |
|---|---|
| Parametri | 9.968.685 |
| Passo di training (batch 4, crop 96 x 96) | 1,54 s |
| Per campione | 0,39 s |
| **Epoca sull'intero train set (1504 campioni)** | **~10 min** |
| 20 epoche sull'intero train set | ~3,3 h |
| Inferenza sul dominio intero 261 x 401 | 1,51 s |

Conclusione: il training alla risoluzione nativa 0.25 gradi **e' praticabile su questa
CPU** grazie all'approccio a crop. L'ipotesi piu' rischiosa del progetto e' quindi
verificata.

## 6. Problemi aperti

### 6.1 Test set monostagionale (RISOLTO con la finestra mobile)

Misurato sul periodo 2024-01-01 .. 2026-08-11 con split contiguo:

| Split | Periodo | Slot | Mesi nevosi |
|---|---|---|---|
| train | 2024-01-01 .. 2025-10-29 | 2003 | 32% |
| val | 2025-11-08 .. 2026-03-31 | 429 | 84% |
| test | 2026-04-10 .. 2026-08-11 | 370 | **0%** |

Risolto passando alla rolling origin validation (vedi 2.1): 6 fold i cui blocchi di
test coprono tutti i 12 mesi. Copertura verificata da test automatico, non a occhio.

### 6.2 Latenza ERA5

ERA5/ERA5T ha 5-6 giorni di ritardo. Il notebook di inferenza produrra' quindi una
previsione per giorni **gia' trascorsi** (hindcast verificabile, utile per validare,
ma non una previsione operativa). Per il tempo reale servirebbe una seconda sorgente
(`data.ecmwf.int`). Il downloader e' progettato con sorgente sostituibile.

### 6.3 Credenziali CDS (risolto, con un'avvertenza)

Credenziali presenti in `.env` (ignorato da git) e accesso verificato end-to-end, vedi
5.1. Restano due note operative:

- il token va **ruotato** quando il download massivo e' concluso, perche' e' transitato
  in un canale non controllato;
- il download reale non e' ancora stato lanciato: la pipeline oltre `download.py` e'
  ancora validata solo su dati sintetici.

### 6.4 Fattibilita' del training a 0.25 gradi su CPU (RISOLTO)

Misurato: ~10 minuti per epoca sull'intero train set, 1,5 s per l'inferenza sul
dominio intero. Il training a piena risoluzione e' praticabile. Vedi la tabella dei
costi nella sezione 5.

`samples_per_epoch` in `configs/default.yaml` e' fermo a 512 per prudenza: dato il
costo misurato conviene alzarlo a coprire tutto il train set (1504 campioni).

### 6.5 Installazione dipendenze (risolto)

Il primo `uv sync` e' fallito per timeout di rete su `eccodes`
(`UV_HTTP_TIMEOUT` di default 30 s). Rilanciato con `UV_HTTP_TIMEOUT=600`: completato,
144 pacchetti. Da riportare nel README e nel Dockerfile.

---

## 7. Note verificate sulle librerie

Accertate leggendo il sorgente installato, non la documentazione a memoria.

### `cdsapi` 0.7.7

- Le credenziali sono risolte da `get_url_key_verify`: prima `CDSAPI_URL` e
  `CDSAPI_KEY`, poi il file indicato da `CDSAPI_RC` o `~/.cdsapirc`.
- `Client.__new__` fa **dispatch dinamico**: se il token contiene `:` (vecchio
  formato `<UID>:<APIKEY>`) restituisce `cdsapi.Client`, altrimenti, come accade con
  il Personal Access Token attuale, restituisce
  `ecmwf.datastores.legacy_client.LegacyClient`. Verificato che le due classi
  accettino gli stessi keyword argument, quindi il codice resta valido su entrambi i
  percorsi.
- La risoluzione delle credenziali avviene dentro `__new__`, quindi la costruzione
  del client fallisce prima di qualunque richiesta. Il progetto le verifica a monte
  per produrre un messaggio con la procedura da seguire.
- **Sicurezza**: con `debug=True` il client registra `dict(url=..., key=...)`,
  esponendo la chiave nei log. Il client va costruito con `debug=False` (default).
- `retrieve(name, request, target=...)` scrive il file direttamente: **non** va
  concatenato `.download()` come in `Data/era5-request.py`.

### Altre

- `zarr` risolto alla **3.3.0**: l'API dei codec differisce dalla 2.x. Il progetto usa
  solo l'astrazione di xarray per non accoppiarsi a quella differenza.
- Il parametro `grid` (regridding lato server) non compare nel form web del CDS e
  alcune installazioni lo rifiutano: viene inviato solo quando la risoluzione
  richiesta differisce dalla nativa.
