# Stato dello sviluppo

Documento di tracciamento, aggiornato a ogni fase completata. Registra decisioni
prese, verifiche effettivamente eseguite e problemi aperti.

- **Branch di lavoro**: `feature/era5-forecasting-pipeline`
- **Ultimo aggiornamento**: 2026-08-17
- **Fase corrente**: costruzione della pipeline dati (nessun dato reale ancora scaricato)
- **Stato verifiche**: 184 test superati, `ruff` senza rilievi

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
| Periodo | 2024-08 .. 2026-07 (24 mesi, 2190 slot) | 2 anni per validare, poi estendibile |
| Slot giornalieri | 06, 12, 18 UTC | Corrispondono a mattina / mezzogiorno / sera |
| Cumulate | finestra di 8 h centrata sullo slot | Copre 20 h su 24 con 4 conteggi ridondanti; nessuna finestra sconfina dal giorno, quindi l'ingestione resta mensile |
| Storage numerico | Zarr + dask, **solo layer di ingestione** | Accesso casuale a finestre spaziotemporali senza caricare tutto in RAM |
| Storage tabellare | **Polars / Parquet**, layer dati principale | Richiesta dell'utente; `slots.parquet` e' il registro autorevole degli slot |
| Config | pydantic con validazione runtime | Impedisce che download, training e inferenza divergano sui parametri condivisi |
| Training | CPU, patch-based su crop 96 x 96 | Nessuna GPU disponibile; la rete e' completamente convoluzionale e in inferenza si applica a 261 x 401 |
| Packaging | `uv` + `pyproject.toml` + Docker | Richiesta dell'utente |

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
  tables.py               layer Polars/Parquet con schemi verificati
tests/
  test_slots.py           test dell'algebra temporale
  test_environment.py     smoke test delle capacita' runtime (serve anche per Docker)
data/                     (ignorato da git) raw GRIB, Zarr, tables, artifacts
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

Da fare: `download.py`, `ingest.py` (GRIB -> Zarr), `features.py`, `dataset.py`.

## 5. Verifiche eseguite

| Cosa | Come | Esito |
|---|---|---|
| Registro variabili | import ed estrazione spec, nome sconosciuto | superato: 24 spec, errore esplicito |
| Finestre di accumulo | slot 06 -> ore 3..10, slot 18 -> 15..22 | superato, coerente con la convenzione ERA5 `(H-1, H]` |
| Copertura giornaliera | `accumulation_coverage([6,12,18], 8)` | 20 ore su 24, 4 conteggi ridondanti (come dichiarato) |
| Vincolo mezzanotte | slot notturni con finestra 8 h | superato: solleva `ValueError` |
| Rilevamento buchi | serie completa e serie bucata | superato |
| Split su 2 anni | 2190 slot | train 1504 campioni, val 299, test 240; nessuna finestra attraversa i confini |
| Codifica temporale | ciclicita' giornaliera e stagionale | superato |
| Ambiente runtime | `tests/test_environment.py` | superato: backend `cfgrib` registrato in xarray, binari eccodes raggiungibili, torch senza CUDA, round-trip Parquet |
| Configurazione | `tests/test_config.py` | superato: 30 casi, incluse tutte le incoerenze semantiche |
| Layout canali e rete | `tests/test_models.py` | superato: 45 casi |
| Rete su dominio reale | forward 261 x 401 (non divisibile per 8) | superato: forma preservata grazie al padding riflesso |
| Costo su CPU | `scripts/benchmark_model.py` | misurato, vedi sotto |
| Lint | `ruff check src tests scripts` | nessun rilievo |
| Suite completa | `pytest tests` | 184 superati |

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

### 6.1 Test set monostagionale (da decidere)

Con 24 mesi e split contiguo 70/15/15, il blocco di test copre **solo maggio-luglio
2026**. In quel periodo la neve sull'area e' praticamente assente, quindi **le
metriche sulla neve non sarebbero misurabili** e quelle su temperatura e
precipitazione descriverebbero solo il regime estivo.

| Split | Periodo | Slot | Campioni |
|---|---|---|---|
| train | 2024-08-01 .. 2025-12-24 | 1533 | 1504 |
| val | 2026-01-04 .. 2026-04-23 | 328 | 299 |
| test | 2026-05-03 .. 2026-07-31 | 269 | 240 |

Opzioni: estendere il periodo a piu' anni, oppure aggiungere una strategia di split
a blocchi mensili distribuiti sulle stagioni (mantiene la copertura stagionale senza
introdurre leakage, a costo di piu' giunzioni). **Decisione non ancora presa.**

### 6.2 Latenza ERA5

ERA5/ERA5T ha 5-6 giorni di ritardo. Il notebook di inferenza produrra' quindi una
previsione per giorni **gia' trascorsi** (hindcast verificabile, utile per validare,
ma non una previsione operativa). Per il tempo reale servirebbe una seconda sorgente
(`data.ecmwf.int`). Il downloader e' progettato con sorgente sostituibile.

### 6.3 Credenziali CDS assenti

Ne' `~/.cdsapirc` ne' variabili d'ambiente sono presenti: **nessun dato reale e'
stato scaricato**. Procedura per ottenerle in `README.md`. Finche' non ci sono, la
pipeline e' validata solo su dati sintetici con struttura identica a quella reale.

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
