# Come funziona l'ingestione dei dati

Questo documento spiega come i file GRIB scaricati dal Climate Data Store diventano
tensori pronti per l'addestramento. È pensato per chi riprende il progetto senza averlo
scritto: descrive **cosa** fa ogni passaggio, **perché** è fatto così e **come**
rieseguirlo o estenderlo.

I numeri riportati sono misurati sull'ingestione reale di gennaio 2024, non stimati.

---

## 1. Il problema

ERA5 distribuisce GRIB, un formato pensato per l'archiviazione meteorologica: compatto,
autodescrittivo, ma pessimo per l'addestramento. Leggere un campo richiede di
decodificare messaggi sequenzialmente, non esiste accesso casuale efficiente a "lo slot
numero 1417", e le variabili istantanee e cumulate hanno strutture temporali diverse e
incompatibili.

Un modello che addestra ha bisogno dell'opposto: leggere rapidamente finestre arbitrarie
di 30 slot consecutivi, migliaia di volte per epoca, in ordine casuale.

La pipeline risolve questo con **due livelli di archiviazione**, ciascuno scelto per il
compito che deve svolgere.

---

## 2. I due livelli

### Livello 1 — Zarr: i tensori

`datasets/era5_slots.zarr` contiene i dati numerici, con dimensioni
`slot × latitude × longitude` = `2862 × 261 × 401`, una variabile per array.

Zarr è, in sostanza, **un array numpy su disco**: suddiviso in blocchi (*chunk*)
compressi indipendenti, leggibili in parallelo e senza caricare il resto. Ogni chunk
copre 8 slot e l'intera griglia spaziale, quindi leggere una finestra temporale non
tocca mai dati spaziali inutili.

Perché non altro:

| Formato | Perché scartato |
|---|---|
| CSV | 229 milioni di righe per mese, testo non compresso, nessun accesso casuale. Improponibile. |
| GRIB direttamente | Nessun accesso casuale; decodifica ripetuta a ogni epoca. |
| NetCDF singolo file | Un solo file da GB, difficile da scrivere in modo incrementale e concorrente. |
| Parquet per i tensori | Colonnare orientato alle righe: un tensore 3-D diventa una tabella lunga e la lettura amplifica di circa 11 volte. |

Zarr è inoltre indipendente dal linguaggio: lo stesso store si legge da Python, Julia,
R o JavaScript.

### Livello 2 — Parquet: i metadati

`datasets/tables/*.parquet` contiene tutto ciò che **descrive** i dati e si interroga con
Polars: quali slot esistono, quali sono utilizzabili, quali statistiche hanno, come sono
divisi in fold.

Sono tabelle piccole (25 KB per il catalogo di 2862 slot) su cui si fanno filtri e
raggruppamenti. È esattamente il caso d'uso in cui Polars eccelle, mentre i tensori
restano in Zarr.

**La regola divisoria**: numeri su griglia → Zarr; tutto ciò su cui si vuole fare una
query → Parquet.

---

## 3. Il flusso completo

```
CDS API                  scripts/download_era5.py
   │
   ├─ instantaneous_YYYY-MM.grib   7 variabili, ore 3..22, ~139 MB
   ├─ accumulated_YYYY-MM.grib     2 variabili, corse di previsione, ~248 MB
   └─ static.grib                  2 campi invarianti, 613 KB
   │
   ▼                       scripts/ingest_era5.py
 ingestione
   │
   ├─ era5_slots.zarr      9 variabili × 2862 slot
   ├─ era5_static.zarr     2 campi × griglia
   └─ tables/*.parquet     slots, slot_stats, folds, variables
```

### 3.1 Preallocazione dello store

`initialize_store` crea l'intero store per **tutto** il periodo configurato, riempito di
NaN, prima di ingerire qualsiasi mese. Ogni mese viene poi scritto nella sua regione con
`region=`, senza riscrivere il resto.

Questo permette di ingerire i mesi **in qualunque ordine e mentre il download prosegue**:
i mesi non ancora arrivati restano NaN e il catalogo li marca come non utilizzabili.

Un dettaglio verificato che rende la scelta economica: **Zarr non scrive su disco i chunk
interamente uguali al valore di riempimento**. Dopo aver preallocato 2862 slot e ingerito
solo gennaio 2024, esistono 12 chunk per variabile invece di 358. Preallocare non costa
spazio.

### 3.2 Variabili istantanee

Struttura reale, verificata sui file: un asse `time` piatto con tutte le ore richieste,
più `latitude` e `longitude`. La lettura è diretta: si selezionano gli istanti degli slot
(06, 12, 18 UTC) e si scrivono.

### 3.3 Variabili cumulate — il punto delicato

Qui il GRIB non ha un asse temporale piatto. ERA5 archivia le cumulate come **corse di
previsione**: due corse al giorno, con base alle 06 e alle 18 UTC, ciascuna con più
`step` orari. La struttura è `(time, step)` e l'istante reale di ogni valore sta in una
coordinata **bidimensionale** `valid_time = time + step`.

`flatten_accumulated` la riduce a un unico asse ordinato:

1. `stack` fonde `(time, step)` in un asse unico;
2. `swap_dims` lo sostituisce con `valid_time`;
3. `sortby` ordina cronologicamente;
4. gli istanti duplicati vengono rimossi tenendo il primo;
5. **`transpose` impone l'ordine `(valid_time, latitude, longitude)`**.

Il punto 5 non è cosmetico. `stack` colloca il nuovo asse in **ultima** posizione, quindi
senza transpose gli array escono come `(latitude, longitude, valid_time)` e ogni
indicizzazione temporale seleziona in realtà la latitudine. Questo difetto si è
manifestato come `IndexError: index 261 is out of bounds` ed è coperto da un test
dedicato.

Fatti verificati sui file reali: 756 istanti validi per un mese di 31 giorni, **zero
duplicati**, tutte le 24 ore presenti. Le due corse quotidiane si affiancano senza
sovrapporsi.

### 3.4 La finestra di accumulo

Precipitazione e neve non hanno un valore "istantaneo": hanno senso solo su un
intervallo. Ogni slot riceve la somma delle **8 ore centrate su di sé**:

| Slot | Ore sommate |
|---|---|
| 06 UTC | 03–10 |
| 12 UTC | 09–16 |
| 18 UTC | 15–22 |

Tre finestre da 8 ore coprono 20 delle 24 ore del giorno. La scelta è vincolata da una
proprietà che semplifica tutto: **nessuna finestra attraversa la mezzanotte**, quindi ogni
mese è autosufficiente e si può ingerire senza leggere il mese precedente o successivo.

L'aggregazione è stata verificata ricalcolandola in modo indipendente dal GRIB per uno
slot: **differenza massima 0,0**.

### 3.5 Controlli di qualità

Per ogni slot e variabile, `compute_stats` registra minimo, massimo, media, numero di
valori validi e di NaN in `slot_stats.parquet`.

Il catalogo marca `usable = false` se un mese non è stato ingerito **o** se contiene
valori non finiti. Uno slot inutilizzabile propagherebbe NaN a tutti i campioni che lo
contengono, quindi il filtro è a monte del dataset di addestramento.

Verifica di plausibilità fisica su gennaio 2024, tutte le variabili in range:

| Variabile | Minimo | Massimo | Media |
|---|---|---|---|
| `t2m` (K) | 217,1 | 313,2 | 281,0 |
| `msl` (Pa) | 93 876 | 104 809 | 101 379 |
| `u10` (m/s) | −23,2 | 30,0 | 0,3 |
| `tcc` (0–1) | 0,00 | 1,00 | 0,62 |
| `sd` (m) | 0,0 | 10,0 | 0,22 |
| `tp` (m/8h) | 0,0 | 0,070 | 0,0007 |
| `sf` (m/8h) | 0,0 | 0,049 | 0,0002 |

### 3.6 Un'anomalia reale: `sf > tp`

Fisicamente la neve, espressa in equivalente in acqua, non può superare la
precipitazione totale. Nei dati succede nel **12,26 %** dei punti.

Non è un difetto dell'ingestione: è **rumore di quantizzazione del GRIB**, che comprime i
valori in interi scalati. `tp` e `sf` sono impacchettati in modo indipendente, quindi
quando nevica puro (`sf ≈ tp`) l'arrotondamento può far superare `tp`. Misure:

- violazione massima: **0,0055 mm**;
- rapporto `sf/tp` massimo: **1,043**;
- dove `tp = 0`, `sf` massimo è 0,005 mm.

L'ingestione **non corregge** il dato, per restare fedele alla sorgente. La correzione
appartiene alla costruzione del target, dove il rapporto viene limitato a [0, 1] e
definito solo sopra la soglia di 0,1 mm.

### 3.7 I quattordici campi invarianti, verificati uno per uno

I descrittori di superficie sono stati scaricati in un'unica richiesta da 4,0 MB
(`static.grib`, un solo istante) e letti con `tmp/diagnostica/verifica_statici.py`. Tutti
e quattordici sono presenti sulla griglia 261x401. Intervalli misurati:

| Campo | Significato | Minimo | Massimo | Media |
|---|---|---|---|---|
| `lsm` | frazione di terra | 0 | 1 | 0,5200 |
| `z` | geopotenziale (m2 s-2) | -1260,77 | 31544,98 | 2588,78 |
| `slt` | tipo di suolo (classi) | 0 | 7 | 1,11 |
| `cvh` | frazione vegetazione alta | 0 | 1 | 0,1335 |
| `cvl` | frazione vegetazione bassa | 0 | 1 | 0,1921 |
| `tvh` | tipo vegetazione alta | 0 | 19 | 4,05 |
| `tvl` | tipo vegetazione bassa | 0 | 17 | 2,26 |
| `cl` | frazione acque interne | 0 | 1 | 0,0165 |
| `dl` | profondita' acque interne (m) | 0,50 | 6218,29 | 1129,69 |
| `sdor` | dispersione orografia (m) | 0 | 672,35 | 30,53 |
| `isor` | anisotropia orografia | 0 | 0,9869 | 0,2712 |
| `anor` | orientamento orografia (rad) | -1,5578 | 1,5627 | 0,3823 |
| `slor` | pendenza orografia | 0,0001 | 0,1177 | 0,0051 |
| `sdfor` | dispersione orografia filtrata (m) | 0 | 526,94 | 21,09 |

Due valori sembrano anomali e vanno spiegati, non corretti.

**Il geopotenziale e' negativo dove c'e' terra sotto il livello del mare.** Il minimo
-1260,77 m2 s-2 corrisponde a -128,6 m: e' terreno reale, non un errore di segno. Il
dominio comprende la depressione del Caspio e altre aree sotto il livello del mare.

**`dl` non e' un campo utilizzabile come canale grezzo.** Una profondita' media di 1130 m
e' impossibile per le acque interne europee, e il massimo di 6218 m non appartiene a
nessun lago: ERA5 definisce `dl` **su tutta la griglia**, con valori di riempimento dove
non ci sono laghi, e la frazione `cl` ha media 0,0165, cioe' il campo e' significativo su
meno del 2% dei punti. Usato cosi' come sta, `dl` inietterebbe un segnale di "acqua
profonda" sopra l'oceano e sopra la terraferma. Va quindi usato **solo moltiplicato per
`cl`**, oppure lasciato fuori: per questo non entra nella lista predefinita dei campi
statici, mentre gli altri tredici sono utilizzabili direttamente.

---

## 4. Le tabelle Parquet

| Tabella | Contenuto | Ruolo |
|---|---|---|
| `slots` | un record per slot: istante, anno, mese, ora, `usable`, `split` indicativo | catalogo generale |
| `slot_stats` | statistiche per slot e variabile | controllo qualità |
| `folds` | **autorevole**: `fold`, `split`, `slot_index`, `is_sample_start` | divisione dei dati |
| `variables` | metadati delle variabili: unità, famiglia, ruolo | documentazione |
| `downloads` | esito di ogni download | tracciamento |

### Perché i fold hanno una tabella propria

Con la validazione a finestra mobile, **lo stesso slot cambia split da un fold all'altro**:
ciò che è test nel fold 0 diventa train nel fold 3. Una singola colonna `split` non può
rappresentarlo, quindi `folds.parquet` è la fonte autorevole e la colonna `split` di
`slots` è solo indicativa. Un test verifica proprio che esista almeno uno slot con più
split.

`is_sample_start` è vero solo se **tutti** i 30 slot della finestra (21 di input + 9 di
target) sono utilizzabili. Il dataset di addestramento legge solo questa colonna.

---

## 5. Come eseguirlo

```bash
# 1. Scaricare i GRIB (lungo: ~9 min per mese)
uv run python scripts/download_era5.py --dry-run        # anteprima
uv run python scripts/download_era5.py --from-month 2025-01

# 2. Vedere cosa è pronto
uv run python scripts/ingest_era5.py --list

# 3. Ingerire
uv run python scripts/ingest_era5.py                    # tutti i mesi disponibili
uv run python scripts/ingest_era5.py --month 2024-01    # uno solo
```

L'ingestione è **idempotente**: rieseguirla su un mese già fatto lo riscrive con lo
stesso risultato. `--recreate-store` azzera lo store e obbliga a reingerire tutto; serve
solo se cambia la configurazione della griglia o del periodo.

### Costi misurati

| Operazione | Tempo | Spazio |
|---|---|---|
| Download di un mese | ~9 min | 387 MB (GRIB) |
| Ingestione di un mese | 17,7 s | ~150 MB (Zarr) |
| Periodo completo, 32 mesi | ~4,7 h | ~12,4 GB GRIB + ~4,6 GB Zarr |

Lo Zarr è più piccolo del GRIB perché è una **riduzione**: conserva 3 slot al giorno
invece di 24 campi orari, e le cumulate già sommate sulla finestra.

---

## 6. Come estenderlo

**Aggiungere una variabile**: dichiararla in `src/dwf/variables.py` con nome CDS e nome
GRIB, aggiungerla alla lista in `configs/default.yaml`, riscaricare e reingerire con
`--recreate-store` (lo store ha un array per variabile, deciso alla creazione).

**Allungare il periodo**: modificare `time.start` / `time.end` nella configurazione,
scaricare i mesi nuovi, reingerire con `--recreate-store`. Attenzione: cambiare il
periodo cambia anche il numero di fold.

**Cambiare la finestra di accumulo**: `time.accum_window_hours`. Il vincolo di non
attraversare la mezzanotte è verificato da un test, che fallirà se la finestra diventa
troppo ampia per gli slot configurati.

**Cambiare la regione**: `region` nella configurazione. Richiede di riscaricare tutto.

---

## 7. Difetti trovati e risolti durante lo sviluppo

Elenco onesto, utile a chi debuggherà il codice in futuro:

| Difetto | Sintomo | Causa |
|---|---|---|
| Ordine degli assi | `IndexError: index 261 out of bounds` | `stack` mette il nuovo asse per ultimo |
| Fusi orari | `UserWarning: no explicit representation of timezones` | numpy scartava il fuso in silenzio |
| Mese fuori periodo | Restituiva il mese intero | Confronto sbagliato dei limiti in `days_for_month` |
| Destinazione dati | I GB finivano in `Data/` tracciata da git | Windows ignora la differenza di maiuscole |

---

## 8. Cosa non è ancora fatto

L'ingestione è completa e verificata. Restano da costruire, a valle:

- `features.py`: canali derivati (variazioni temporali, codifiche stagionali, statici);
- `dataset.py`: dataset torch che legge `folds.parquet` e ritaglia le finestre;
- normalizzazione calcolata **solo sul train** di ciascun fold, per non far filtrare
  informazione dal futuro.
