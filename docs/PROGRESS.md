# Deep Weather Forecasting - documento di subentro

Questo documento e' scritto perche' una persona che non ha mai visto il progetto possa
riprenderlo in mano. Non racconta solo **cosa** c'e', ma **perche'** e' cosi', quali
alternative sono state scartate e con quale misura, e dove sono i limiti veri.

Regola di lettura adottata in tutto il progetto: **i dati sono veri**. ERA5 e' una
rianalisi prodotta assimilando osservazioni in un modello fisico. Davanti a un numero
sorprendente, la prima ipotesi da verificare e' un errore di chi analizza. Questo
documento riporta i casi in cui quell'ipotesi si e' rivelata giusta, perche' sono la
parte piu' istruttiva del lavoro.

---

## 1. Che cosa fa

Previsione a **tre giorni** su griglia, addestrata e ottenuta **in locale su CPU**.

- **Ingresso**: 7 giorni di storico ERA5, cioe' 21 istanti (06, 12, 18 UTC).
- **Uscita**: 9 istanti futuri (3 giorni x 3 ore del giorno), prodotti **tutti in una
  sola passata**, non in modo autoregressivo.
- **Dominio**: euro-atlantico, 261 x 401 celle a 0,25 gradi (75 N .. 10 N, 40 W .. 60 E).
- **Grandezze previste**: temperatura a 2 m in **gradi Celsius**, precipitazione
  (probabilita' e quantita'), neve (frazione della precipitazione) e **incertezza
  calibrata** su ciascuna.
- **Consegna**: tabelle Parquet, due notebook e un **report PDF** con mappe.

Il progetto ha un luogo di interesse dichiarato, **Vigo di Cadore**, che compare nella
pesatura della perdita, nell'analisi dei dati e nel report.

### Stato

| | |
|---|---|
| Test | **678**, tutti verdi, anche dentro il container |
| Lint | `ruff` pulito su `src`, `tests`, `scripts` |
| Commit sul ramo | 46, nessuno spinto |
| Dati scaricati | 21 mesi (2024-01, 2025-01 .. 2026-08); il 2024 residuo e' in scaricamento |
| Dati ingeriti | 16 mesi, 1458 istanti, zero valori mancanti |
| Docker | immagine costruita e **verificata eseguendo la suite al suo interno** |

---

## 2. Come si esegue

### In locale

```bash
uv sync --extra notebooks          # ambiente riproducibile dal lockfile
uv run python scripts/check_cds_access.py
uv run python scripts/download_era5.py --dry-run
uv run python scripts/download_era5.py
uv run python scripts/ingest_era5.py
uv run python scripts/analyze_data.py      # -> DATA_ANALYSIS.md
uv run python scripts/compare_variants.py  # -> VARIANTS.md
uv run python scripts/train_model.py --fold 0
uv run python scripts/evaluate_model.py --fold 0 --split test
uv run python scripts/predict_forecast.py --fold 0
uv run python scripts/report_forecast.py --fold 0   # -> PDF
```

Le credenziali CDS stanno in `.env`, che non e' tracciato. Non vanno mai incollate in
chat ne' committate: una chiave transitata in un canale non cifrato va considerata
compromessa e ruotata.

### In container

```bash
docker compose build
docker compose run --rm dwf python scripts/ingest_era5.py --list
docker compose up jupyter        # notebook su http://localhost:8888
```

Dati, configurazioni, modelli e notebook sono **volumi**, non contenuti
dell'immagine: pesano decine di gigabyte e devono sopravvivere a una ricostruzione.

**Avvertenza verificata**: Debian bookworm fornisce ecCodes 2.28 mentre `cfgrib`
raccomanda 2.42. La suite passa comunque nel container, ma l'ingestione dei GRIB
conviene eseguirla sull'host.

---

## 3. Mappa del repository

```
src/dwf/
  config.py          Configurazione validata con pydantic; unico punto di verita'
  variables.py       Anagrafica delle 24 variabili: unita', trasformazioni, offset
  slots.py           Aritmetica degli istanti; qui vive diurnal_reference_index
  tables.py          Schemi Parquet dichiarati e verificati
  credentials.py     Lettura delle credenziali CDS, mai stampate
  solar.py           Geometria solare (Spencer 1971): 3 canali
  thermo.py          Termodinamica dell'aria umida: 4 canali
  weighting.py       Pesi spaziali della perdita (area + fuoco locale)
  persistence.py     Salvataggio dei modelli senza pickle
  calibration.py     Calibrazione isotonica delle probabilita'
  train.py evaluate.py predict.py report.py
  data/
    download.py ingest.py features.py dataset.py freshness.py refresh.py
  models/
    heads.py blocks.py network.py losses.py
    variants/        conv, attention, fourier, recurrent, hybrid
scripts/             12 punti d'ingresso da riga di comando
tests/               20 file, 673 test
```

Documenti: `INGESTION.md` (fatti verificati sui GRIB), `RESEARCH.md` (ricerca
tecnologica e scarti motivati), `DATA_ANALYSIS.md` (analisi esplorativa),
`VARIANTS.md` (confronto fra architetture).

---

## 4. I dati

### 4.1 Dominio e periodo

Il dominio non e' stato scelto a mano: e' stato ricavato dagli output gia' presenti
nel notebook esplorativo del repository originale. Il periodo termina alla data
dichiarata dai metadati della collection CDS, letta a runtime e non scritta a memoria:
ERA5 ha circa sei giorni di latenza, quindi **non arriva a oggi**. Ne consegue un
fatto importante per l'onesta' del progetto: cio' che chiamiamo "previsione" e' in
realta' un *hindcast verificabile*, e questo e' un pregio, perche' ogni previsione ha
una verita' con cui confrontarsi.

### 4.2 Struttura di archiviazione

Zarr per il tensore, Parquet per i cataloghi. La divisione non e' estetica: un tensore
denso di 2862 x 261 x 401 valori per variabile ha bisogno di accesso a blocchi e
compressione, cose che un formato colonnare orientato alle righe non offre. Polars
resta per cio' in cui e' imbattibile, cioe' i registri, le metriche e le giunzioni.

Blocchi `(8, 261, 401)`. Un tentativo di riblocchettare anche nello spazio e' stato
**misurato e scartato**: peggiorava le letture per finestra, che sono il caso d'uso
dominante.

### 4.3 Anomalie spiegate, non corrette

**Neve maggiore della precipitazione totale nel 12,26 % delle celle.** Sembra una
violazione fisica. Non lo e': i messaggi GRIB impacchettano `tp` e `sf` come interi
con passi di quantizzazione **indipendenti**. La violazione non supera mai 1,5 volte il
passo di quantizzazione e la sua correlazione con l'intensita' della precipitazione e'
0,013, cioe' nulla. E' rumore di rappresentazione, non un errore del dato. Per questo
il bersaglio della frazione nevosa viene limitato a [0, 1] **nel bersaglio** e non nei
dati archiviati: si vincola cio' che si chiede al modello, non si falsifica l'archivio.

**Gli istanti non sono equidistanti.** 06Z, 12Z e 18Z distano 6, 6 e 12 ore. Tre
istanti fanno esattamente un giorno. Sembra un dettaglio ed e' invece la scoperta piu'
importante dell'analisi: vedi la sezione 6.

---

## 5. Le grandezze in ingresso

245 canali: 189 di stato, 27 di tendenza, 21 di velocita' del vento, 2 statici, 2 di
latitudine, 4 di codifica temporale.

**Le temperature sono in gradi Celsius.** La conversione e' dichiarata sulla variabile
e applicata **una volta sola**, subito dopo la lettura. Questo non e' cosmetico:
convertendo piu' a valle, `normalize` e `denormalize` smetterebbero di essere l'una
l'inversa dell'altra. Il checkpoint gia' addestrato e' rimasto valido **bit per bit**,
perche' traslare dato e media della stessa quantita' non cambia il valore normalizzato,
e un test lo verifica.

### 5.1 Fisica derivata, senza scaricare nulla di nuovo

**`solar.py`** implementa le formule di Spencer (1971): declinazione, fattore di
distanza Terra-Sole, equazione del tempo, coseno dell'angolo zenitale, insolazione al
limite dell'atmosfera, durata del giorno con gestione esplicita del caso polare. Trenta
test la confrontano con riferimenti astronomici noti: obliquita' 23,44 gradi, perielio
al terzo giorno dell'anno, afelio al 185esimo.

**`thermo.py`** ricava dalla temperatura e dal punto di rugiada gia' scaricati:
tensione di vapore saturo, umidita' relativa, depressione del punto di rugiada,
pressione al suolo dalla pressione al livello del mare, umidita' specifica, rapporto di
mescolanza, **contenuto di calore latente** e temperatura potenziale equivalente di
Bolton. Trentacinque test contro valori tabulati.

Durante quella verifica un test falliva. **Il codice era giusto, l'asserzione era
sbagliata**: l'invariante che avevo scritto, theta_e >= T, vale solo sotto i 1000 hPa,
perche' sopra quella pressione la compressione porta la temperatura potenziale sotto
quella reale. L'invariante corretto e' theta_e >= theta. E' stato corretto il test, non
il codice.

---

## 6. La scoperta che ha cambiato il progetto

L'analisi di prevedibilita' mostrava una correlazione che **non** decadeva in modo
monotono con la scadenza: risaliva a 3, 6 e 9 istanti. Applicando la regola "i dati
sono veri", la spiegazione e' risultata essere un mio errore di etichetta: avevo
scritto la colonna delle ore come `scadenza x 6`, ma gli istanti non sono equidistanti,
e 3 istanti sono **un giorno esatto**. I picchi erano semplicemente **la stessa ora del
giorno**.

Questo ha smascherato un riferimento molto piu' forte di quello che stavamo usando.

| scadenza | ore | persistenza ingenua | **persistenza diurna** | guadagno |
|---:|---:|---:|---:|---:|
| 1 | 8 | 4,535 | **2,401** | 47,1 % |
| 3 | 24 | 2,401 | 2,401 | 0 % |
| 5 | 40 | 5,201 | **3,173** | 39,0 % |
| 9 | 72 | 3,557 | 3,557 | 0 % |

Ripetere *ieri alla stessa ora* costa zero e raggiunge un errore quadratico di circa
3,0 gradi. Il modello addestrato ne faceva **4,45**. Il vantaggio dichiarato in
precedenza era quindi un artefatto di un riferimento troppo debole: contro il
riferimento giusto, **il modello perdeva**.

### 6.1 La conseguenza operativa

Se un riferimento gratuito e' cosi' forte, chiedere alla rete di ricostruirlo da zero
e' uno spreco di capacita'. La testa gaussiana ora produce uno **scarto** che viene
sommato all'osservazione piu' recente alla stessa ora del bersaglio. Solo la media
viene traslata: la log-varianza descrive l'incertezza dello scarto e non va spostata.

L'effetto e' misurato, non supposto: a parita' assoluta di protocollo, tre passate
ridotte portano da **6,613** a **3,652** gradi di errore quadratico.

Il riferimento diurno e' anche entrato nella valutazione come modello a se
(`persistence_diurnal`), accanto a quella ingenua e alla climatologia.

---

## 7. Il modello

### 7.1 Contratto

La rete e' un encoder-decoder a U completamente convoluzionale: si addestra su ritagli
e si applica alla griglia intera. Produce **un solo tensore** `(B, C, H, W)`; la
mappatura dei canali su (variabile, componente, scadenza) e' dichiarata in
`OutputLayout`. Indicizzare quei canali a mano sarebbe l'errore piu' silenzioso
possibile: scambiare media e log-varianza non fa fallire nulla, produce solo previsioni
sbagliate.

I pesi finali sono azzerati all'inizializzazione: la rete parte da una previsione
costante, non da rumore.

### 7.2 Le varianti confrontabili

Il confronto fra architetture ha valore solo se **cambia una cosa sola**. Qui la cosa
sola e' il **blocco elementare**: scheletro, canali di ingresso, layout di uscita, dati,
perdita e protocollo restano identici.

| variante | parametri | ms/passata | idea |
|---|---:|---:|---|
| `conv` | 9,98 M | 225 | convoluzione residua, riferimento |
| `attention` | 13,37 M | 415 | attenzione a finestre 8x8 con bias di posizione relativa |
| `fourier` | 8,99 M | 201 | convoluzione spettrale sui modi bassi, ricettivo globale |
| `recurrent` | 28,93 M | 1114 | ricorrenza convoluzionale a pesi condivisi |
| `hybrid` | 16,42 M | 371 | somma di ramo locale e ramo spettrale |

**Esito del confronto** (dettaglio in `VARIANTS.md`). A parita' assoluta di protocollo
le cinque architetture segnano fra 3,642 e 3,666 gradi. Ma ripetere **la stessa**
variante cambiando solo il seme produce 3,652 / 3,557 / 3,665, cioe' uno scarto tipo di
**0,059 gradi**: l'intera differenza fra le architetture, 0,024 gradi, sta dentro meno
di mezzo scarto tipo. **Il banco non riesce a distinguerle.** Proclamare vincitrice la
prima in classifica significherebbe leggere il seme, non il modello. E' esattamente la
saturazione dei backbone documentata in arXiv:2407.14129.

Il default va quindi a `conv`, non perche' abbia vinto ma perche' a parita' statistica
di accuratezza costa 99 s per passata contro 151, 159, 185 e 428 delle altre. Su CPU il
tempo e' il vincolo reale.

**Le varianti che perdono non vengono cancellate.** Restano in
`src/dwf/models/variants/`, documentate e selezionabili con `model.variant`, perche' il
risultato potrebbe ribaltarsi con piu' dati e perche' la misura che le ha scartate deve
restare riproducibile.

L'unico effetto che **esce** dal rumore e' l'ancoraggio diurno: 6,613 contro una media
di 3,625, circa cinquanta scarti tipo. Il modo di porre il problema ha pesato piu' di
ogni scelta di architettura.

Due precisazioni oneste:

- La variante di Fourier era arrivata a **228 milioni di parametri**. La causa era mia:
  allocavo 16 modi per asse mentre al collo di bottiglia la griglia si riduce a 12
  celle, quindi la gran parte dei pesi veniva troncata a ogni passata e non veniva mai
  addestrata. Con 8 modi e un ramo spettrale piu' stretto costa **meno** della
  convoluzione.
- La variante ricorrente **non e' il ConvLSTM temporale** della letteratura. Quello
  consuma una sequenza `(B, T, C, H, W)`, il che cambierebbe il layout di ingresso,
  cioe' proprio la variabile che il confronto tiene ferma. Qui si misura l'altra
  proprieta' interessante: la profondita' effettiva a parametri costanti.

Tecnologie escluse e perche', in dettaglio in `RESEARCH.md`: rappresentazioni sferiche
(il dominio non e' una sfera), grafi (su griglia regolare sono una convoluzione piu'
lenta), diffusione (produce ensemble, mentre qui l'incertezza e' gia' calibrata),
modelli fondazionali (richiedono livelli di pressione non scaricati).

### 7.3 La perdita

Somma pesata delle teste, piu' due termini aggiunti su richiesta e ciascuno con la
propria giustificazione misurata.

**Peso di area.** La griglia e' regolare in gradi, non in chilometri: a 70 gradi una
cella copre il 34 % di una cella equatoriale. Senza correzione la rete spenderebbe
capacita' sull'Artico.

**Fuoco su Vigo di Cadore.** Una campana attorno al punto, isotropa in chilometri e non
in gradi. Il guadagno e' volutamente contenuto: alzarlo trasformerebbe un modello di
dominio in un modello locale addestrato su una manciata di celle, che generalizzerebbe
peggio ovunque, Vigo compreso.

Entrambi i pesi sono **normalizzati a media unitaria**, quindi cambiarli non cambia la
scala della perdita e i pesi relativi fra le teste restano confrontabili.

**Termine spettrale.** Sotto errore quadratico puro, se la correlazione fra previsione
e realta' vale rho, il minimo si ottiene producendo un campo con ampiezza rho volte
quella vera: sfumare conviene, perche' una struttura nel posto sbagliato viene punita
due volte, dove c'e' e dove manca. E' esattamente il difetto misurato sul primo
modello, che sottostimava di 4,8 gradi l'escursione a mezzogiorno. Confrontare i moduli
della trasformata di Fourier premia l'ampiezza corretta **senza** reintrodurre la
penalizzazione di posizione. Un test lo dimostra su dati sintetici: con il solo errore
quadratico l'ottimo cade a 0,6 per una correlazione di 0,6, e aggiungendo il termine si
sposta verso l'ampiezza piena. Un secondo test verifica che traslare il campo lasci il
termine a zero.

Il termine e' applicato **solo alle variabili continue**: la letteratura documenta un
peggioramento delle metriche di occorrenza della precipitazione.

---

## 8. Valutazione

### 8.1 Protocollo

Validazione a **finestra mobile**, 6 fold. Uno split unico in tre blocchi contigui
avrebbe concentrato il test nella coda estiva del periodo: la neve non sarebbe stata
valutabile e la temperatura sarebbe stata misurata su un solo regime. Con i fold i
blocchi di test coprono **tutti e dodici i mesi**, mantenendo in ciascun fold l'ordine
train -> validazione -> test. Alle giunzioni si scartano 30 istanti per attenuare
l'autocorrelazione.

La calibrazione e le soglie di decisione si stimano **sulla validazione** e si misurano
**sul test**. Stimarle e misurarle sullo stesso blocco gonfierebbe il risultato.

### 8.2 Risultati sul test del fold 0

| | dwf | persistenza | **persistenza diurna** |
|---|---:|---:|---:|
| t2m RMSE (degC) | 4,45 | 4,73 | **3,16** |
| tp Brier | **0,182** | 0,261 | 0,274 |
| tp skill score | **+0,193** | -0,156 | -0,212 |
| sf Brier | **0,061** | 0,075 | 0,081 |

Lettura onesta: il modello **vince nettamente sulle probabilita'** di pioggia e neve, e
sulla temperatura **perde** contro il riferimento diurno. E' la ragione per cui e' stato
introdotto l'ancoraggio, il cui effetto e' gia' misurato nel banco comparativo.

### 8.3 Calibrazione

Isotonica, con PAVA scritto a mano. Sul test: errore di calibrazione da 0,0702 a
**0,0392**, Brier da 0,1841 a **0,1785**. Le soglie scelte sulla validazione sono 0,38
per la pioggia e 0,23 per la neve. Quest'ultima ha corretto un difetto reale: la soglia
era rimasta a 0,5 e il punteggio F1 della neve valeva 0,194; scegliendola sui dati e'
salito a **0,553**.

---

## 9. Vigo di Cadore

Cella riga 114, colonna 210 (46,50 N, 12,50 E), frazione di terra 1,00.

| | |
|---|---|
| quota del modello | 1463 m |
| quota reale del paese | 951 m |
| scarto | **512 m** |
| bias termico implicato (6,5 K/km) | circa **3,3 K** piu' freddo |

Non e' un errore del modello ne' del dato: a 0,25 gradi una cella copre circa 28 km e
media tutto il Cadore, creste comprese. Per passare dalla cella al paese serve una
correzione di quota esplicita. Il fatto e' dichiarato nel report PDF, cosi' chi legge
non scambia un limite di risoluzione per un errore di previsione.

L'escursione diurna locale misurata e' di **6,7 gradi**, quasi il doppio della media di
dominio: e' un punto severo per il modello.

---

## 10. Sicurezza

I pesi si salvano in `models/` come `.npz` letto con `allow_pickle=False`, con i
metadati in JSON separato. Il motivo e' concreto: il codice usava
`torch.load(..., weights_only=False)`, cioe' l'impostazione massimamente insicura, che
esegue codice arbitrario contenuto nel file.

La sicurezza non e' **dichiarata** ma **dimostrata**: un test costruisce un payload che
sotto pickle **si esegue davvero**, e un secondo test verifica che il caricatore del
progetto lo rifiuti **senza eseguirlo**. Senza il primo test, il secondo non
proverebbe nulla.

Le credenziali non vengono mai stampate ne' registrate. `.env`, `datasets/` e `models/`
sono fuori dal controllo di versione.

---

## 11. Difetti trovati e corretti

Molti sono miei. Sono elencati perche' il metodo conta quanto il risultato.

| difetto | come e' emerso | esito |
|---|---|---|
| Ordine degli assi nell'impilamento | `IndexError` a runtime | trasposizione esplicita |
| `log1p` inefficace sulla pioggia in metri | ispezione della scala | fattore di scala 1000 |
| 0,42 s per campione | profilazione, non intuito | `valid_time` in cache, 32 volte piu' veloce |
| Indice sbagliato in PAVA | test di monotonia | riscritto con variabili esplicite |
| Soglia neve lasciata a 0,5 | F1 assurdamente basso | scelta sulla validazione, F1 0,553 |
| Invariante theta_e sbagliato **nel test** | test rosso | corretto il test, non il codice |
| Percorso Zarr sbagliato | "non trovo i dati" | i dati c'erano, sbagliavo io |
| Manifest scritto solo a fine corsa | ispezione | scritto dopo ogni task |
| `torch.load(weights_only=False)` | revisione di sicurezza | sostituito e dimostrato sicuro |
| Etichette orarie `scadenza x 6` | correlazione non monotona | istanti non equidistanti |
| Riferimento troppo debole | conseguenza della precedente | aggiunta la persistenza diurna |
| 228 M di parametri nella variante spettrale | misura del costo | modi ridotti, ora piu' leggera di `conv` |
| Build Docker fallita all'ultimo strato | build reale | `README.md` e `LICENSE` mancanti nell'immagine |

---

## 12. Limiti noti

1. **Il modello attuale e' ancora quello non ancorato.** L'ancoraggio, la pesatura e il
   termine spettrale sono implementati e testati, e il banco comparativo ne misura
   l'effetto, ma il modello finale a scala piena va riaddestrato.
2. **Il protocollo del banco e' ridotto** (ritagli piccoli, poche passate) e misurato su
   **un solo fold**. La dispersione fra semi e' stata misurata solo per `conv` e si
   assume simile per le altre: plausibile, non verificato. I tempi per passata sono
   inoltre contaminati dal carico della macchina, quindi vanno confrontati solo entro
   la stessa corsa.
3. **Il calore latente nel report** usa il punto di rugiada dell'ultimo istante
   osservato, perche' la previsione non lo contiene. E' un'ipotesi debole su 72 ore:
   quel campo va letto come struttura spaziale, non come previsione di umidita'. Il
   limite e' stampato sulla pagina.
4. **Le mappe non hanno proporzioni geografiche fedeli**: cartopy non e' fra le
   dipendenze e a latitudini diverse la scala nord-sud e est-ovest divergono.
5. **ecCodes 2.28 nel container** contro il 2.42 raccomandato: ingerire sull'host.
6. **La chiave CDS va ruotata** se e' mai transitata in un canale non cifrato.

Il punto sull'incompletezza del 2024 e' superato: l'ingestione copre 2862 slot su 2862
attesi, dal 2024-01-01 al 2026-08-11, senza buchi.

---

## 13. Che cosa farei dopo

> Superata: il piano aggiornato e' in **[PIANO.md](PIANO.md)**, con lo stato veritiero di
> ogni voce. Quanto segue e' la lista come era prima delle misure della sezione 14 e
> resta solo come traccia storica.

1. Riaddestrare `conv` ancorata a scala piena e rivalutarla onestamente contro la
   persistenza diurna. Il banco e' chiuso: la scelta e' fatta e motivata.
2. Misurare la **curva di capacita'** invece di ingrandire la rete a intuito. Il banco
   ha gia' fornito la prima evidenza in questa direzione: 28,9 milioni di parametri
   (`recurrent`) non battono 9,9 milioni (`conv`), e costano quattro volte tanto.
3. Screening delle caratteristiche candidate contro il **cambiamento futuro**, non
   contro il valore futuro: una variabile che predice bene il valore ma non il
   cambiamento non aggiunge nulla alla persistenza.
4. Correzione di quota esplicita per il passaggio da cella a localita'.
5. Estendere i fold ai due anni completi.

---

## 14. Il modello non e' mal progettato: e' poco addestrato

Questa sezione e' posteriore alle precedenti e, dove le contraddice, prevale.

### 14.1 Il conto che nessuno aveva fatto

`samples_per_epoch: 512` con `batch_size: 4` sono **128 passi di ottimizzazione per
epoca**. Venti epoche fanno **2560 passi** per una rete da dieci milioni di parametri.

La copertura dei dati e' ancora piu' netta. Un ritaglio 96x96 e' 9216 punti su 104 661
del dominio, l'8,8%:

| | punti-griglia |
|---|---|
| disponibili in addestramento (961 finestre) | 100 579 221 |
| visti in venti epoche (10 240 ritagli) | 94 371 840 |
| rapporto | **0,94** |

In tutto l'addestramento il modello vede l'equivalente di **meno di una passata** sui
dati. La parola "epoca" nei log e' fuorviante: rivisita 961 finestre dieci volte
ciascuna guardando ogni volta un decimo del dominio, non passa venti volte sui dati.

Conseguenza per l'interpretazione di tutto quanto precede: ogni confronto fra
architetture, varianti e caratteristiche finora e' stato condotto in regime di
sotto-addestramento, dove vince chi parte meglio, non chi arriva piu' lontano. E' la
stessa ragione per cui l'ancoraggio della pioggia sembrava utile a scala ridotta.

### 14.2 Il campo recettivo efficace e' minuscolo

Derivando un punto di uscita del modello addestrato rispetto a tutto l'ingresso: il
**50% dell'influenza arriva da meno di 130 km**, il 90% da 2189 km. Un sistema di media
latitudine viaggia 500-1000 km al giorno, quindi a tre giorni l'informazione utile parte
da 1500-3000 km. La rete puo' arrivarci in teoria e non ci arriva in pratica.

Attenzione al ritaglio: addestrando su 96x96 il modello non vede **mai** nulla oltre 96
pixel, cioe' 2664 km. Non puo' imparare una relazione che non gli e' mai stata mostrata.
Parte del campo recettivo stretto puo' essere causata dal ritaglio, non dalla
convoluzione. Va separato ingrandendo la finestra vista, non riducendo i dati.

### 14.3 Cosa manca nei dati

Tutte le variabili di ingresso sono **di superficie**. L'unico geopotenziale presente e'
l'orografia statica. Il tempo alle medie latitudini e' pilotato dal flusso a 500 hPa, e
i modelli che funzionano (GraphCast, Pangu) usano pochi istanti temporali ma **molti
livelli verticali**. E' la lacuna piu' probabile fra tutte quelle elencate.

Scelta dei campi, guidata dalle variabili di riferimento di WeatherBench 2 e non
dall'intuito: **z500** (pilota il flusso), **t850** (avvezione termica, standard per la
neve), **t500** (stabilita' con t850), **q700** (umidita' disponibile). Esclusi u500 e
v500: una rete convoluzionale ricava il vento geostrofico dal gradiente di z500, quindi
sarebbero in gran parte ridondanti a costo pieno.

Verificato su file reali gia' scaricati, non solo in teoria: z500 fra 4886 e 5939 metri
geopotenziali (gennaio europeo tipico 4900-5800), t500 fra -47 e -2 C, t850 fino a +29 C
sul bordo sahariano del dominio, q700 fra 0 e 0,010 kg/kg. Anche il caso a rischio, due
variabili sullo stesso livello nello stesso GRIB, si legge correttamente.

Abilitarli porta i canali da 245 a 341.

### 14.4 Due checkpoint perduti, e la protezione che mancava

La cartella di destinazione di un addestramento **non dipende dall'architettura**:
`models/fold_00` per tutte. Due corse lanciate insieme finiscono nello stesso posto e la
seconda sovrascrive la prima appena migliora. E' accaduto: persi i pesi del modello a
piena scala e quelli del rivale, senza un solo messaggio.

Il campo per separarle, `paths.models_subdir`, esisteva: l'errore e' stato dell'operatore.
Il difetto del codice era un altro e piu' grave: i metadati **non registravano
l'architettura**, quindi un checkpoint su disco era indistinguibile da uno prodotto da
un'altra rete. Ora l'architettura sta nei metadati, il caricamento la verifica e l'avvio
rifiuta di scrivere sopra un'architettura diversa.

### 14.5 Ordine dei prossimi interventi, per effetto atteso

1. **Piu' passi di ottimizzazione.** E' il vincolo che lega tutto il resto.
2. **Piu' anni.** 2022 e 2023 in scaricamento portano le finestre da 961 a circa 1750.
3. **Livelli di pressione.** In scaricamento, ~2,4 GB.
4. **Finestra vista piu' larga**, per separare il ritaglio dall'architettura.
5. **Architettura.** Il rivale a contesto globale e' cinque volte piu' piccolo e tre
   volte piu' veloce sul dominio intero: a parita' di ore di CPU concede piu' passi, che
   per il punto 1 e' il vantaggio che conta.

### 14.6 Quanto vale il modello contro il non fare nulla

La media su nove scadenze nasconde il numero che conta. Errore quadratico medio della
temperatura in gradi, split di test, 241 finestre, confronto con la persistenza diurna
("domani come ieri alla stessa ora"):

| scadenza | ore avanti | modello | ieri stessa ora | guadagno |
|---|---|---|---|---|
| 0 | +12 | 1,866 | 2,429 | +23,2 % |
| 1 | +18 | 2,124 | 2,409 | +11,8 % |
| 2 | +24 | 2,270 | 2,404 | **+5,5 %** |
| 3 | +36 | 2,785 | 3,208 | +13,2 % |
| 4 | +42 | 2,924 | 3,232 | +9,5 % |
| 5 | +48 | 2,992 | 3,237 | +7,6 % |
| 6 | +60 | 3,278 | 3,711 | +11,7 % |
| 7 | +66 | 3,335 | 3,717 | +10,3 % |
| 8 | +72 | 3,322 | 3,685 | +9,8 % |

Il guadagno e' minimo alle scadenze multiple di 24 ore, dove la persistenza diurna
coincide con la persistenza semplice ed e' quindi al suo massimo di forza. A ventiquattro
ore il modello batte del **5,5%** l'ipotesi di non fare nulla.

Questo, e non il valore assoluto di 2,27 gradi, e' il difetto: 2,40 gradi si ottengono
senza alcun modello. Cio' che il modello ha imparato e' il ciclo giornaliero, che gli era
gia' dato dall'ancoraggio, piu' un lisciamento locale. La dinamica, cioe' il fatto che
domani arrivi aria diversa da altrove, non c'e'.

Le tre misure di questa giornata convergono: campo recettivo efficace di 130 km, nessuna
variabile in quota, 2,7 visite per finestra in tutto l'addestramento. Per imparare la
dinamica mancano contemporaneamente la portata spaziale, l'informazione sul flusso e il
tempo di calcolo. Nessuna delle tre da sola spiegherebbe il risultato.

Soglia dichiarata dall'utente: sotto 2 gradi a ventiquattro ore. Serve portare il
guadagno sulla persistenza dal 5,5% al 17%, cioe' triplicarlo.
