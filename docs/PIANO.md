# Piano di lavoro

Documento vivo. Sostituisce la sezione "Che cosa farei dopo" di `PROGRESS.md`, che era
rimasta indietro. Ogni voce ha uno stato veritiero: se una cosa non e' stata misurata, lo
dice.

Legenda: **fatto** verificato con una prova eseguibile · **in corso** avviato · **da
fare** non iniziato · **incerto** fatto ma non validato a piena scala.

---

## 1. Difetti trovati e chiusi

| | difetto | prova che era reale |
|---|---|---|
| fatto | `sample_starts` filtrava una colonna congelata: tutte le lunghezze di input davano 961 finestre | quattro configurazioni diverse, stesso conteggio |
| fatto | campi statici indicizzati per slot: `lsm` medio 0,41745 invece di 0,5200 | confronto con i valori veri dello store |
| fatto | l'ancoraggio diurno valeva solo per la temperatura, le altre teste sollevavano un'eccezione silenziata | Brier identico alla prima e all'ultima scadenza |
| fatto | `diurnal_reference_index` non controllava il limite superiore | test di proprieta' su tutte le combinazioni plausibili |
| fatto | un addestramento cancellava il checkpoint di un'altra architettura senza dirlo | due checkpoint perduti davvero |
| fatto | i metadati del checkpoint non registravano l'architettura | un file di pesi era indistinguibile da quello di un'altra rete |
| fatto | il ritaglio di validazione cambiava a ogni epoca, quindi la scelta dell'epoca migliore conteneva rumore | due reti diverse crollano entrambe **esattamente** all'epoca 9: 2,609 -> 1,303 e 2,299 -> 0,976 |
| fatto | ogni lotto conteneva quattro ritagli della **stessa** finestra: il gradiente descriveva un solo giorno | lettura del campionatore, poi `windows_per_batch` |
| fatto | la relazione affermava che le impostazioni escluse erano peggiori quando nessuna era stata esclusa | lettura del modello di relazione |
| fatto | il banco a tre epoche premia per costruzione i termini a priori | l'ancoraggio della pioggia adottato a 0,1907 e smentito a piena scala, 0,181 contro 0,192 |

## 2. Il difetto principale, ancora aperto

**Il modello guadagna il 5,5% sul non fare nulla a ventiquattro ore.** Non e' che
l'errore sia 2,27 gradi: e' che 2,40 gradi si ottengono ripetendo ieri alla stessa ora.
Dettaglio per scadenza in `PROGRESS.md` sezione 14.6.

Tre cause misurate, che vanno affrontate insieme perche' nessuna spiega il risultato da
sola:

| stato | causa | misura |
|---|---|---|
| in corso | poco addestramento | 2560 passi, 2,7 visite per finestra, 0,94 passate sui dati |
| in corso | nessuna variabile in quota | tutti gli ingressi sono di superficie, l'unico geopotenziale e' l'orografia statica |
| da fare | portata spaziale effettiva insufficiente | 50% dell'influenza entro 130 km, contro 500-1000 km al giorno di una struttura sinottica |

## 3. In corso adesso

| stato | voce | note |
|---|---|---|
| in corso | scarico livelli di pressione 2024-2026 | 96 richieste, ~2,4 GB, z500 t850 t500 q700 |
| in corso | scarico 2022 e 2023 | 121 richieste, ~11 GB, porta le finestre da 961 a circa 1750 |
| in corso | rifacimento interfaccia dashboard | delegato a un sotto-agente |
| da fare | corsa lunga sulla rete globale | 12.288 passi contro 2560, configurazione pronta |

## 4. Verifiche mai fatte, in ordine di rischio

Queste sono le cose che potrebbero essere rotte senza che nessuno lo sappia.

| stato | che cosa verificare | perche' e' rischioso |
|---|---|---|
| da fare | previsione a dominio intero con la rete globale | mai eseguita: `predict` ed `evaluate` non l'hanno mai vista, e l'attenzione su 1683 token e' un percorso nuovo |
| da fare | la valutazione su test usa ritagli casuali? | la validazione lo faceva, e se lo fa anche la valutazione tutte le metriche pubblicate contengono rumore |
| da fare | ingestione dei livelli di pressione su un mese completo | provata solo la lettura di un file, non la scrittura nello Zarr con 341 canali |
| da fare | `num_workers > 0` su Windows | la cache del lettore e' per processo: con piu' processi il risparmio di letture potrebbe svanire |
| da fare | coerenza checkpoint-configurazione in dashboard, predict, report | il controllo sull'architettura esiste solo in `load_checkpoint` |
| da fare | `weight_decay: 1e-5` | valore mille volte piu' basso del tipico per AdamW, mai giustificato; con piu' passi il sovradattamento cresce |
| da fare | quanta parte del tempo per epoca e' lettura e quanta calcolo | tutte le decisioni sul budget si basano su una stima, non su una misura |
| da fare | Docker end to end | ecCodes 2.28 nel container contro 2.42 raccomandato |
| da fare | i notebook girano ancora? | non rigenerati dopo i cambi di layout dei canali |

## 5. Migliorie da provare, in ordine di effetto atteso

| stato | voce | perche' |
|---|---|---|
| da fare | piu' passi di ottimizzazione | e' il vincolo che lega tutto il resto |
| da fare | piu' anni di dati | 2022-2023 in arrivo |
| da fare | variabili in quota, 245 -> 341 canali | il flusso a 500 hPa e' cio' che pilota il tempo alle medie latitudini |
| da fare | finestra di ritaglio piu' larga | separa il ritaglio dall'architettura come causa del campo recettivo stretto; **da fare ingrandendo**, mai riducendo i dati |
| incerto | rete a nucleo globale | vince a nove epoche su dieci ed e' 2,4 volte piu' veloce, ma dodici epoche sono poche |
| da fare | andamento del passo di apprendimento a coseno | implementato e spento, mai misurato qui |
| da fare | ancoraggio avvettato invece che diurno | il riferimento attuale ignora che l'aria si sposta |
| da fare | dorsali preaddestrate con timm o torchvision | da provare **appaiato**, preaddestrato contro casuale, altrimenti forma e pesi restano confusi |
| da fare | ottimizzatore Muon e residui in stile mHC | dal lavoro su DeepSeek-V4, `RESEARCH.md` sezione 7 |
| da fare | media di piu' semi | riduzione dell'errore tipica del 3-8%, costo lineare in corse |

## 6. Chiusura del progetto

| stato | voce |
|---|---|
| da fare | prova a piena scala finale, per volonta' dell'utente rinviata alla fine |
| da fare | relazione PDF con data e ora nel nome |
| da fare | consolidare i branch e preparare il push, con l'utente che approva |
| da fare | ricordare all'utente la rotazione del token CDS |
| da fare | indice della documentazione in `docs/` |

## 7. Regole imposte dall'utente, da non violare

1. **Tutti i dati disponibili.** Sette giorni di ingresso, nessuna riduzione della
   finestra o del periodo per comodita' di prova.
2. **Sorgente unica**: ERA5 dal CDS, mai cambiare.
3. **Nessun push** senza approvazione esplicita immediatamente prima.
4. **Un branch per compito**, commit atomici, messaggi in inglese.
5. **Le anomalie si spiegano, non si correggono**: se un dato non torna, l'errore e'
   nell'interpretazione fino a prova contraria.
6. **Non stare in attesa**: se un calcolo e' in corso, si lavora su altro.
7. **Soglia dichiarata**: sotto 2 gradi di errore a ventiquattro ore.
