# Piano di sviluppo

Ogni voce ha un **branch dedicato**, una descrizione di cosa va fatto e un **criterio di
completamento** verificabile. Un passo non e' concluso perche' il codice esiste: e'
concluso quando il criterio e' soddisfatto e la verifica e' stata eseguita davvero.

Legenda stato: `[ ]` da fare, `[~]` in corso, `[x]` fatto, `[!]` bloccato.

I passi marcati **R** sono revisioni ricorrenti. Non stanno in fondo: sono intercalati,
perche' una revisione fatta solo alla fine trova problemi quando costa di piu'
correggerli.

---

## Fase 0 — chiudere il lavoro gia' aperto

Nulla di nuovo comincia finche' questa fase non e' chiusa: lasciare esperimenti a meta'
e' il modo piu' rapido per non poter piu' dire quale numero appartiene a quale codice.

- [~] **0.1 Finestra di ingresso** — `feature/input-window-screening`
  Completare il confronto a 3, 7, 10, 14 giorni e scegliere la finestra.
  *Fatto quando*: `docs/finestra-input.md` riporta le quattro impostazioni con almeno
  due semi ciascuna, il guadagno e' confrontato con il rumore fra semi (0,059 gradi) e
  la scelta e' motivata o dichiarata indistinguibile.
  *Nota*: 10 e 14 giorni erano falliti per un difetto di percorso dei checkpoint,
  corretto in `Config.fold_dir` con `paths.models_subdir`.

- [ ] **0.2 Valutazione onesta del modello a scala piena** — `feature/full-scale-evaluation`
  Il modello addestrato (fold 0, migliore all'epoca 16) va misurato sul blocco di
  **test**, mai visto, contro persistenza ingenua e persistenza diurna.
  *Fatto quando*: esiste una tabella con RMSE t2m, Brier e F1 neve per il modello e per
  entrambi i riferimenti, e i casi in cui il modello perde sono riportati, non omessi.

- [ ] **0.3 Tabella prestazioni nel README** — `docs/pipeline-and-metrics`
  Riempire la tabella lasciata vuota con i numeri di 0.2.
  *Fatto quando*: nessuna cella contiene un segnaposto, e ogni numero e' riproducibile
  con il comando indicato nella riga sopra la tabella.

- [ ] **0.4 Ingestione del riempimento 2024** — `feature/backfill-2024`
  Portare nello store i mesi scaricati e verificare la continuita' della serie.
  *Fatto quando*: `scripts/analyze_data.py` non segnala buchi inattesi e il conteggio
  degli slot presenti coincide con quello dei mesi ingeriti.

- [ ] **0.5 Prova da un capo all'altro** — `test/end-to-end`
  Scaricare, ingerire, addestrare poche epoche, valutare, prevedere, generare il PDF,
  rigenerare i due notebook, il tutto dentro l'immagine Docker.
  *Fatto quando*: la sequenza gira senza intervento manuale e il PDF prodotto e'
  leggibile.

---

## R1 — Revisione del codice e ricerca di tecnologie (dopo la Fase 0)

Due attivita' distinte che conviene fare insieme, perche' la seconda suggerisce dove
guardare nella prima.

- [ ] **R1.a Revisione** — `review/r1`
  Rileggere i moduli toccati dalla Fase 0 cercando: duplicazione di logica (e' gia'
  successo con il lettore del manifesto), percorsi costruiti a mano invece che tramite
  `Config`, valori numerici scritti in piu' punti, `except` che nascondono errori,
  funzioni che superano una schermata.
  *Fatto quando*: esiste un elenco dei difetti trovati con gravita' e file, quelli
  gravi sono corretti, gli altri sono annotati in `STATO.md`.

- [ ] **R1.b Ricerca di tecnologie utili alla previsione** — `research/r1`
  Non "leggere articoli", ma rispondere a una domanda precisa: *cosa, applicato a
  questa griglia con questa potenza di calcolo, potrebbe ridurre l'errore?*
  Piste gia' individuate e da valutare, in ordine di rapporto fra beneficio atteso e
  costo:
  1. **Perdita CRPS** al posto della gaussiana: misura la previsione probabilistica
     senza assumerne la forma.
  2. **Diffusione sui residui**: genera scenari invece della sola media, e attenua il
     problema della doppia penalita' dimostrato in `docs/ricerca.md`.
  3. **Riduzione di scala verso Vigo**: la cella e' a 1463 metri contro 951 reali; una
     correzione basata sull'altimetria puo' valere piu' di qualunque cambio di rete.
  4. **Ingressi multi-sorgente**: analisi IFS accanto a ERA5 (vedi Fase 1).
  5. **Attenzione lungo l'asse temporale** invece che spaziale: gli slot non sono
     equidistanti, e la rete attuale lo ignora.
  *Fatto quando*: ogni pista ha una riga in `docs/ricerca.md` con costo stimato,
  beneficio atteso e la ragione per cui e' stata presa o scartata; almeno una e'
  provata sul banco.

---

## Fase 1 — I dati: acquisizione automatica, archivio senza duplicati

L'obiettivo e' che il programma **si procuri da solo** cio' che gli serve, e nulla di
piu'. Chi vuole solo una previsione non deve scaricare anni di storia; chi vuole
addestrare deve poter scegliere i mesi.

- [ ] **1.1 Colmare il buco del 13-14 agosto** — `feature/opendata-backfill`
  Verificato: il mirror AWS di ECMWF Open Data conserva quei giorni, mentre il portale
  li ha gia' fatti scorrere via. Vedi `docs/tempo-reale.md`.
  *Fatto quando*: i due giorni sono nello store, marcati con la fonte di provenienza.

- [ ] **1.2 Provenienza esplicita** — `feature/data-provenance`
  Ogni slot deve sapere da dove viene: ERA5 definitivo, ERA5T preliminare, analisi IFS.
  Senza questo campo, mescolare le fonti rende impossibile capire un errore.
  *Fatto quando*: lo store ha una coordinata `fonte`, le tabelle la riportano e la
  dashboard la mostra.

- [ ] **1.3 Misura dello scarto ERA5 contro analisi IFS** — `research/era5-vs-ifs`
  Le due fonti si sovrappongono di due o tre giorni: li' lo scarto si misura invece di
  presumerlo. Va fatto **prima** di usarle insieme in produzione.
  *Fatto quando*: esiste per ogni variabile lo scarto medio, la sua dispersione e la
  struttura spaziale, confrontati con l'errore del modello.

- [ ] **1.4 Raccolta giornaliera automatica** — `feature/daily-collection`
  L'archivio aperto e' a scorrimento: quello che non si prende oggi si perde. Va aperta
  subito una raccolta ricorrente.
  *Fatto quando*: un comando solo aggiorna lo store all'ultimo slot disponibile ed e'
  idempotente, cioe' rieseguirlo non scarica nulla di gia' presente.

- [ ] **1.5 Archivio senza duplicati** — `feature/store-dedup`
  Oggi lo stesso istante puo' arrivare da piu' scaricamenti. Serve una chiave unica per
  slot, la regola di precedenza fra fonti (ERA5 definitivo batte ERA5T, che batte
  l'analisi IFS) e una verifica che **nessuna informazione venga persa** nella
  deduplicazione: quando si scarta un duplicato si registra cosa e perche'.
  *Fatto quando*: una prova con duplicati costruiti a mano dimostra che il vincitore e'
  quello atteso, che il conteggio degli slot non cambia e che lo spazio occupato cala.

- [ ] **1.6 Acquisizione su richiesta** — `feature/on-demand-fetch`
  Il pezzo che rende il progetto usabile da chi arriva per la prima volta.
  - modalita' **previsione**: dato un modello e un istante, il programma calcola da solo
    la finestra di ingresso necessaria, verifica cosa manca e scarica **solo quello**;
  - modalita' **addestramento**: l'utente sceglie i mesi, il programma calcola cosa
    manca e scarica solo la differenza.
  *Fatto quando*: da uno store vuoto, chiedere una previsione scarica un volume pari
  alla finestra e non di piu', misurato in megabyte; e chiedere di nuovo la stessa
  previsione non scarica nulla.

- [ ] **1.7 Stima prima di scaricare** — `feature/download-estimate`
  Prima di iniziare, dire quanti file, quanti megabyte e quanto tempo. La misura c'e'
  gia': circa 387 MB e nove minuti al mese.
  *Fatto quando*: la stima compare da riga di comando e sul sito, e lo scarto fra
  stimato e reale e' registrato per correggerla.

---

## Fase 2 — Il sito deve funzionare

Lo stato attuale non e' accettabile: la dashboard esiste ma non si comporta come
dovrebbe. Prima di aggiungere pagine va reso solido quello che c'e'.

- [ ] **2.1 Diagnosi** — `fix/dashboard-reliability`
  Aprire ogni sezione con lo store reale e annotare cosa non funziona: errori, attese
  lunghe, numeri sbagliati, grafici vuoti. Nessuna correzione prima di avere l'elenco.
  *Fatto quando*: l'elenco esiste, con la causa di ciascun problema, non solo il sintomo.

- [ ] **2.2 Correzioni** — stesso branch
  *Fatto quando*: ogni voce dell'elenco e' chiusa o dichiarata non risolvibile con la
  ragione; nessuna sezione mostra una traccia di errore all'utente.

- [ ] **2.3 Comportamento quando manca qualcosa** — stesso branch
  Store vuoto, modello assente, previsione mai eseguita: il sito deve **dire cosa fare**,
  non rompersi ne' mostrare una pagina bianca.
  *Fatto quando*: esiste una prova che parte da una cartella dati vuota e ottiene
  istruzioni leggibili in ogni sezione.

- [ ] **2.4 Tempi di risposta** — stesso branch
  Misurare quanto impiega ogni sezione. Quello che supera i due secondi va messo in
  cache o calcolato prima.
  *Fatto quando*: esiste la tabella dei tempi e nessuna sezione supera i due secondi al
  secondo caricamento.

---

## Fase 3 — Controllo dell'addestramento dal sito

- [ ] **3.1 Vedere i dati di ingresso** — `feature/dashboard-training`
  Scegliere istante e variabile, vedere le mappe dei canali che entrano davvero nella
  rete, compresi quelli costruiti (solari, termodinamici, tendenze), con i valori in
  unita' fisiche e non normalizzati.
  *Fatto quando*: si puo' navigare l'intera finestra di ingresso e ogni canale mostra
  nome, unita' e provenienza.

- [ ] **3.2 Vedere i dati di uscita** — stesso branch
  Le nove scadenze, media e incertezza, accanto all'osservato quando esiste.
  *Fatto quando*: previsto, osservato e differenza sono visibili insieme per ogni
  scadenza e variabile.

- [ ] **3.3 Modificare i parametri** — stesso branch
  Epoche, passo di apprendimento, dimensione del ritaglio, campioni per epoca, variante,
  pesi della perdita, seme. I valori vanno **validati con lo stesso schema** della
  configurazione: il sito non deve poter costruire una configurazione che il programma
  rifiuterebbe.
  *Fatto quando*: un valore fuori intervallo viene rifiutato con lo stesso messaggio che
  darebbe da riga di comando.

- [ ] **3.4 Avviare e seguire l'addestramento** — stesso branch
  Avvio, arresto, avanzamento, curva di perdita che si aggiorna, uso di processore e
  memoria. L'addestramento gira in un processo separato: il sito non deve bloccarsi.
  *Fatto quando*: si avvia un addestramento breve dal sito, lo si segue, lo si ferma, e
  il modello prodotto e' caricabile.

- [ ] **3.5 Confronto fra esecuzioni** — stesso branch
  Ogni esecuzione salva configurazione e risultato in una cartella propria (esiste gia'
  `paths.models_subdir`). Il sito le elenca e le confronta.
  *Fatto quando*: due esecuzioni con parametri diversi sono confrontabili in una tabella
  senza toccare il filesystem a mano.

---

## R2 — Revisione e ricerca (dopo la Fase 3)

- [ ] **R2.a Revisione** — `review/r2`
  Concentrata sul confine fra sito e libreria: il sito non deve contenere logica di
  calcolo, deve chiamarla. Se una funzione esiste solo per la dashboard, e' nel posto
  sbagliato.
- [ ] **R2.b Ricerca** — `research/r2`
  Rivalutare le piste di R1.b con quello che nel frattempo si e' imparato, e provarne
  almeno un'altra.

---

## Fase 4 — Cambiamento climatico

Il progetto ha una serie oraria su un dominio ampio: e' materiale adatto a mostrare
tendenze, purche' si dica con onore quanto sono solide.

- [ ] **4.1 Tendenze di base** — `feature/dashboard-climate`
  Temperatura media per anno e per stagione, con retta di tendenza e sua incertezza,
  sul dominio intero, su sottoregioni e sulla cella di Vigo.
  *Fatto quando*: ogni tendenza riporta pendenza, intervallo di confidenza e numero di
  anni su cui e' calcolata.

- [ ] **4.2 Mappa delle tendenze** — stesso branch
  Il riscaldamento non e' uniforme: la mappa per cella lo mostra meglio di qualunque
  media.
  *Fatto quando*: esiste la mappa con scala divergente centrata sullo zero e le zone non
  significative sono distinte.

- [ ] **4.3 Estremi e giorni caratteristici** — stesso branch
  Giorni di gelo, notti tropicali, precipitazione intensa, spessore di neve: conteggi
  per anno.
  *Fatto quando*: ogni indice ha la sua definizione scritta accanto al grafico.

- [ ] **4.4 Onesta' statistica** — stesso branch
  Con pochi anni una tendenza climatica **non e' significativa**. Va detto sulla pagina,
  non nascosto.
  *Fatto quando*: la pagina dichiara il periodo coperto e avverte quando e' troppo breve
  per concludere.

---

## Fase 5 — Orizzonte di previsione piu' lungo

- [ ] **5.1 Fin dove ha senso** — `research/longer-horizon`
  Misurare quando il modello smette di battere la persistenza diurna e la climatologia.
  Allungare oltre quel punto produce numeri, non previsioni.
  *Fatto quando*: esiste la curva dell'errore per scadenza con le due soglie segnate.

- [ ] **5.2 Estensione** — `feature/longer-horizon`
  Portare l'uscita da 3 a 5-7 giorni se 5.1 lo giustifica, valutando le due strade:
  uscita diretta piu' larga, oppure applicazione ripetuta del modello su se stesso.
  *Fatto quando*: le due strade sono confrontate sullo stesso blocco di test.

- [ ] **5.3 Confronto con IFS** — `feature/ifs-baseline`
  Le previsioni IFS a 144 ore sono gratuite e sono lo stato dell'arte. Come riferimento
  sono molto piu' severe della persistenza.
  *Fatto quando*: la valutazione riporta tre riferimenti: ingenuo, diurno, IFS.

---

## Fase 6 — Capire il modello

Richiesta esplicita: si deve poter capire come funziona e cosa c'e' dietro.

- [ ] **6.1 Peso degli ingressi** — `feature/explainability-inputs`
  Quali canali contano, per variabile e per scadenza, con occlusione o gradiente.
  *Fatto quando*: esiste la classifica per variabile prevista, con il metodo dichiarato
  e i suoi limiti.

- [ ] **6.2 Portata spaziale** — stesso branch
  Da quanto lontano arriva l'informazione che determina una cella. Si misura
  perturbando e osservando dove cambia l'uscita.
  *Fatto quando*: esiste la mappa di influenza per almeno una cella significativa (Vigo).

- [ ] **6.3 Anatomia della rete** — `feature/explainability-model`
  Schema dei blocchi, forme dei tensori, parametri per blocco, cosa fa ogni testa.
  *Fatto quando*: si puo' seguire il percorso di un tensore dall'ingresso all'uscita
  senza leggere il codice.

- [ ] **6.4 Dove sbaglia** — `feature/explainability-errors`
  L'errore per regione, stagione, ora del giorno, quota, mare contro terra.
  *Fatto quando*: esistono le scomposizioni e almeno un regime di errore e' spiegato.

- [ ] **6.5 Perche' questa previsione** — stesso branch
  Per una previsione specifica: quanto viene dall'ancoraggio diurno e quanto dalla
  correzione della rete.
  *Fatto quando*: la scomposizione compare accanto alla mappa prevista.

---

## R3 — Revisione e ricerca (dopo la Fase 6)

- [ ] **R3.a Revisione** — `review/r3`
- [ ] **R3.b Ricerca** — `research/r3`
  Ultima occasione per introdurre una tecnica prima del consolidamento.

---

## Fase 7 — Il sito come punto centrale

Vincolo da rispettare: **tutto deve restare eseguibile da Python**. Il sito e' una via
d'accesso in piu', non l'unica; se una funzione esiste solo li', e' un errore di
struttura.

- [ ] **7.1 Struttura delle pagine** — `feature/dashboard-hub`
  Riorganizzare in pagine vere invece di una sequenza di sezioni: Panoramica, Dati,
  Modello, Addestramento, Previsione, Valutazione, Clima, Sistema.
  *Fatto quando*: ogni pagina ha uno scopo dichiarato e nessuna informazione compare in
  due punti con due valori diversi.

- [ ] **7.2 Previsione dal sito** — stesso branch
  Scegliere modello e istante, far scaricare al programma quello che manca (Fase 1.6),
  ottenere la previsione, scaricare il PDF.
  *Fatto quando*: da store vuoto e senza toccare il terminale si arriva al PDF.

- [ ] **7.3 Gestione dei dati dal sito** — stesso branch
  Vedere cosa c'e', scegliere i mesi, avviare lo scaricamento, seguirlo.
  *Fatto quando*: le operazioni della Fase 1 sono disponibili dal sito con la stessa
  validazione della riga di comando.

- [ ] **7.4 Tutte le metriche, ordinate** — stesso branch
  Le infografiche gia' prodotte (calibrazione, affidabilita', mappe di errore, confronto
  fra varianti, schermatura delle caratteristiche) vanno raccolte dove servono, non
  ammucchiate.
  *Fatto quando*: ogni grafico prodotto dal progetto e' raggiungibile dal sito e ha una
  riga che dice cosa mostra e come leggerlo.

- [ ] **7.5 Parita' fra le due vie** — stesso branch
  *Fatto quando*: esiste la tabella che per ogni operazione indica il comando Python e
  la pagina corrispondente, e non ci sono caselle vuote.

---

## Fase 8 — Consegna

- [ ] **8.1 Riordino dei documenti** — `docs/reorganization`
  Documenti per l'utente in `docs/`, documenti per lo sviluppo in `agent/`. Richiede di
  aver prima riunito i branch: **serve l'autorizzazione esplicita dell'utente** per i
  merge.
- [ ] **8.2 Relazione finale** — `docs/final-report`
  Molto dettagliata: cosa fa, come, con quali risultati misurati, con quali limiti.
- [ ] **8.3 Prova a freddo** — `test/cold-start`
  Da macchina pulita: installazione, primo avvio, prima previsione. Cronometrata.
  *Fatto quando*: un lettore che non conosce il progetto arriva a una previsione
  seguendo solo il README.

---

## Come si aggiorna questo piano

Chi lavora segna `[~]` quando comincia e `[x]` quando il criterio e' soddisfatto. Un
passo che si rivela sbagliato non si cancella: si marca superato e si scrive perche'.
Passi nuovi si aggiungono nella fase pertinente con lo stesso formato.
