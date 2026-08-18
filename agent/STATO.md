# Stato del progetto

Aggiornato: 2026-08-19, mattina. Chi modifica qualcosa di sostanziale riscrive questo file.

## In una frase

La pipeline completa esiste e gira: scaricamento, ingestione, costruzione dei canali,
addestramento, valutazione, previsione, calibrazione, PDF, sito locale. L'addestramento
a scala piena e' **concluso e valutato sul test**: il modello ora **batte la persistenza
diurna sulla temperatura a tutte e nove le scadenze** (2,917 contro 3,160 gradi di
RMSE). Restano due debolezze misurate: **perde come classificatore sulla neve** e la
sua **probabilita' di pioggia dipende troppo poco dalla scadenza**.

## Numeri che servono per orientarsi

| grandezza | valore | dove si verifica |
|---|---|---|
| Griglia | 261 x 401, passo 0,25 gradi | `configs/default.yaml` |
| Slot catalogati | 2862 | `scripts/analyze_data.py` |
| Slot presenti nello store | 1458 | stesso comando, colonna `usable` |
| Canali in ingresso (7 giorni) | 245 | `InputLayout.from_config` |
| Canali in uscita | 45 | `OutputLayout` |
| Parametri del modello | 9 979 053 | `scripts/benchmark_model.py` |
| Tempo per epoca a scala piena | 266 s (mediana su 24) | `models/fold_00/history.json` |
| Epoca migliore / validazione | 16 / -0,7567 | stesso file |
| Finestre nel blocco di test | 241 | `models/fold_00/metrics.parquet` |
| Rumore fra semi sul banco | 0,059 gradi | `docs/varianti.md` |
| Rumore fra semi sul banco dei giorni | 0,045 gradi | `INPUT_DAYS.md` |
| Test automatici | circa 740 | `pytest -q` |

**Gli slot non sono equidistanti**: 06, 12, 18 UTC significa 6, 6 e 12 ore. Tre slot
sono un giorno, non diciotto ore. Un errore di etichetta su questo punto ha gia' fatto
sembrare non monotona la correlazione con la scadenza.

## Cosa gira adesso

| processo | stato |
|---|---|
| Scaricamento 2024 | **concluso**, 23 mesi su 23 |
| Confronto sui giorni di ingresso | **concluso**, tutte e quattro le impostazioni |
| Addestramento a scala piena | **concluso**, 24 epoche, migliore la 16 con -0,7567 |
| Valutazione sul test | **conclusa**, 241 finestre, fold 0 |

Nessun processo lungo e' in esecuzione.

Il modello che risultava concluso all'epoca 16 **e' stato distrutto** da una prova del
banco che scriveva nella stessa cartella. La cartella e' conservata come
`models/fold_00_contaminato` a scopo di prova e la dashboard ha ora un controllo,
`coerenza_artefatti`, che riconosce esattamente quella situazione. L'addestramento
rilanciato da zero ha riprodotto **esattamente** lo stesso risultato (stesso seme):
epoca migliore 16, validazione -0,7567. La riproducibilita' e' quindi verificata per
incidente.

## Risultati sul test, fold 0, 241 finestre

I due riferimenti sono `persistence` (ripeti l'ultimo slot) e `persistence_diurnal`
(ripeti lo stesso slot del giorno prima). Il secondo e' quello serio.

| grandezza | modello | diurna | ingenua |
|---|---|---|---|
| t2m RMSE (gradi) | **2,917** | 3,160 | 4,733 |
| t2m MAE (gradi) | **2,058** | 2,165 | 3,144 |
| pioggia Brier | **0,181** | 0,274 | 0,261 |
| pioggia errore di calibrazione | **0,053** | 0,274 | 0,261 |
| pioggia F1 | **0,635** | 0,609 | 0,624 |
| neve Brier | **0,066** | 0,081 | 0,075 |
| neve F1 | 0,493 | 0,561 | **0,590** |
| neve accuratezza | 0,847 | 0,919 | **0,925** |

Sulla temperatura vince a **tutte e nove** le scadenze: 2,02 contro 2,43 alla prima,
3,40 contro 3,68 all'ultima. L'errore cresce a gradini di tre slot, cioe' al cambio di
giorno, come ci si aspetta con l'ancoraggio diurno.

L'accuratezza sulla neve **non va usata**: la frequenza di base e' 0,091, quindi
rispondere sempre "no" darebbe 0,909, piu' del modello e di entrambi i riferimenti.

## Deciso da poco, con la misura che lo sostiene

| decisione | misura |
|---|---|
| Finestra di ingresso a **7 giorni** | guadagni: 3 g -0,033; 7 g +0,065; 10 g -0,050; 14 g +0,001. Sette e quattordici sono indistinguibili (margine 0,063 contro incertezza 0,090), quindi vince la piu' economica: 245 canali invece di 455. |
| Il buco del 13-14 agosto **e' recuperabile** | il mirror AWS di ECMWF Open Data ha quei giorni e arriva indietro fino almeno a meta' 2023, con tutte e 11 le variabili nell'indice. |
| L'addestramento si comanda dal sito | ogni corsa e' un processo separato con cartella e configurazione proprie, verificato avviandone una vera dal browser. |
| Ogni checkpoint porta l'**impronta dei dati** con cui e' stato addestrato | `data_fingerprint` registra lunghezze delle finestre, slot catalogati e utilizzabili e i confini per split; `confronta_impronte` segnala lo scostamento. Serve perche' reingerire il 2024 **sposta i confini dei fold**: senza impronta si confronterebbero in silenzio modelli addestrati su dati diversi. |

## Problemi aperti

1. ~~Il modello perde sulla temperatura contro la persistenza diurna.~~ **Risolto e
   verificato a scala piena**: 2,917 contro 3,160 gradi sul test, vittoria a tutte e
   nove le scadenze. L'ancoraggio diurno era la correzione giusta.
1-bis. **La probabilita' di pioggia dipende troppo poco dalla scadenza.** Il Brier del
   modello passa da 0,168 alla prima scadenza a 0,188 all'ultima, cioe' peggiora del
   12 % in tre giorni, mentre la persistenza ingenua passa da 0,165 a 0,297, cioe'
   dell'80 %. Alla **prima** scadenza il modello quindi **perde** contro il semplice
   ripetere l'ultima osservazione. Il campo di probabilita' e' ricco nello spazio ma
   quasi statico nel tempo: assomiglia a una climatologia condizionata dallo stato
   iniziale. E' il comportamento previsto dalla letteratura sulla doppia penalita'
   (`RESEARCH.md`): sfocare minimizza l'errore quadratico su un campo caotico. Si
   corregge cambiando la funzione obiettivo, non i dati. **Questo e' ora il problema
   aperto piu' interessante.**
1-ter. **Sulla neve il modello perde come classificatore**: F1 0,493 contro 0,590. Ha
   richiamo 0,818 contro 0,595 ma precisione 0,352 contro 0,585, cioe' segnala molto e
   sbaglia spesso. La sua *probabilita'* resta pero' migliore (Brier 0,066 contro
   0,075): l'informazione c'e', e' la soglia a essere spostata verso la cautela. Prima
   di "aggiustarlo" decidere se si vuole un allarme prudente o un classificatore
   bilanciato: sono obiettivi diversi.
2. ~~Il sito non si comporta come dovrebbe.~~ **Risolto.** Diagnosi eseguita sezione per
   sezione con i tempi misurati: la struttura dei fold restituiva 12 690 righe, la
   tabella delle metriche di validazione era vuota senza dirlo, la mappa degli errori
   costava 46 s a ogni interazione e la pagina del modello leggeva chiavi che il
   checkpoint non scrive. Corretti tutti; aggiunte le pagine Clima e Addestramento.
3. **Le architetture non sono distinguibili** fra loro: lo scarto fra le cinque varianti
   e' 0,024 gradi contro un rumore fra semi di 0,059. Dichiarare un vincitore
   significherebbe leggere il seme.
4. **Vigo di Cadore e' a 1463 metri nel modello contro 951 reali.** Nessun cambio di
   rete compensa 512 metri di quota.
5. **Il token CDS va ruotato**: e' stato incollato in chiaro in una conversazione.

## Difetti trovati e corretti, da non reintrodurre

Elenco parziale, i piu' istruttivi. Dettagli in `DECISIONI.md`.

- `paths.artifacts_subdir` non influenzava `fold_dir`: ogni prova del banco scriveva
  nella stessa cartella. Ha rotto il confronto sui giorni quando un secondo
  addestramento girava in parallelo. Corretto con `paths.models_subdir` e due test.
- `scripts/download_era5.py` aveva una copia propria del lettore del manifesto, quella
  rigida, mentre in `freshness` esisteva gia' quella tollerante. Il riempimento moriva
  su un manifesto vecchio.
- `persistence.py` rifiutava `complex64`, quindi le varianti spettrali non si
  salvavano. Il criterio giusto non e' "reale" ma "rappresentazione binaria a
  dimensione fissa".
- La dashboard contava gli slot **catalogati** invece di quelli presenti: 2862 invece
  di 1458.
- `.streamlit/config.toml` scritto da PowerShell con il BOM veniva ignorato in
  silenzio. Vale per qualunque file di configurazione scritto da PowerShell.
- La dashboard era raggiungibile dalla rete. Ora e' vincolata a 127.0.0.1.
- `folds.parquet` congela la lunghezza della finestra usata al momento dell'ingestione.
  Allungarla dopo faceva leggere partenze validate per una finestra piu' corta, le cui
  code cadevano su slot mai ingeriti: perdita non finita dalla prima epoca e nessun
  checkpoint, in silenzio. Visibile solo a 10 e 14 giorni (14 % e 33 % delle finestre),
  invisibile a 3 e 7. Ora `sample_starts` ricontrolla e una perdita non finita solleva.
- `model_copy(update=...)` **non rivalida**: avrebbe accettato zero epoche e un passo di
  apprendimento negativo dal modulo web. I parametri si ricostruiscono con
  `model_validate`.
- Il nome della variante non era validato: un nome inesistente falliva solo alla
  costruzione della rete, dopo la lettura dei dati. Ora lo verifica lo schema.
- Un numero di processo, da solo, non identifica un processo: vengono riciclati. Lo
  stato di una corsa usa la coppia numero piu' istante di nascita.
- Il verdetto del banco sui giorni confrontava una differenza fra **medie** con la
  dispersione di una **singola** misura, dichiarando reale meta' del rumore.

## Da dove ripartire

`PIANO.md`, Fase 0. I passi sono in ordine e ciascuno dice quando e' concluso.

Ordine consigliato adesso che la valutazione e' congelata:

1. **Ingerire il 2024 gia' scaricato.** Va fatto ora, non prima: sposta i confini dei
   fold e invalida i numeri qui sopra. L'impronta dei dati rendera' lo spostamento
   visibile invece che silenzioso. Dopo l'ingestione **tutte** le metriche vanno
   rigenerate e questo file riscritto.
2. **Attaccare la piattezza della probabilita' di pioggia** (problema 1-bis). E' un
   lavoro sulla funzione obiettivo. La perdita spettrale in `weighting.py` esiste gia'
   ma non e' attiva nella configurazione predefinita.
3. Colmare il 13-15 agosto dal mirror AWS e misurare lo scarto ERA5 contro IFS sulla
   sovrapposizione, che ora e' molto piu' ampia.
4. Consolidare i dodici rami. **Richiede una decisione dell'utente**: nessun merge e
   nessun push sono stati eseguiti.
