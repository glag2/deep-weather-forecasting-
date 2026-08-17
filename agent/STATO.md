# Stato del progetto

Aggiornato: 2026-08-18, notte. Chi modifica qualcosa di sostanziale riscrive questo file.

## In una frase

La pipeline completa esiste e gira: scaricamento, ingestione, costruzione dei canali,
addestramento, valutazione, previsione, calibrazione, PDF, sito locale. Il modello
batte la persistenza ingenua e vince nettamente sulle probabilita' (pioggia, neve), ma
**sulla temperatura non batte ancora la persistenza diurna**, che e' il riferimento
serio. Questo e' il problema aperto piu' importante.

## Numeri che servono per orientarsi

| grandezza | valore | dove si verifica |
|---|---|---|
| Griglia | 261 x 401, passo 0,25 gradi | `configs/default.yaml` |
| Slot catalogati | 2862 | `scripts/analyze_data.py` |
| Slot presenti nello store | 1458 | stesso comando, colonna `usable` |
| Canali in ingresso (7 giorni) | 245 | `InputLayout.from_config` |
| Canali in uscita | 45 | `OutputLayout` |
| Parametri del modello | 9 979 053 | `scripts/benchmark_model.py` |
| Tempo per epoca a scala piena | circa 420 s | `history.json` |
| Rumore fra semi sul banco | 0,059 gradi | `docs/varianti.md` |
| Rumore fra semi sul banco dei giorni | 0,045 gradi | `INPUT_DAYS.md` |
| Test automatici | 731 | `pytest -q` |

**Gli slot non sono equidistanti**: 06, 12, 18 UTC significa 6, 6 e 12 ore. Tre slot
sono un giorno, non diciotto ore. Un errore di etichetta su questo punto ha gia' fatto
sembrare non monotona la correlazione con la scadenza.

## Cosa gira adesso

| processo | stato |
|---|---|
| Scaricamento 2024 | **concluso**, 23 mesi su 23 |
| Confronto sui giorni di ingresso | **concluso**, tutte e quattro le impostazioni |
| Addestramento a scala piena | in corso, epoca 15 su 24, migliore validazione -0,44 |

Il modello che risultava concluso all'epoca 16 **e' stato distrutto** da una prova del
banco che scriveva nella stessa cartella. La cartella e' conservata come
`models/fold_00_contaminato` a scopo di prova e la dashboard ha ora un controllo,
`coerenza_artefatti`, che riconosce esattamente quella situazione. L'addestramento e'
stato rilanciato da zero.

## Deciso da poco, con la misura che lo sostiene

| decisione | misura |
|---|---|
| Finestra di ingresso a **7 giorni** | guadagni: 3 g -0,033; 7 g +0,065; 10 g -0,050; 14 g +0,001. Sette e quattordici sono indistinguibili (margine 0,063 contro incertezza 0,090), quindi vince la piu' economica: 245 canali invece di 455. |
| Il buco del 13-14 agosto **e' recuperabile** | il mirror AWS di ECMWF Open Data ha quei giorni e arriva indietro fino almeno a meta' 2023, con tutte e 11 le variabili nell'indice. |
| L'addestramento si comanda dal sito | ogni corsa e' un processo separato con cartella e configurazione proprie, verificato avviandone una vera dal browser. |

## Problemi aperti

1. **Il modello perde sulla temperatura** contro la persistenza diurna: 4,45 contro
   3,16 gradi sul test del fold 0 prima dell'ancoraggio. L'ancoraggio diurno ha ridotto
   molto lo scarto ma la verifica sul modello a scala piena non e' ancora stata fatta.
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
