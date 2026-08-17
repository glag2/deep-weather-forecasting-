# Stato del progetto

Aggiornato: 2026-08-18. Chi modifica qualcosa di sostanziale riscrive questo file.

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
| Test automatici | 698 | `pytest -q` |

**Gli slot non sono equidistanti**: 06, 12, 18 UTC significa 6, 6 e 12 ore. Tre slot
sono un giorno, non diciotto ore. Un errore di etichetta su questo punto ha gia' fatto
sembrare non monotona la correlazione con la scadenza.

## Cosa gira adesso

| processo | stato |
|---|---|
| Scaricamento 2024 | in corso, 12 mesi su 23 |
| Confronto 10 e 14 giorni | rilanciato dopo la correzione dei percorsi |
| Addestramento a scala piena | concluso, migliore all'epoca 16, in `models/fold_00` |

## Problemi aperti

1. **Il modello perde sulla temperatura** contro la persistenza diurna: 4,45 contro
   3,16 gradi sul test del fold 0 prima dell'ancoraggio. L'ancoraggio diurno ha ridotto
   molto lo scarto ma la verifica sul modello a scala piena non e' ancora stata fatta.
2. **Il sito non si comporta come dovrebbe.** Diagnosi da fare, e' il passo 2.1 del piano.
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

## Da dove ripartire

`PIANO.md`, Fase 0. I passi sono in ordine e ciascuno dice quando e' concluso.
