# Ancoraggio della probabilita' di pioggia alla persistenza diurna

Generato da `scripts/screen_occurrence_anchor.py`.

## Cosa si sta misurando

La probabilita' di pioggia del modello a scala piena varia fra la prima e
l'ultima scadenza circa nove volte meno di quanto vari la realta'. La testa non
e' collassata: nessuna coppia di scadenze e' identica. Quello che manca e' il
riferimento, che le teste gaussiane hanno e quelle di occorrenza no.

L'ampiezza e' un logit sommato al logit di occorrenza: positivo dove ieri alla
stessa ora pioveva, negativo dove non pioveva. **Ampiezza 0 e' il modello
attuale**, quindi la prima riga e' il termine di paragone, non una prova.

La colonna *movimento* e' l'escursione della probabilita' prevista fra le
scadenze divisa per l'escursione osservata: 1,00 vuol dire che il modello segue
la giornata come la realta', 0,10 che la appiattisce di dieci volte. Va letta
insieme al Brier, perche' un modello puo' migliorare il Brier restando piatto.

## Protocollo

Fold 0, 3 passate, ritaglio 64, 192 campioni per passata, semi [1234, 101].

## Risultati

| ampiezza | Brier | movimento | Brier prima scadenza | Brier ultima scadenza |
|---:|---:|---:|---:|---:|
| persistenza diurna | 0.2556 | 1.06 | 0.2259 | 0.2819 |
| 0 (attuale) | 0.2433 | 0.43 | 0.2422 | 0.2420 |
| 0.6 | 0.2007 | 0.55 | 0.1927 | 0.2055 |
| 1.1 | 0.1907 | 0.60 | 0.1775 | 0.2001 |
| 1.8 | 0.2032 | 0.62 | 0.1844 | 0.2175 |

## Lettura

Brier migliore: **ampiezza 1.1** (0.1907 contro 0.2433 del modello attuale, guadagno +0.0526).

Dispersione fra semi: 0.0032. 
Il guadagno supera la dispersione fra semi: l'ancoraggio dell'occorrenza va adottato, con questa ampiezza.

Movimento fra le scadenze: modello attuale 0.43x, migliore 0.60x, persistenza diurna 1.06x il vero.

## Limiti

1. Protocollo ridotto e un solo fold, come per gli altri banchi.
2. Il Brier e' misurato sulla validazione, non sul test: serve a scegliere, non
   a dichiarare la bravura del modello finale.
3. L'ampiezza e' fissa e uguale per tutte le scadenze. Alla scadenza piu' lunga
   la persistenza vale meno, quindi il valore migliore qui e' un compromesso.
