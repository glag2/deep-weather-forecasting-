# Quanti giorni di storico in ingresso

Generato da `scripts/screen_input_days.py`.

## Perche' non basta confrontare l'errore

Allungare la finestra di ingresso cambia **quali** finestre esistono: una storia
di 14 giorni scarta l'inizio del periodo, quindi il blocco di validazione non e'
piu' lo stesso e gli errori grezzi non sono confrontabili.

Ogni impostazione e' percio' misurata contro la persistenza diurna calcolata
**sulle sue stesse finestre**, e si confronta il **guadagno** su quel
riferimento. Ogni impostazione e' ripetuta con piu' semi, perche' il rumore fra
ripetizioni e' dello stesso ordine delle differenze attese: il banco delle
varianti lo misura in 0,059 degC, e qui viene stimato di nuovo sui propri dati
invece di essere dato per noto.

## Protocollo

Fold 0, 3 passate, ritaglio 64, 192 campioni per passata, semi [1234, 101].

## Risultati

| giorni | slot | canali | tendenze | finestre | riferimento | RMSE | guadagno | s/ep |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 3 | 9 | 116 | [1, 3] | 64 | 3.386 | 3.419 | **-0.033** | 108 |
| 7 | 21 | 245 | [1, 3, 9] | 64 | 3.669 | 3.604 | **+0.065** | 194 |
| 10 | 30 | 335 | [1, 3, 9] | 55 | 3.637 | 3.688 | **-0.050** | 156 |
| 14 | 42 | 455 | [1, 3, 9] | 43 | 3.470 | 3.469 | **+0.001** | 210 |

## Lettura

Guadagno maggiore: **7 giorni** (+0.065 degC sul proprio riferimento).

Il margine sul secondo classificato e' 0.063 degC, **inferiore** ai 0.090 degC di incertezza a due deviazioni standard (dispersione fra semi 0.045 degC, misurata qui): la differenza non e'
distinguibile dal caso a questo budget.

Impostazioni non separabili dalla migliore: 7 giorni, 14 giorni. Fra queste si sceglie la piu' economica, cioe' **7 giorni** (245 canali), perche' a parita' di risultato misurabile costa meno memoria e lascia piu'
finestre utilizzabili. Le altre impostazioni restano fuori: non sono
equivalenti, sono misurabilmente peggiori.

## Limiti

1. Protocollo ridotto e un solo fold, come per il confronto fra varianti.
2. Il numero di canali cresce linearmente con gli slot di ingresso: una
   finestra lunga non e' solo piu' informativa, e' anche piu' difficile da
   addestrare a parita' di campioni.
3. Una finestra piu' lunga riduce le finestre disponibili, quindi in parte si
   sta misurando anche la perdita di dati di addestramento.
