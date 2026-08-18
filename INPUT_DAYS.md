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

## Il verdetto precedente era contaminato

La prima versione di questo confronto era invalida e la tabella qui sotto la
sostituisce. Il sospetto e' nato da un dettaglio che non poteva essere vero: le
quattro lunghezze dichiaravano **esattamente 961 finestre di train ciascuna**,
mentre una finestra piu' lunga deve per forza produrne meno.

La causa era che `folds.parquet` congela la colonna delle partenze ammesse per
**una sola** lunghezza di finestra, e la selezione si limitava a filtrare quella
lista invece di rienumerare le partenze. Conseguenza misurata: a 3 giorni si
perdevano 12 partenze legittime, mentre a 10 e a 14 giorni restavano dentro
rispettivamente 9 e 21 finestre il cui bersaglio sfoggiava oltre la fine del
blocco, dentro il margine di separazione. Le lunghezze lunghe stavano quindi
prendendo dati che non avrebbero dovuto vedere, e quelle corte ne perdevano.

Dopo la correzione i conteggi sono quelli dovuti, 973 / 961 / 952 / 940, con zero
sconfinamenti. La tabella seguente e' rigenerata con il codice corretto e sullo
store completo (2862 istanti contro i 1458 di prima).

## Protocollo

Fold 0, 3 passate, ritaglio 64, 192 campioni per passata, semi [1234, 101].

## Risultati

| giorni | slot | canali | tendenze | finestre | riferimento | RMSE | guadagno | s/ep |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 3 | 9 | 116 | [1, 3] | 973 | 3.633 | 3.595 | **+0.038** | 97 |
| 7 | 21 | 245 | [1, 3, 9] | 961 | 3.287 | 3.340 | **-0.052** | 108 |
| 10 | 30 | 335 | [1, 3, 9] | 952 | 3.147 | 3.174 | **-0.027** | 114 |
| 14 | 42 | 455 | [1, 3, 9] | 940 | 3.001 | 3.011 | **-0.010** | 123 |

## Lettura

Guadagno maggiore: **3 giorni** (+0.038 degC sul proprio riferimento).

Il margine sul secondo classificato e' 0.048 degC, **inferiore** ai 0.094 degC di incertezza a due deviazioni standard (dispersione fra semi 0.047 degC, misurata qui): la differenza non e'
distinguibile dal caso a questo budget.

Impostazioni non separabili dalla migliore: 3 giorni, 7 giorni, 10 giorni, 14 giorni. Fra queste si sceglie la piu' economica, cioe' **3 giorni** (116 canali), perche' a parita' di risultato misurabile costa meno memoria e lascia piu'
finestre utilizzabili. Nessuna impostazione risulta esclusa: il banco non separa nessuna delle lunghezze provate, quindi qui non si sta scegliendo la migliore, si sta scegliendo la meno cara fra pari.

## Limiti

1. Protocollo ridotto e un solo fold, come per il confronto fra varianti.
2. Il numero di canali cresce linearmente con gli slot di ingresso: una
   finestra lunga non e' solo piu' informativa, e' anche piu' difficile da
   addestrare a parita' di campioni.
3. Una finestra piu' lunga riduce le finestre disponibili, quindi in parte si
   sta misurando anche la perdita di dati di addestramento.
