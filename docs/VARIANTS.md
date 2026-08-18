# Confronto fra varianti

Documento generato dal lavoro di `scripts/compare_variants.py` e completato a mano con
la misura del rumore, che il banco da solo non produce.

## Il protocollo

Il confronto ha valore solo se **cambia una cosa sola**. Qui la cosa sola e' il blocco
elementare della rete: scheletro encoder-decoder, canali di ingresso, layout di uscita,
dati, perdita, ottimizzatore, seme e numero di passate restano identici.

| | |
|---|---|
| fold | 0 |
| passate | 3 |
| ritaglio | 64 x 64 |
| campioni per passata | 192 |
| finestre di validazione | 40 |
| riferimento | persistenza diurna, **3,669 degC** |

Il protocollo e' **volutamente ridotto**: serve a ordinare le alternative in un tempo
utile, non a produrre il modello da consegnare.

## I risultati

| prova | parametri | perdita val | RMSE degC | s/passata |
|---|---:|---:|---:|---:|
| `base_senza_ancoraggio` | 9.979.053 | 9,0619 | **6,613** | 89 |
| `conv` | 9.979.053 | 7,4855 | 3,652 | 99 |
| `conv_spettrale` | 9.979.053 | 7,4898 | 3,649 | 185 |
| `attention` | 13.368.357 | 7,5782 | **3,642** | 151 |
| `fourier` | 8.989.341 | 7,5805 | 3,666 | 114 |
| `recurrent` | 28.933.485 | 7,5673 | 3,661 | 428 |
| `hybrid` | 16.417.451 | 7,5595 | 3,644 | 159 |

## Quanto di questo e' rumore

Una classifica senza una misura di dispersione non e' una classifica. La stessa prova
`conv`, identica in tutto tranne il seme, e' stata ripetuta tre volte:

| seme | RMSE degC |
|---|---:|
| predefinito | 3,652 |
| 101 | **3,557** |
| 202 | 3,665 |

Media 3,625, **scarto tipo 0,059 degC**, escursione 0,108 degC.

Il confronto che conta e' questo:

- distanza fra la migliore e la peggiore delle cinque architetture competitive:
  3,666 - 3,642 = **0,024 degC**;
- rumore fra ripetizioni della stessa architettura: **0,059 degC**.

**L'intera differenza fra le architetture sta dentro meno di mezzo scarto tipo.** A
questo budget il banco **non riesce a distinguerle**. Dichiarare vincitrice `attention`
perche' segna 3,642 significherebbe leggere il seme, non il modello.

Questo riproduce esattamente cio' che la letteratura controllata riporta (arXiv:2407.14129):
a parita' di dati e di protocollo i backbone **saturano**, e la scelta
dell'architettura conta molto meno di quanto suggerisca l'entusiasmo per l'ultima
pubblicazione.

## L'unico effetto che esce dal rumore

`base_senza_ancoraggio` segna **6,613** contro una media di 3,625 con scarto tipo
0,059: circa **cinquanta** scarti tipo di distanza. Non e' un dettaglio di
inizializzazione, e' l'unica scelta di progetto che il banco misura in modo
inequivocabile.

L'ancoraggio diurno consiste nel far produrre alla rete uno **scarto** rispetto
all'osservazione piu' recente alla stessa ora del bersaglio, invece del valore assoluto.
Senza di esso la rete spende gran parte della capacita' a ricostruire un riferimento che
si ottiene gratis, e infatti a parita' di budget resta lontanissima.

Lezione generale: **il modo in cui si pone il problema ha pesato piu' di ogni scelta di
architettura**.

## La scelta

Default: **`conv`**.

Non perche' abbia vinto, ma perche' a parita' statistica di accuratezza e' la piu'
economica fra le competitive: 99 s/passata contro 151 di `attention`, 159 di `hybrid`,
185 di `conv_spettrale` e 428 di `recurrent`. Su CPU, dove il tempo di calcolo e' il
vincolo reale, un costo doppio a parita' di risultato e' un costo doppio.

Le altre **restano nel repository**, documentate e selezionabili con `model.variant`:

```yaml
model:
  variant: attention   # conv | attention | fourier | recurrent | hybrid
```

Non vengono cancellate per due motivi. Primo, il risultato e' misurato a budget ridotto
e su un fold: con dati e passate sufficienti l'ordine puo' cambiare, e la letteratura
suggerisce che l'attenzione guadagni proprio quando i dati crescono. Secondo, la misura
che le ha scartate deve restare riproducibile: un'architettura rimossa dal codice non e'
piu' verificabile da nessuno.

## Note oneste sui limiti di questa misura

1. **Un solo fold.** La dispersione fra fold non e' stata misurata e puo' essere
   maggiore di quella fra semi.
2. **Tre semi solo per `conv`.** Si assume che le altre varianti abbiano rumore simile.
   E' plausibile ma non verificato.
3. **I tempi sono contaminati dal carico.** Le ripetizioni con seme diverso segnano
   71 s/passata contro 99 della prova originale, a codice identico, perche' nel
   frattempo era in corso uno scaricamento. I tempi vanno confrontati fra loro solo
   entro la stessa corsa; i rapporti fra varianti restano validi perche' misurati di
   seguito.
4. **Il termine spettrale non si distingue** (3,649 contro 3,652) ma raddoppia il costo.
   A tre passate l'ampiezza non ha ancora avuto modo di collassare, che e' proprio il
   difetto che quel termine dovrebbe correggere: va rivalutato a scala piena, non
   scartato qui.
5. **`recurrent` non e' il ConvLSTM temporale** della letteratura, che consumerebbe una
   sequenza e cambierebbe il layout di ingresso, cioe' la variabile che il confronto
   tiene ferma. Qui misura la profondita' effettiva a parametri condivisi.
