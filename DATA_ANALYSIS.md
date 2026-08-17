# Analisi del dataset ERA5

Generata su **1,458 slot** ingeriti, da 2024-01-01 a 2026-03-31, sottocampionamento spaziale 1 su 4.

> Principio di lettura: **i dati sono veri**. ERA5 assimila osservazioni in un
> modello fisico, quindi davanti a un numero sorprendente la prima ipotesi da
> verificare e' un errore di analisi, non del dato.

Rigenerabile con `python scripts/analyze_data.py`.

## 1. Copertura temporale

```
slot allocati nello store : 2,862
slot effettivamente presenti: 1,458
mesi coperti              : 16
primo istante             : 2024-01-01T06:00
ultimo istante            : 2026-03-31T18:00

slot per ora del giorno   : 06Z=486, 12Z=486, 18Z=486

slot per mese:
  2024-01   93
  2025-01   93
  2025-02   84
  2025-03   93
  2025-04   90
  2025-05   93
  2025-06   90
  2025-07   93
  2025-08   93
  2025-09   90
  2025-10   93
  2025-11   90
  2025-12   93
  2026-01   93
  2026-02   84
  2026-03   93
```

Le tre ore giornaliere sono equilibrate, come deve essere: uno squilibrio farebbe apprendere alla rete una climatologia distorta.

## 2. Distribuzioni e plausibilita' fisica

```
var   unita      minimo       p01   mediana       p99    massimo    NaN
t2m   degC      -58.603   -24.596    14.584    40.150     51.436  0.00%
d2m   degC      -62.439   -28.740     5.603    23.997     32.545  0.00%
msl   Pa      93835.312 98025.812 101401.000 103494.688 105512.625  0.00%
u10   m/s       -28.217   -11.082    -0.055    13.576     29.196  0.00%
v10   m/s       -29.160   -11.835    -0.573    12.235     26.930  0.00%
tcc   0-1         0.000     0.000     0.614     1.000      1.000  0.00%
sd    m           0.000     0.000     0.000    10.000     10.000  0.00%
tp    m           0.000     0.000     0.000     0.009      0.239  0.00%
sf    m           0.000     0.000     0.000     0.003      0.045  0.00%
```

Tutte le variabili stanno negli intervalli fisicamente plausibili.

## 3. Ciclo diurno

```
  ora     media   dev.std  n slot
  6Z    11.893    12.359     486
 12Z    15.380    14.069     486
 18Z    13.732    13.357     486

ampiezza media del ciclo diurno sul dominio: 3.487 degC
```

Sul dominio intero il ciclo diurno vale 3.49 gradi, attenuato dalla media su mare e terra. E' la grandezza che il modello attuale sottostima, ed e' il motivo per cui i canali solari sono stati aggiunti.

## 4. Prevedibilita' e riferimenti da battere

```
lead  ore  corr.ing  RMSE ing  corr.diu  RMSE diu  guadagno  coppie
   1    8    0.9423     4.535    0.9838     2.401    47.1%    1452
   2   16    0.9370     4.738    0.9838     2.401    49.3%    1452
   3   24    0.9838     2.401    0.9838     2.401     0.0%    1452
   4   32    0.9279     5.068    0.9718     3.173    37.4%    1446
   5   40    0.9241     5.201    0.9718     3.173    39.0%    1446
   6   48    0.9718     3.173    0.9718     3.173     0.0%    1446
   7   56    0.9184     5.393    0.9645     3.557    34.0%    1440
   8   64    0.9160     5.471    0.9645     3.557    35.0%    1440
   9   72    0.9645     3.557    0.9645     3.557     0.0%    1440
```

Gli slot non sono equidistanti, quindi la colonna delle ore e' calcolata sui tempi veri e non come multiplo di sei. Le scadenze 3, 6 e 9 cadono esattamente a uno, due e tre giorni, cioe' alla stessa ora del giorno iniziale.

La persistenza diurna e' nettamente migliore di quella ingenua, e il guadagno e' massimo proprio alle scadenze che sfasano il ciclo giorno-notte. Questo sposta l'asticella: **il riferimento onesto da battere e' la persistenza diurna**, non quella ingenua, e la valutazione del modello va aggiornata di conseguenza.

La correlazione della persistenza diurna, da 0.965 a 0.984, e' anche la stima teorica dello smorzamento: sotto errore quadratico l'ampiezza ottimale della previsione vale la correlazione, quindi un modello addestrato a MSE tendera' a produrre quella frazione della variabilita' reale. E' la giustificazione quantitativa del termine spettrale nella nuova loss.

## 5. Diagnostica dei target

```
celle con pioggia oltre 0.1 mm : 36.71%
celle con neve prevalente          : 8.50%
quota nevosa fra le celle piovose  : 23.13%
precipitazione media dove piove    : 1.585 mm
massimo su una cella               : 238.60 mm
```

La pioggia interessa il 36.7% delle celle: la classe positiva e' minoritaria ma non rara, quindi l'accuratezza grezza sarebbe una metrica ingannevole e F1 resta la scelta giusta. La neve e' molto piu' rara, ed e' il motivo per cui la sua soglia di decisione va scelta sui dati e non fissata a 0,5.

## 6. Vigo di Cadore

```
cella della griglia        : riga 114, colonna 210 (46.50 N, 12.50 E)
quota del modello          : 1463 m
quota reale del paese      : circa 951 m
scarto di quota            : 512 m

temperatura media          : 4.17 degC
minima osservata           : -21.68 degC
massima osservata          : 26.62 degC
ampiezza del ciclo diurno  : 6.67 degC
umidita' relativa media    : 76.3%
calore latente medio       : 12.8 kJ/kg
  media a 06Z              : 0.92 degC
  media a 12Z              : 7.59 degC
  media a 18Z              : 4.00 degC
```

La cella sta 512 m piu' in alto del paese, perche' a 0,25 gradi una cella copre circa 28 km e media tutto il Cadore, creste comprese. Con il gradiente termico standard di 6,5 gradi per chilometro corrisponde a circa 3.3 gradi di scarto freddo sistematico rispetto al fondovalle. Non e' un errore del modello ne' del dato: e' il limite di risoluzione, e per passare dalla cella al paese serve una correzione di quota esplicita. L'ampiezza diurna locale, 6.7 gradi, e' molto maggiore di quella media del dominio, il che rende Vigo un punto severo per il modello.

## 7. Coerenza del canale solare

```
istanti campionati            : 122
coseno zenitale medio         : 0.1906
correlazione con la temperatura: +0.8806
```

La correlazione fra insolazione istantanea e temperatura media del dominio vale +0.881. Il segno positivo conferma che il canale porta informazione nella direzione fisicamente attesa. Il valore non e' vicino a uno perche' la temperatura ha una forte inerzia termica e la stagione domina sull'ora: e' proprio questa differenza che la rete deve imparare a comporre.
