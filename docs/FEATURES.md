# Screening delle famiglie di canali

Generato da `scripts/screen_features.py`.

## Che cosa misura

Il bersaglio **non** e' il valore futuro ma il **residuo rispetto alla
persistenza diurna**, cioe' quanto il tempo cambia rispetto a ieri alla stessa
ora. E' la quantita' che il modello ancorato deve davvero produrre.

La distinzione non e' formale. La temperatura di domani a mezzogiorno somiglia
moltissimo a quella di oggi a mezzogiorno: qualunque canale che porti traccia
della temperatura attuale mostrerebbe una correlazione altissima col valore
futuro pur non aggiungendo nulla a un riferimento gratuito. Misurare sul
residuo elimina quel merito apparente.

## Protocollo

| | |
|---|---|
| variabile | `t2m` |
| scadenza | indice 2 |
| fold | 0 |
| punti di train | 25200 |
| punti di validazione | 12600 |
| canali totali | 245 |
| deviazione del residuo | 0.188 |

Sonda lineare con Ridge, adattata sul blocco di train e misurata su quello di
validazione. Le celle sono campionate sparse e non a ritagli contigui: celle
adiacenti portano quasi la stessa informazione, e contarle come punti
indipendenti gonfierebbe la fiducia nel risultato.

## Risultati

Tutte le famiglie insieme spiegano **-0.0212** della varianza del residuo.

| famiglia | canali | da sola | togliendola |
|---|---:|---:|---:|
| `time` | 4 | +0.0082 | +0.0142 |
| `static` | 2 | +0.0032 | +0.0026 |
| `wind_speed` | 21 | +0.0207 | +0.0023 |
| `tendency` | 27 | -0.0344 | -0.0001 |
| `latitude` | 2 | +0.0009 | -0.0045 |
| `state` | 189 | -0.0363 | -0.0268 |

**Come si leggono le due colonne.** *Da sola* dice se in quella famiglia il
segnale esiste. *Togliendola* dice se quel segnale e' suo o gia' disponibile
altrove. Una famiglia forte da sola ma con perdita nulla in ablazione e'
ridondante, e rimuoverla non costa nulla.

## Limiti dichiarati

1. **La sonda e' lineare.** Misura il segnale accessibile linearmente, che e' un
   limite inferiore. Una famiglia debole qui puo' comunque servire alla rete,
   che non lo e'; una famiglia forte qui serve di sicuro.
2. **Una sola variabile e una sola scadenza per esecuzione.** Le conclusioni non
   si trasferiscono automaticamente a pioggia e neve, che sono discontinue.
3. **Un solo fold.** La stabilita' fra fold non e' verificata qui.
4. La sonda opera per cella indipendente e quindi **non vede la struttura
   spaziale**, che e' invece cio' che la rete convoluzionale sfrutta. Le famiglie
   utili solo attraverso il contesto locale risultano sottostimate.
