# Latenza dei dati e fonti a tempo quasi reale

## 1. La latenza di ERA5, misurata

L'affermazione "ERA5 ha 5-6 giorni di latenza" non e' stata ripresa dalla
documentazione: e' stata **misurata contro il servizio**, il 2026-08-17.

**Metadati della collection.** L'estensione temporale dichiarata da
`reanalysis-era5-single-levels` finisce al **2026-08-11**.

**Sonda diretta.** I metadati sono una dichiarazione, non una prova, quindi si e'
tentato un vero scaricamento di una singola cella, una variabile, un'ora:

| data richiesta | esito |
|---|---|
| 2026-08-16 | **rifiutata** — HTTP 400, `invalid request` |
| 2026-08-14 | **rifiutata** — HTTP 400, `invalid request` |
| 2026-08-12 | **consegnata** — 116 byte |

**Conclusione.** L'ultimo giorno realmente ottenibile e' il 2026-08-12, cioe' **5 giorni**
prima della data corrente. Il dato del 16 agosto **non** e' disponibile. La latenza e'
reale e non un'assunzione ereditata.

Nota tecnica: per le date recenti il CDS serve **ERA5T**, la versione preliminare, che
puo' essere rivista nei mesi successivi. La latenza di 5 giorni e' gia' quella del
prodotto preliminare; ERA5 definitivo arriva con **2-3 mesi** di ritardo.

## 2. La fonte che copre il buco

**ECMWF Open Data** (`data.ecmwf.int`), sottoinsieme gratuito delle previsioni
operative IFS e AIFS.

| | |
|---|---|
| Risoluzione | **0,25 gradi**, identica alla nostra griglia |
| Formato | GRIB2 |
| Corse | 4 al giorno: 00, 06, 12, 18 UTC |
| Licenza | CC-BY-4.0, uso commerciale ammesso con attribuzione |
| Accesso | nessuna chiave: client `ecmwf-opendata`, oppure AWS, Azure, GCP |
| Passo 0 | l'**analisi**, cioe' lo stato stimato al momento della corsa |

### Copertura delle nostre variabili

Verificata voce per voce contro la tabella dei parametri IFS `oper`/`fc`:

| nostra | ECMWF Open Data | id |
|---|---|---|
| `t2m` | `2t` 2 metre temperature | 167 |
| `d2m` | `2d` 2 metre dewpoint temperature | 168 |
| `msl` | `msl` Mean sea level pressure | 151 |
| `u10` | `10u` 10 metre U wind component | 165 |
| `v10` | `10v` 10 metre V wind component | 166 |
| `tcc` | `tcc` Total cloud cover | 164 |
| `sd` | `sd` Snow depth water equivalent | 141 |
| `tp` | `tp` Total precipitation | 228 |
| `sf` | `sf` Snowfall water equivalent | 144 |
| `lsm` | `lsm` Land Sea Mask | 172 |
| `z` | `z` Geopotential (step 0) | 129 |

**Tutte e undici sono disponibili.** Non serve rinunciare a nessun canale ne'
riaddestrare con un insieme ridotto.

## 3. Il vincolo che decide come si usa

> "Data are retained for the most recent 12 forecast runs, corresponding to
> approximately 2-3 days of forecasts."

L'archivio e' **a scorrimento**. Non e' un archivio storico: consente di prendere gli
ultimi due o tre giorni, non di recuperare il passato.

Ne segue una conseguenza operativa netta:

- **Oggi** i giorni 13 e 14 agosto non sono recuperabili da nessuna delle due fonti:
  ERA5 arriva all'11-12, l'archivio a scorrimento parte dal 14-15.
- **Da oggi in avanti**, scaricando ogni giorno e archiviando in locale, la serie resta
  continua e la latenza scende praticamente a zero.

Il buco esiste **una volta sola**, all'avvio. Va aperto adesso, non quando servira'.

## 4. Il problema scientifico da non ignorare

ERA5 e IFS **non sono lo stesso prodotto**.

ERA5 e' una rianalisi: assimila osservazioni *anche successive* all'istante descritto e
usa una versione congelata del modello. L'analisi IFS e' prodotta in tempo reale, con
il ciclo operativo corrente, e vede solo le osservazioni gia' arrivate.

Incollare l'una in coda all'altra significa cambiare la distribuzione dei dati
**esattamente negli slot piu' recenti**, che sono i piu' influenti sulla previsione, e
che l'ancoraggio diurno usa come riferimento. Un modello addestrato solo su ERA5
incontrerebbe in produzione un ingresso leggermente diverso da qualunque cosa abbia
visto.

Non e' un motivo per rinunciare, e' un motivo per **misurare**. Le due fonti si
sovrappongono in una finestra di due o tre giorni, e questo permette una verifica
diretta:

1. scaricare gli stessi istanti da entrambe le fonti nella finestra di sovrapposizione;
2. misurare, per variabile, lo scarto medio e la sua struttura spaziale;
3. se lo scarto e' piccolo rispetto all'errore del modello, usarle insieme;
4. se non lo e', correggerlo esplicitamente oppure limitare l'uso della fonte
   operativa ai canali meno sensibili.

Questa verifica va fatta **prima** di dichiarare operativa la pipeline, non dopo.

## 5. L'occasione in piu'

ECMWF Open Data non fornisce solo l'analisi: fornisce le **previsioni** IFS e AIFS a
0,25 gradi, fino a 144 ore per le corse 06 e 18 UTC, con passi da 3 ore.

Sono esattamente lo stesso compito che svolge questo progetto, prodotte dal miglior
centro previsionale al mondo. Diventano quindi **un riferimento operativo vero**,
molto piu' severo della persistenza diurna, e disponibile gratuitamente.

Il confronto sarebbe impietoso e per questo utile: dice quanto dista un modello
addestrato su CPU in locale dallo stato dell'arte, invece di limitarsi a dire che batte
la ripetizione di ieri.

## 6. Che cosa fare, in ordine

1. Aprire subito la raccolta giornaliera da ECMWF Open Data, cosi' il buco resta
   limitato ai giorni gia' persi.
2. Misurare lo scarto ERA5 contro analisi IFS nella finestra di sovrapposizione.
3. Solo dopo, permettere alla pipeline di previsione di attingere alla fonte operativa
   per gli slot piu' recenti.
4. Aggiungere la previsione IFS come terzo termine di confronto nella valutazione.

## Fonti

- Estensione temporale e sonde di scaricamento: misurate contro il CDS il 2026-08-17.
- Catalogo dei parametri, licenza, risoluzione e politica di conservazione:
  <https://www.ecmwf.int/en/forecasts/datasets/open-data>, consultata il 2026-08-17.
- Client di accesso: <https://github.com/ecmwf/ecmwf-opendata>.
