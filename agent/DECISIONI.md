# Registro delle decisioni

Perche' il progetto e' fatto cosi'. In sola aggiunta: una decisione superata si annota
come superata, non si cancella, perche' sapere cosa e' stato provato e non ha funzionato
vale quanto sapere cosa funziona.

Ogni voce riporta la **prova**, non l'opinione.

---

## D1 — La radice dei dati si chiama `datasets/`

Windows non distingue maiuscole e minuscole: `data/` sarebbe collisa con la `Data/`
tracciata nel repository, mescolando gigabyte generati e sorgenti versionati. La regola
in `.gitignore` e' ancorata (`/datasets/`) perche' senza ancora nascondeva anche
`src/dwf/data/`, che e' codice.

## D2 — Zarr per il tensore, Polars per il registro

Valutata la proposta di tenere tutto in Polars. Il tensore e' un array dimensionale
denso da centinaia di milioni di celle: un formato a colonne lo rappresenterebbe come
tabella lunga con le coordinate ripetute per ogni valore, moltiplicando l'occupazione.
Zarr conserva la forma e permette di leggere un ritaglio senza toccare il resto.
Polars resta dove e' bravo: il catalogo degli slot, il manifesto degli scaricamenti,
le metriche.

## D3 — Gli slot non sono equidistanti

06, 12, 18 UTC distano 6, 6 e 12 ore. Scoperto perche' la correlazione con la scadenza
sembrava non monotona, con massimi alle posizioni 3, 6 e 9: non era un fenomeno fisico,
erano i multipli di un giorno. L'etichetta `lead x 6 ore` era sbagliata.

Conseguenza diretta: il riferimento serio non e' ripetere l'ultimo slot, e' ripetere
**lo stesso slot del giorno prima**. Misurato a 24 ore: persistenza ingenua 4,535
gradi, persistenza diurna 2,401.

## D4 — L'ancoraggio diurno e' l'unico effetto fuori dal rumore

Il modello prevede lo scarto rispetto alla persistenza diurna invece del valore
assoluto. Sul banco: senza ancoraggio 6,613 gradi, con ancoraggio circa 3,625. Sono
circa cinquanta volte la dispersione fra semi. Nessun'altra scelta architetturale si
avvicina.

## D5 — Le architetture non sono distinguibili, e non si dichiara un vincitore

Cinque varianti (convolutiva, attenzione, spettrale, ricorrente, ibrida) coprono
l'intervallo 3,642-3,666 gradi, cioe' 0,024 di scarto. Ripetendo la sola variante
convolutiva con tre semi diversi: 3,652, 3,557, 3,665, cioe' una dispersione di 0,059.

**Il rumore e' piu' del doppio del segnale.** Nominare un vincitore significherebbe
riportare quale seme e' stato fortunato. Le varianti restano tutte disponibili nel
registro; la predefinita e' la convolutiva perche' e' la piu' economica fra le
equivalenti (99 secondi per epoca contro 428 della ricorrente).

Coerente con la letteratura letta: in `docs/ricerca.md`, arXiv:2407.14129 e
arXiv:2501.19374 riportano che i backbone saturano e che il guadagno viene dai dati e
dalla formulazione, non dal blocco.

## D6 — Il modello perde sulla temperatura, e va detto

Fold 0, blocco di test, prima dell'ancoraggio: RMSE t2m 4,45 per il modello, 4,73 per
la persistenza ingenua, **3,16 per quella diurna**.

Il modello vince invece nettamente sulle probabilita': Brier pioggia 0,182 contro 0,274,
guadagno di abilita' +0,193; Brier neve 0,061.

Riportare solo il confronto con la persistenza ingenua avrebbe dato un quadro falso.

## D7 — I modelli si salvano in `.npz`, non con pickle

Un checkpoint e' un file che puo' arrivare da fuori. Con pickle, caricarlo significa
eseguirlo. La difesa e' `numpy.savez` letto con `allow_pickle=False`.

Dimostrato con un carico realmente malevolo: sotto pickle viene eseguito, attraverso il
lettore del progetto viene rifiutato.

Il criterio non e' "solo numeri reali" ma **rappresentazione binaria a dimensione
fissa**. La formulazione ristretta escludeva i complessi e impediva di salvare le
varianti spettrali; verificato che NumPy li tratta senza pickle, sono stati ammessi.

## D8 — L'anomalia `sf > tp` e' impacchettamento, non un errore

Nel 12,26 per cento dei punti la neve risulta superiore alla precipitazione totale, che
fisicamente e' impossibile. Causa: GRIB quantizza **ogni campo separatamente** su interi,
quindi i due campi hanno passi di quantizzazione diversi.

Prove: la violazione non supera mai una volta e mezzo il passo di impacchettamento, e la
sua correlazione con l'intensita' della precipitazione e' 0,013, cioe' assente. Se fosse
un errore fisico crescerebbe con l'intensita'.

I dati non vengono corretti. Sono veri: se qualcosa non torna, l'errore e' nostro.

## D9 — La latenza di ERA5 e' stata sondata, non citata

Metadati della collection: fine al 2026-08-11. Sonde di scaricamento reale: 2026-08-12
consegnata, 08-14 e 08-16 rifiutate con HTTP 400. La latenza di cinque giorni e' reale.

Per le date recenti il CDS serve ERA5T, preliminare; ERA5 definitivo arriva a due o tre
mesi.

## D10 — Il buco del 13-14 agosto si colma dal mirror AWS

Il portale ECMWF conserva solo le ultime dodici corse, due o tre giorni. Ma **il mirror
AWS della stessa fonte conserva molto di piu'**: l'archivio risale ad almeno meta' 2023.

Verificato il 2026-08-17: la corsa del 13 agosto e' presente per tutte e quattro le
emissioni, lo step 0 dello stream `oper` pesa 130,9 MB, e il suo indice contiene 187
campi fra cui **tutte e undici** le variabili del progetto.

Resta il punto scientifico: l'analisi IFS non e' la rianalisi ERA5, e sostituirla negli
slot piu' recenti sposta la distribuzione proprio dove il modello e' piu' sensibile. Lo
scarto va misurato nella finestra di sovrapposizione prima di usare le due fonti
insieme.

## D11 — Ogni esecuzione isola i propri checkpoint

`paths.artifacts_subdir` vive sotto la radice dei dati e non toccava `fold_dir`:
impostarlo nei banchi di prova non aveva alcun effetto e tutte le prove scrivevano in
`models/fold_00`.

Il difetto era invisibile finche' le prove erano sequenziali e con lo stesso numero di
canali, perche' ciascuna rileggeva quello che aveva appena scritto. E' emerso quando un
addestramento a scala piena girava in parallelo e ha sovrascritto il checkpoint fra
l'addestramento di una prova e la sua valutazione: errore
`il checkpoint attende 245 canali, la configurazione ne produce 335`.

Corretto introducendo `paths.models_subdir`, che entra in `fold_dir`, con verifica che
il percorso risolto resti sotto la cartella dei modelli, perche' il valore puo' arrivare
da riga di comando.

## D12 — Il sito ascolta solo su 127.0.0.1

All'avvio Streamlit annunciava anche un indirizzo di rete: la dashboard era raggiungibile
da altre macchine ed espone struttura interna e percorsi. Vincolata al loopback e
verificato con `netstat`.

Il primo tentativo di configurazione non aveva effetto perche' PowerShell aveva scritto
il file con il BOM. E' la seconda volta che il BOM costa tempo in questo progetto.
