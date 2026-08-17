# Convenzioni e trappole

Regole locali che non si deducono leggendo il codice, e trappole che hanno gia' fatto
perdere tempo almeno una volta.

## Lingua

Codice, commenti, documenti e messaggi all'utente in **italiano**, senza accenti nei
sorgenti (si scrive `perche'`, non `perché`): evita sorprese di codifica fra Windows,
Linux e Docker. I **messaggi di commit** sono in **inglese**.

## Nomi

Nomi descrittivi e distesi, anche lunghi. Niente abbreviazioni opache ne' variabili di
una lettera, salvo convenzioni matematiche evidenti. Una funzione che si chiama
`rmse_su_validazione` dice cosa restituisce; una che si chiama `ev` no.

## Git

- **Un branch per compito**, con nome che descrive il compito.
- Mai commit diretti su `main`.
- Staging **file per file**. Mai `git add -A`.
- Messaggio in inglese, imperativo, che spiega **perche'**, non cosa: il diff dice gia'
  cosa.
- Nessun push, merge, rebase o reset distruttivo senza autorizzazione esplicita
  dell'utente, chiesta subito prima dell'operazione.
- Prima di ogni commit: `pytest -q` e `ruff check`.

## Verifica

Nessun numero entra in un documento senza il comando che lo produce. Se una cosa non e'
stata misurata si scrive che non e' stata misurata. Le formule si dicono "verificate"
solo dopo essere state eseguite.

Il progetto ha cambiato direzione tre volte per una misura che smentiva
un'aspettativa. La regola vale soprattutto quando il risultato e' comodo.

## Trappole dell'ambiente

**PowerShell 5.1**: non conosce `&&` ne' `||`, si concatena con `;`. Non ha here-doc.
Per file di piu' di poche righe si usa lo strumento di scrittura, non l'eco da shell.

**Il BOM.** PowerShell scrive UTF-8 **con** BOM. Diversi lettori lo rifiutano in
silenzio: e' gia' successo con le credenziali CDS e con la configurazione di Streamlit,
due volte lo stesso errore. Se un file di configurazione sembra ignorato, la prima cosa
da controllare sono i primi tre byte.

**`data` contro `Data`.** Windows non distingue le maiuscole: la radice dei dati si
chiama `datasets/`, non `data/`, altrimenti collide con la cartella tracciata. E in
`.gitignore` la regola e' ancorata (`/datasets/`), perche' senza ancora nascondeva
anche `src/dwf/data/`.

## Trappole del dominio

**Gli slot non sono equidistanti.** 06, 12, 18 UTC: 6, 6, 12 ore. Tre slot fanno un
giorno. Qualunque calcolo che moltiplichi l'indice per un intervallo fisso e' sbagliato.

**Il riferimento serio e' la persistenza diurna**, non quella ingenua. Ripetere l'ultimo
slot e' un riferimento debole: batterlo non significa nulla. Ripetere lo stesso slot del
giorno prima e' molto piu' difficile da battere.

**GRIB impacchetta ogni campo con interi indipendenti.** Per questo si osserva
`sf > tp` in circa il 12 per cento dei punti: non e' un errore dei dati, e' l'errore di
quantizzazione dei due campi, ed e' entro un passo e mezzo di impacchettamento. I dati
sono veri: se qualcosa non torna, l'errore e' nostro.

**Le accumulate hanno ordine degli assi diverso** dalle istantanee. Il trasferimento
richiede una trasposizione esplicita: e' dichiarata in `ACCUMULATED_DIMS`.

## Modelli su disco

Salvati in `.npz` con `allow_pickle=False`. Non si usa `torch.load` con
`weights_only=False`: un checkpoint e' un file che puo' arrivare da fuori, e con pickle
caricarlo significa eseguirlo. Esiste un test che lo dimostra con un carico realmente
malevolo.

I tipi ammessi sono quelli a **rappresentazione binaria a dimensione fissa**: reali,
interi, booleani, e anche i **complessi**, che servono alle varianti spettrali.

## Cartelle dei modelli

Ogni esecuzione che non sia quella principale deve isolarsi con `paths.models_subdir`.
Senza, tutte scrivono in `models/fold_00` e si sovrascrivono: e' gia' successo, e il
sintomo era un errore incomprensibile sul numero di canali.

## Test

Al vero comportamento, non alla forma. Un test che verifica che uno schema sia
parsabile non serve; uno che verifica che una configurazione incoerente venga rifiutata
si'. Nelle prove di rete la pausa fra tentativi va azzerata da fixture: senza, la suite
passava da 9 a 338 secondi.
