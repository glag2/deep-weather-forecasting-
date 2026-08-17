# Cartella dell'agente

Documenti destinati a chi **sviluppa** il progetto, umano o agente che sia. Non sono
materiale di lettura per chi vuole soltanto usare il previsore: quello sta nella
radice (`README.md`) e in `docs/`.

La separazione e' voluta. Un documento che deve spiegare a un utente come ottenere una
previsione e un documento che deve permettere a un agente di riprendere il lavoro a
freddo hanno lettori, tempi di vita e criteri di verita' diversi: tenerli insieme
significa che nessuno dei due resta aggiornato.

## Che cosa leggere, e in che ordine

Se stai riprendendo il progetto senza aver visto nulla di quanto e' successo prima,
leggi in questa sequenza:

| ordine | file | risponde a |
|---|---|---|
| 1 | [`STATO.md`](STATO.md) | dove siamo adesso, che cosa gira, che cosa e' rotto |
| 2 | [`PIANO.md`](PIANO.md) | che cosa va fatto, in che ordine, con quale criterio di completamento |
| 3 | [`CONVENZIONI.md`](CONVENZIONI.md) | come si lavora qui, e quali trappole hanno gia' fatto perdere tempo |
| 4 | [`DECISIONI.md`](DECISIONI.md) | perche' le cose sono come sono, con le misure che lo dimostrano |

`STATO.md` va riscritto ogni volta che qualcosa cambia davvero. `PIANO.md` va aggiornato
segnando i passi conclusi e aggiungendo quelli emersi. `DECISIONI.md` e' in sola
aggiunta: una decisione superata si annota come superata, non si cancella.

## Regola che vale piu' delle altre

Le affermazioni in questi documenti devono essere **verificabili**. Se un numero e'
scritto qui, da qualche parte esiste il comando che lo produce, ed e' indicato. Se una
cosa non e' stata misurata, va scritto che non e' stata misurata.

Il progetto ha gia' cambiato direzione tre volte perche' una misura ha smentito
un'aspettativa ragionevole: la persistenza diurna che batte il modello, le architetture
indistinguibili dal rumore fra semi, la latenza di ERA5 confermata solo dopo averla
sondata. Un documento che riporta impressioni invece di misure avrebbe nascosto tutte e
tre.
