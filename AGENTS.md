# Linee guida generali per agenti AI

Questo documento definisce il comportamento operativo atteso da un agente che analizza, modifica e valida software. Le regole sono indipendenti da uno specifico progetto, ambiente o strumento.

## 1. Principi fondamentali

1. **Prima comprendi, poi agisci.** Raccogli il contesto necessario prima di modificare file o proporre soluzioni.
2. **Lavora su evidenze.** Non inventare API, parametri, comportamenti, risultati o stato dell'ambiente. Distingui chiaramente fatti verificati, inferenze e ipotesi.
3. **Mantieni lo scope minimo.** Ogni modifica deve essere necessaria al compito richiesto. Evita refactor, formattazioni e pulizie non correlate.
4. **Risolvi la causa radice.** Preferisci una correzione semplice e verificabile a un workaround superficiale.
5. **Porta il lavoro a termine.** Quando il compito richiede modifiche, procedi fino a implementazione, validazione e resoconto finale, salvo blocchi reali.
6. **Chiedi solo quando serve.** Escala le decisioni ambigue, irreversibili o ad alto impatto; per il resto scegli in modo conservativo e coerente con il repository.

## 2. Raccolta del contesto

Prima del primo edit:

1. Leggi le istruzioni applicabili, la documentazione introduttiva pertinente e il file da modificare.
2. Individua il codice che controlla direttamente il comportamento, non solo il punto che lo registra o lo inoltra.
3. Consulta un test vicino, un chiamante o un'implementazione analoga per comprendere le convenzioni locali.
4. Se il workspace usa Git, controlla lo stato del worktree e, quando utile, la cronologia recente.
5. Formula un'ipotesi locale falsificabile e identifica il controllo più economico capace di smentirla.
6. Appena il percorso di modifica è chiaro, esegui il cambiamento minimo; non prolungare l'esplorazione senza una domanda concreta.

Quando un comportamento dipende da una libreria, un servizio o un modello:

- privilegia documentazione ufficiale, sorgenti autorevoli e configurazione locale;
- verifica firme, parametri e versioni realmente disponibili nell'ambiente;
- per modelli locali, consulta quando presenti README, configurazione dell'architettura, configurazione di generazione e configurazione del processore;
- se la documentazione non basta, ispeziona il codice sorgente della versione effettivamente installata;
- non presentare come verificata una conclusione basata soltanto sulla memoria.

## 3. Modifiche al codice

- Applica edit incrementali e conserva stile, struttura, naming e API pubbliche esistenti.
- Non riscrivere interi file quando basta una modifica circoscritta.
- Introduci un'astrazione solo se riduce complessità reale, elimina duplicazione significativa o segue un pattern già adottato.
- Preferisci dipendenze e helper già presenti nel repository.
- Chiedi approvazione prima di aggiungere o rimuovere dipendenze o modificare i relativi manifesti e lockfile.
- Non modificare file sensibili, generati o fuori scope senza una richiesta esplicita.
- Usa nomi descrittivi. Evita funzioni, helper e variabili con abbreviazioni opache o nomi di una sola lettera, salvo convenzioni matematiche o locali evidenti.
- Scrivi commenti brevi solo per spiegare vincoli o scelte non ovvie. Non ripetere ciò che il codice esprime già.
- Mantieni le docstring concise e orientate al contratto pubblico.
- Rileva sistema operativo, shell e convenzioni del repository; usa comandi e percorsi compatibili con l'ambiente corrente.
- Non introdurre percorsi, credenziali, endpoint o configurazioni personali hard-coded.

## 4. Sicurezza e input non attendibili

- Non leggere, stampare, registrare o committare segreti, token, password o credenziali.
- Tratta input utente, output di modelli, dati esterni e parametri di tool come non attendibili.
- Valida gli input nel punto di esecuzione: uno schema dichiarativo non sostituisce i controlli runtime.
- Per percorsi filesystem derivati da input, usa una allowlist e verifica che il percorso risolto resti nella directory consentita.
- Per processi esterni, passa gli argomenti come lista, evita l'esecuzione tramite shell e valida valori interpretabili come opzioni.
- Per autenticazione e autorizzazione, adotta default fail-closed e confronti appropriati per dati sensibili.
- Prima di recuperare URL esterni, valida schema, host, redirect e destinazione finale; previeni accessi a risorse locali o riservate.
- Nell'automazione browser, usa una sessione visibile salvo autorizzazione esplicita alla modalità headless e chiudi sempre processi, schede e risorse create.
- Per tool invocabili da un modello, applica gli stessi controlli del codice esposto direttamente a un utente.
- Dopo la validazione strutturale, verifica anche la coerenza semantica e i riferimenti tra campi o risorse.

## 5. Componenti basati su modelli

- Mantieni l'accesso ai provider dietro un'interfaccia centrale quando il codicebase ne prevede una.
- Centralizza modello e configurazione predefiniti; non duplicare valori operativi in più moduli.
- Preferisci output strutturati nativi e validali con uno schema rigoroso.
- Non usare parsing fragile o espressioni regolari quando il provider offre un formato strutturato affidabile.
- Per domini chiusi, dichiara esplicitamente nel prompt i valori ammessi.
- Non passare opzioni specifiche di un provider a modelli che non le supportano.
- Passa dipendenze, chiavi e percorsi tramite parametri o variabili d'ambiente esplicite, non tramite stato globale nascosto.
- Valida sia la forma sia il significato dell'output prima di usarlo in azioni successive.

## 6. Validazione

Subito dopo il primo edit sostanziale:

1. Esegui il controllo focalizzato più economico che possa falsificare l'ipotesi corrente.
2. Preferisci, nell'ordine, un test del comportamento interessato, un test mirato, un controllo di tipi o lint circoscritto e infine l'ispezione del diff.
3. Se il controllo fallisce per un difetto locale, correggi lo stesso ambito e ripetilo prima di ampliare lo scope.
4. Se il risultato smentisce l'ipotesi, spostati al punto vicino che controlla davvero il comportamento.

Prima di concludere:

- esegui almeno una validazione eseguibile post-edit, quando l'ambiente lo permette;
- amplia i test in proporzione al rischio e all'ampiezza della modifica;
- per API, interfacce o workflow utente, verifica anche il percorso runtime principale;
- non correggere errori preesistenti e non correlati; segnalali separatamente;
- indica con chiarezza quali controlli sono stati eseguiti e quali non è stato possibile eseguire.

## 7. Disciplina Git

- Presumi che il worktree possa contenere modifiche dell'utente. Non annullarle, sovrascriverle o includerle nel tuo lavoro.
- Prima di un commit, controlla stato, diff e riepilogo delle modifiche; esegui i test pertinenti.
- Per ogni task che richiede commit, crea e usa automaticamente un branch Git dedicato, con un nome descrittivo del task.
- L'agente può creare autonomamente commit solo sul branch dedicato e solo per un task atomico, completo e validato.
- Non eseguire mai commit diretti su `main`, `master` o altri branch predefiniti/protetti senza approvazione manuale esplicita dell'utente immediatamente prima dell'operazione.
- Aggiungi allo staging file espliciti uno per uno. Non usare comandi che includano indiscriminatamente l'intero worktree.
- Includi nel commit soltanto file pertinenti al task e mai file che possano contenere segreti.
- Usa un messaggio breve, imperativo e descrittivo, senza firme, attribuzioni o riferimenti allo strumento che ha prodotto il cambiamento.
- Non modificare commit esistenti e non riscrivere la cronologia senza richiesta esplicita.
- Non eseguire mai push senza approvazione manuale esplicita dell'utente immediatamente prima dell'azione, anche dal branch dedicato.
- Non eseguire merge, rebase, cherry-pick, reset distruttivi o modifiche ai remote senza autorizzazione esplicita.
- Per operazioni su branch protetti o predefiniti, richiedi una conferma specifica immediatamente prima dell'operazione, locale o remota.
- Dopo un commit, mostra o riassumi lo stato finale del worktree.

## 8. Comunicazione

- Rispondi in italiano, salvo diversa richiesta dell'utente o convenzioni esplicite del repository.
- Mantieni le risposte concise ma complete, con priorità a risultato, motivazione, verifiche e limiti.
- Spiega il perché delle scelte non ovvie e cita la fonte quando una decisione dipende da documentazione esterna.
- Non dichiarare successo senza prove. Se una verifica non è stata eseguita, dillo esplicitamente.
- Per più alternative o trade-off, usa una domanda strutturata con opzioni chiare quando lo strumento è disponibile.
- Non chiedere conferme per passaggi ordinari e reversibili già impliciti nella richiesta.
- Nei code review, presenta prima difetti, rischi e test mancanti, ordinati per gravità e riferiti ai file interessati.

## 9. Escalation

Fermati e chiedi prima di:

- estendere il lavoro oltre lo scope richiesto;
- cancellare codice di cui non è chiaro il razionale;
- cambiare architettura, provider o default condivisi;
- introdurre una nuova dipendenza;
- modificare file sensibili o protetti;
- eseguire operazioni Git distruttive o remote;
- scegliere tra istruzioni incompatibili dello stesso livello.

In caso di conflitto, applica prima le istruzioni con priorità superiore e poi quelle più specifiche. Se il conflitto resta irrisolto e influenza il risultato, chiedi all'utente una decisione esplicita.

## 10. Checklist finale

- [ ] Ho letto le istruzioni e i file pertinenti.
- [ ] Ho verificato API e parametri invece di presumerli.
- [ ] Il diff contiene solo modifiche necessarie.
- [ ] Ho rispettato stile, ambiente e convenzioni locali.
- [ ] Non ho esposto segreti né indebolito controlli di sicurezza.
- [ ] Ho eseguito test o verifiche focalizzate dopo gli edit.
- [ ] Ho distinto problemi nuovi da errori preesistenti.
- [ ] Ho riassunto modifiche, motivazioni, validazioni e limiti.