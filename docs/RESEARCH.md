# Ricerca tecnologica: quali reti neurali usare, e quali no

Documento decisionale. Ogni scelta e' motivata da evidenza pubblicata e citata, non da
intuizione. Le tecnologie scartate sono elencate con il motivo, perche' sapere cosa non
usare vale quanto sapere cosa usare.

Il criterio imposto e' esplicito: **solo tecnologie provate**, niente di dubbio, niente
che comporti grandi compromessi.

---

## 1. Il nostro problema, in termini confrontabili con la letteratura

Prima di leggere i risultati altrui serve sapere quale colonna della tabella ci
riguarda, perche' le conclusioni della letteratura cambiano radicalmente con il regime.

| Dimensione | Noi | GraphCast / Pangu / FourCastNet |
|---|---|---|
| Dominio | regionale, 261x401, 65 gradi di latitudine | globale, sfera intera |
| Scadenza | 6-72 ore | 10-15 giorni |
| Generazione | **tutti i 9 slot in un colpo** | autoregressiva, 40-60 passi |
| Dati di addestramento | ~2 anni | 40 anni |
| Hardware | **1 CPU** | centinaia di TPU/GPU per settimane |
| Variabili | 9 di superficie | 67-221 su piu' livelli |

Tre conseguenze immediate, che useremo per filtrare la letteratura:

1. **Non siamo autoregressivi.** Tutta la letteratura sulla stabilita' dei rollout
   lunghi, sulla deriva climatica e sulle rappresentazioni sferiche esiste per
   risolvere un problema che noi **non abbiamo**. Adottare quelle soluzioni
   significherebbe pagarne il costo senza incassarne il beneficio.
2. **Non siamo globali.** La curvatura della sfera e la singolarita' ai poli motivano
   GraphCast e SFNO. Il nostro riquadro va da 10 a 75 gradi nord: la convergenza dei
   meridiani esiste ed e' gia' gestita con i canali di latitudine.
3. **Il vincolo dominante e' il dato, non il modello.** Con circa 2 anni di dati contro
   40, la capacita' del modello non e' la risorsa scarsa. Questo e' il punto su cui la
   ricerca ha prodotto il risultato piu' scomodo, sezione 3.

---

## 2. Il confronto controllato fra backbone

**Fonte**: Karlbauer, Maddix et al. (AWS AI Labs, Caltech, Amazon), *Comparing and
Contrasting Deep Learning Weather Prediction Backbones on Navier-Stokes and Atmospheric
Dynamics*, arXiv:2407.14129.

E' l'unico lavoro che confronta i backbone **a parita' di conteggio parametri,
protocollo di addestramento e variabili**. Ogni altro paper confronta il proprio modello
con quelli altrui addestrati diversamente, il che rende impossibile attribuire il
merito all'architettura invece che al protocollo. Per una decisione architetturale e'
la fonte giusta.

### Risultato principale per il nostro regime

Sulle scadenze **brevi e medie**, che sono le nostre, il paper e' netto:

> "Over short-to-mid-ranged lead times we observe a surprising forecast accuracy of
> ConvLSTM (the only recurrent and oldest architecture in our comparison), followed by
> SwinTransformer and FourCastNet."

E, verificato **specificamente sulla temperatura a 2 metri**, che e' la nostra
variabile principale:

> "Both the results on T2m and on the ACC metric support our findings, showing the
> superiority of ConvLSTM, FourCastNet, and SwinTransformer on short-to-mid-ranged
> forecasts."

Le architetture piu' celebri (GraphCast, SFNO, Pangu) vincono su un asse diverso:

> "In terms of stability, explicit model designs tailored to weather forecasting are
> beneficial, e.g., Pangu-Weather, GraphCast, and Spherical FNO."

La stabilita' riguarda i rollout di 365 giorni e 50 anni. **Non ci riguarda.**

### Due motivi di scarto diversi, da non confondere

Una prima stesura di questo documento scartava alcune tecnologie perche' costose su
CPU. E' un criterio sbagliato: il costo si misura, e se una tecnologia e' davvero
migliore si paga. Le tecnologie qui sotto sono divise secondo il **solo** criterio
legittimo, cioe' se risolvono o no il nostro problema.

### Implementate e messe a confronto

Tutte e tre entrano nel benchmark come varianti del blocco di elaborazione, misurate
sullo stesso protocollo. Il costo viene rilevato, non presunto.

| Tecnologia | Perche' vale la pena | Evidenza |
|---|---|---|
| **Attenzione a finestre (Swin)** | Fra i primi tre nel confronto controllato sulle scadenze brevi e **specificamente su T2m**. Colma il difetto della convoluzione pura: il campo recettivo locale non collega punti lontani, mentre l'avvezione a 72 ore sposta una massa d'aria di centinaia di chilometri. | arXiv:2407.14129, sez. 3.2.1 e B.4 |
| **Operatore neurale di Fourier (stile AFNO/FourCastNet)** | **Il migliore in assoluto** sui dati sintetici (TFNO2D, RMSE 0,0041) e nel gruppo di testa su dati reali a breve termine. Costa O(N log N) grazie alla FFT, quindi non e' nemmeno pesante: e' fra le opzioni piu' economiche. Miscela informazione su **tutto** il dominio in un colpo solo, cosa che la convoluzione non fa. | arXiv:2407.14129, Tab. 1 e Fig. 2 |
| **Ricorrenza convoluzionale (ConvLSTM)** | E' **il piu' accurato a breve termine** nel confronto controllato. Il costo si abbatte applicando la ricorrenza al bottleneck, dove la griglia e' gia' ridotta di 8 volte per lato, cioe' 64 volte in area. Il paper segnala instabilita' oltre 4M parametri: e' un limite reale da verificare, non da presumere. | arXiv:2407.14129, sez. 3.2.1, Tab. 1, Fig. 8 |

La FFT serve comunque per il termine spettrale della loss (sezione 4), quindi
l'infrastruttura e' condivisa fra loss e operatore di Fourier.

### Scartate perche' non risolvono il nostro problema

Queste non sono scartate per il costo. Sarebbero da scartare anche con hardware
illimitato, perche' rispondono a domande che non abbiamo posto.

| Tecnologia | Motivo |
|---|---|
| **Rappresentazioni sferiche** (SFNO, mesh icosaedriche) | Esistono per la curvatura della sfera e la singolarita' ai poli. Il nostro riquadro va da 10 a 75 gradi nord e la convergenza dei meridiani e' gia' gestita con i canali di latitudine. Inoltre, nello stesso confronto, SFNO risulta **piu' debole** di FourCastNet a breve termine: "Surprised by the competitive results of FourCastNet and comparably poor performance of SFNO" (sez. B.3). |
| **Graph Neural Network** | Il vantaggio di un grafo e' rappresentare una **mesh irregolare**. I nostri dati sono una griglia regolare: su una griglia regolare un GNN a stencil fisso e' letteralmente una convoluzione, solo implementata piu' lentamente e con message passing esplicito. A questo si aggiunge che nel confronto controllato e' l'unica architettura che non converge (RMSE 0,52 contro 0,005 del migliore) e va in out-of-memory oltre 500k parametri. |
| **Modelli di diffusione** (GenCast) | Risolvono la **generazione di ensemble**, cioe' campionare futuri alternativi. Noi produciamo gia' incertezza in forma esplicita e calibrata dalle teste probabilistiche, ed e' la forma che serve al progetto ("affidabilita'"). Sarebbe sostituire una risposta diretta con una campionata. |
| **Foundation model preaddestrati** (Aurora, ClimaX) | Sono l'unica famiglia che attaccherebbe davvero il nostro vincolo dominante, la scarsita' di dati. Restano fuori per un motivo strutturale e non di velocita': i checkpoint sono globali, a variabili fisse e su livelli di pressione che **non abbiamo scaricato**, quindi non sono applicabili al nostro tensore di input senza rifare la raccolta dati. Vale la pena riconsiderarli se il progetto passasse ai livelli di pressione. |

Un dettaglio operativo dal paper, che seguiremo: conviene privilegiare **piu' strati per
blocco** invece di piu' blocchi con pochi strati.

---

## 3. Il risultato scomodo: ingrandire il modello non basta

Hai chiesto un modello **piu' grande e piu' preciso**, spendendo piu' risorse. La
ricerca dice che la prima parte non implica la seconda, e lo dice in modo esplicito:

> "we observe that all of these model backbones **'saturate'**, i.e., none of them
> exhibit so-called neural scaling, which highlights an important direction for future
> work"

Nella tabella del paper la saturazione e' visibile a occhio: U-Net satura a 1M
parametri, FourCastNet a 1M, TFNO2D a 500k. Oltre quella soglia i parametri aggiunti
non comprano accuratezza. In alcuni casi la peggiorano: ConvLSTM passa da RMSE 0,009 a
1M parametri a **0,44 a 4M**, cioe' quasi il livello della persistenza.

Il nostro modello attuale ha gia' **9,98 milioni di parametri** e viene addestrato su
poche centinaia di finestre. Siamo gia' oltre il punto in cui la letteratura osserva
saturazione, con un rapporto dati/parametri molto peggiore di quello dei paper.

**Conseguenza operativa.** Non ingrandiro' il modello alla cieca. Misurero' una curva:
piu' configurazioni di capacita' crescente, addestrate con lo stesso protocollo, con
errore di validazione e costo per epoca. Se la curva sale, si cresce; se satura o
peggiora, ingrandire sarebbe spendere ore di CPU per un risultato peggiore, e te lo
diro' con i numeri in mano. Le risorse in piu' che hai autorizzato verranno spese dove
la curva dice che rendono, e i candidati piu' probabili sono l'addestramento piu' lungo,
i sei fold invece di uno, e le feature nuove.

---

## 4. La scoperta piu' utile: la loss standard ha un difetto dimostrato

**Fonte**: Subich, Husain, Separovic, Yang (Environment and Climate Change Canada),
*Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified
Spherical Harmonic Loss Function*, arXiv:2501.19374.

Questo risultato spiega **esattamente** il difetto che avevo misurato sul nostro
modello: coglie la fase del ciclo diurno ma ne sottostima l'ampiezza, con un errore di
-4,8 K alle 12 UTC.

### La dimostrazione

Sia `Y` il valore vero e `X` la previsione, con correlazione `ρ` e deviazione standard
`σ_X`. L'errore quadratico atteso vale:

```
E[MSE] = σ_X² + 1 − 2·σ_X·ρ
```

Derivando rispetto a `σ_X` e annullando, il minimo **non** e' in `σ_X = 1`, ma in:

```
σ_X = ρ
```

Il significato e' brutale: se un fenomeno e' prevedibile solo al 70 %, la previsione che
minimizza l'errore quadratico e' quella che ha **il 70 % dell'ampiezza reale**. Il
modello non sta sbagliando: sta facendo esattamente cio' per cui lo stiamo premiando.
Smorzare e' la strategia ottimale sotto MSE.

E' il fenomeno noto come **doppia penalizzazione**: una previsione corretta ma spostata
viene punita due volte, una per non aver messo l'evento dov'era e una per averlo messo
dove non era. Prevedere una media piatta evita entrambe le punizioni.

### La correzione

Gli autori separano l'errore di **ampiezza** da quello di **decorrelazione**,
sfruttando il teorema di Parseval:

```
AMSE = Σ_k ( √PSD_k(x) − √PSD_k(y) )²  +  2·max(PSD_k(x), PSD_k(y))·(1 − Coh_k(x,y))
```

Il primo termine punisce l'ampiezza sbagliata a ogni scala, il secondo la
decorrelazione, e i due non si compensano piu' a vicenda. Proprieta' rilevanti:

- **e' privo di parametri**, non introduce iperparametri da tarare;
- vale zero **se e solo se** x = y, come l'MSE;
- ha **lo stesso sviluppo di Taylor** dell'MSE attorno alla soluzione esatta, quindi non
  cambia il comportamento vicino all'ottimo;
- e' un **rimpiazzo diretto** in addestramento.

Risultato riportato: risoluzione efficace di GraphCast da **1250 km a 160 km**.

### Adattamento al nostro caso, e i suoi limiti

Il paper usa armoniche sferiche perche' lavora sulla sfera. Noi siamo su un riquadro
regolare, quindi useremo la **trasformata di Fourier bidimensionale**: il paper autorizza
esplicitamente la sostituzione, perche' richiede solo "any decomposition (partition of
unity) that obeys Parseval's theorem", e la FFT lo soddisfa. In torch e' `rfft2`,
disponibile su CPU e poco costosa.

Due limiti che il paper documenta e che riporto perche' ci riguardano direttamente:

1. **Sulla temperatura a 2 metri il guadagno di nitidezza e' piccolo**, perche' quel
   campo e' gia' poco smorzato: e' ancorato all'orografia, che il modello conosce. Il
   guadagno li' si vede nell'abilita' previsionale, non nella distribuzione.
2. **Sulla precipitazione l'AMSE peggiora le cose a breve scadenza.** Gli autori ne
   danno la ragione: la precipitazione e' localizzata e non negativa, quindi la sua
   decomposizione spettrale non somiglia alle variabili gaussiane per cui la formula e'
   derivata.

Da qui una decisione precisa: **applicheremo il termine spettrale solo alle variabili
continue e approssimativamente gaussiane** (temperatura e pressione), lasciando le
teste probabilistiche di pioggia e neve sulla loro verosimiglianza attuale, che e' gia'
calibrata e funziona. Applicarlo ovunque sarebbe imitare il paper invece di usarlo.

---

## 5. Il difetto strutturale che nessuna loss risolve

**Fonte**: Bonavita, *On Some Limitations of Current Machine Learning Weather Prediction
Models*, Geophysical Research Letters, 2024.

> "Forecasts from Machine Learning models have energy spectra notably different from
> those of their training reanalysis fields and Numerical Weather Prediction models.
> This results in overly smooth [forecasts]."

Conferma indipendente che lo smorzamento e' universale nei modelli guidati dai dati, non
un nostro difetto di implementazione. Utile come termine di paragone onesto: quando
riporteremo i risultati, un residuo di smorzamento e' atteso anche nei modelli
operativi.

---

## 6. Decisioni finali

| Ambito | Decisione | Fondamento |
|---|---|---|
| Backbone | U-Net convoluzionale come telaio comune | Gruppo di testa a breve termine nel confronto controllato |
| Blocco di elaborazione | **Quattro varianti a confronto misurato**: convoluzione (base), attenzione a finestre, operatore di Fourier, ricorrenza convoluzionale | Sono le tre famiglie in testa sulle scadenze brevi; il vincitore si decide sui nostri dati, non sui loro |
| Capacita' | **Curva misurata**, non aumento a priori | Saturazione osservata su tutti i backbone; siamo gia' a 10M parametri |
| Loss, variabili continue | Termine spettrale di ampiezza via FFT 2-D | Doppia penalizzazione dimostrata analiticamente; spiega il nostro -4,8 K |
| Loss, pioggia e neve | Verosimiglianza attuale invariata | Il paper documenta peggioramento dell'AMSE sulla precipitazione |
| Loss, spazio | Peso di area per latitudine + peso su Vigo di Cadore | Pratica standard del settore; requisito del progetto |
| Rappresentazione sferica | Scartata | Risolve la curvatura della sfera, che non abbiamo; e piu' debole di FourCastNet a breve termine |
| Grafi | Scartati | Su griglia regolare equivalgono a una convoluzione piu' lenta; unica architettura non convergente nel confronto |
| Diffusione | Scartata | Risolve la generazione di ensemble; produciamo gia' incertezza esplicita e calibrata |
| Foundation model | Rimandati | Richiedono livelli di pressione non presenti nel nostro dataset |

## 7. Idee dai modelli linguistici: cosa si trasferisce e cosa no

DeepSeek-V4 (arXiv:2606.19348) e' stato letto perche' esplicitamente richiesto. La sua
tesi centrale non ci riguarda, ma due ingredienti secondari si'. Vale la pena separarli,
perche' e' facile importare un nome invece di un meccanismo.

| Ingrediente | Cosa risolve nel paper | Da noi |
|---|---|---|
| CSA e HCA, attenzione compressa e sparsa | Il costo quadratico dell'attenzione su un milione di token | **Non si applica.** Non abbiamo una sequenza lunga: abbiamo una griglia 261x401 letta in una volta sola. Il collo di bottiglia che quelle tecniche rimuovono qui non esiste |
| mHC, connessioni residue su varieta' vincolata | Un flusso residuo piu' espressivo: si allarga fra i blocchi e si restringe prima di ciascuno | **Si applica.** E' indipendente dall'attenzione e vale per qualunque rete residua, U-Net convoluzionale compresa. Candidato vero per il confronto fra architetture |
| Ottimizzatore Muon | Addestramento piu' efficiente a parita' di passi | **Si applica**, ed e' la prova piu' economica: cambia l'ottimizzatore, non la rete |
| Previsione multi-token | Emettere piu' passi futuri in una volta | **Gia' fatto**, per costruzione: usciamo con tutte e nove le scadenze insieme |
| Post-addestramento in due fasi con rinforzo | Allineamento e ragionamento | Non si applica: non c'e' preferenza umana da allineare, c'e' un'osservazione da colpire |

La conclusione onesta e' che il titolo del paper, l'attenzione, e' la parte meno
trasferibile, e le parti trasferibili non hanno nulla a che vedere con la meteorologia:
sono un modo di collegare i blocchi e un modo di aggiornare i pesi. Vanno quindi
misurate sul nostro banco come qualunque altra variante, senza credito d'ingresso.

## 8. Riferimenti

1. Karlbauer, Maddix, Ansari, Han, Gupta, Wang, Stuart, Mahoney (2024). *Comparing and
   Contrasting Deep Learning Weather Prediction Backbones on Navier-Stokes and
   Atmospheric Dynamics*. arXiv:2407.14129.
2. Subich, Husain, Separovic, Yang (2025). *Fixing the Double Penalty in Data-Driven
   Weather Forecasting Through a Modified Spherical Harmonic Loss Function*.
   arXiv:2501.19374.
3. Bonavita (2024). *On Some Limitations of Current Machine Learning Weather Prediction
   Models*. Geophysical Research Letters, 10.1029/2023GL107377.
4. Adamov, Oskarsson, Denby et al. (2025). *Building Machine Learning Limited Area
   Models: Kilometer-Scale Weather Forecasting in Realistic Settings*. arXiv:2504.09340.
5. Lam et al. (2023). *Learning skillful medium-range global weather forecasting*.
   Science, 10.1126/science.adi2336.
6. Liu et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted
   Windows*. ICCV.
7. DeepSeek-AI (2026). *DeepSeek-V4: Towards Highly Efficient Million-Token Context
   Intelligence*. arXiv:2606.19348.
