"""Screening delle famiglie di canali contro il *cambiamento* futuro.

Perche' non si misura la correlazione col valore futuro
-------------------------------------------------------
La temperatura di domani alle 12 e' quasi uguale a quella di oggi alle 12. Qualunque
canale che porti traccia della temperatura attuale mostrera' quindi una correlazione
altissima col bersaglio, senza aggiungere **nulla** a un riferimento che si ottiene
gratis. Uno screening basato sul valore futuro promuoverebbe i canali inutili.

Qui il bersaglio e' il **residuo** rispetto alla persistenza diurna, cioe' esattamente
la quantita' che il modello ancorato deve produrre. Un canale vale se aiuta a prevedere
quanto il tempo *cambiera'*, non quanto somigliera' a se stesso.

Il metodo
---------
Sonda lineare con regolarizzazione di Ridge, addestrata sui punti del blocco di train e
misurata su quelli di validazione, mai sugli stessi. Per ogni famiglia si riportano due
numeri complementari:

- **da sola**: quanta varianza del residuo spiega la famiglia isolata; dice se il
  segnale c'e';
- **togliendola**: quanta se ne perde rimuovendola da tutte le altre; dice se il segnale
  e' *suo* o duplicato altrove.

Una famiglia forte da sola ma con perdita nulla in ablazione e' ridondante: la sua
informazione e' gia' presente altrove, e toglierla non costa niente.

Limite dichiarato: la sonda e' **lineare**. Misura il segnale accessibile linearmente,
che e' un limite inferiore. Una famiglia che risulta debole qui potrebbe comunque
servire alla rete, che e' non lineare; una che risulta forte serve di sicuro.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:  # pragma: no cover - avvio da riga di comando
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dwf.config import Config  # noqa: E402
from dwf.data.dataset import (  # noqa: E402
    KEY_BASELINE_PREFIX,
    KEY_FEATURES,
    WeatherWindowDataset,
    build_reader,
    sample_starts,
)
from dwf.data.features import InputLayout  # noqa: E402
from dwf.train import compute_fold_stats  # noqa: E402

# Ridge minimo: serve solo a rendere invertibile la matrice normale quando due canali
# sono quasi collineari, non a regolarizzare per davvero.
RIDGE = 1e-3


def raccogli_punti(
    config: Config,
    layout: InputLayout,
    stats,
    fold: int,
    split: str,
    n_finestre: int,
    celle_per_finestra: int,
    variabile: str,
    lead: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Estrae coppie (canali, residuo) da celle campionate a caso.

    Si campionano celle sparse invece di ritagli contigui: due celle adiacenti portano
    quasi la stessa informazione, quindi un ritaglio da 64x64 vale molto meno di 4096
    punti indipendenti e gonfierebbe la fiducia nel risultato.
    """
    inizi = sample_starts(config, fold, split)
    if not inizi:
        raise RuntimeError(f"Nessuna finestra nel blocco {split!r}")
    rng.shuffle(inizi)
    inizi = inizi[:n_finestre]

    lettore = build_reader(config, layout)
    dataset = WeatherWindowDataset(
        config, layout, stats, inizi, lettore, crop_size=None, crops_per_window=1
    )

    chiave_bersaglio = f"target_{variabile}"
    chiave_riferimento = f"{KEY_BASELINE_PREFIX}{variabile}"
    caratteristiche: list[np.ndarray] = []
    residui: list[np.ndarray] = []

    for indice in range(len(dataset)):
        campione = dataset[indice]
        if chiave_riferimento not in campione:
            raise RuntimeError(
                f"Il campione non porta {chiave_riferimento!r}: l'ancoraggio diurno "
                f"deve essere attivo perche' lo screening abbia senso."
            )
        canali = campione[KEY_FEATURES].numpy()
        bersaglio = campione[chiave_bersaglio].numpy()[lead]
        riferimento = campione[chiave_riferimento].numpy()[lead]

        altezza, larghezza = bersaglio.shape
        righe = rng.integers(0, altezza, celle_per_finestra)
        colonne = rng.integers(0, larghezza, celle_per_finestra)

        caratteristiche.append(canali[:, righe, colonne].T)
        residui.append(bersaglio[righe, colonne] - riferimento[righe, colonne])

    return np.concatenate(caratteristiche), np.concatenate(residui)


def varianza_spiegata(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray
) -> float:
    """Frazione di varianza del residuo spiegata sul blocco di validazione.

    Zero significa "non meglio della persistenza diurna". Puo' essere negativo, e in tal
    caso la sonda sta peggiorando le cose invece di aiutare.
    """
    if x_train.shape[1] == 0:
        return 0.0
    # Intercetta esplicita: il residuo puo' avere una media non nulla.
    uno_train = np.ones((x_train.shape[0], 1), dtype=x_train.dtype)
    uno_val = np.ones((x_val.shape[0], 1), dtype=x_val.dtype)
    a = np.hstack([x_train, uno_train]).astype(np.float64)
    b = np.hstack([x_val, uno_val]).astype(np.float64)

    normale = a.T @ a + RIDGE * np.eye(a.shape[1])
    coefficienti = np.linalg.solve(normale, a.T @ y_train.astype(np.float64))

    previsti = b @ coefficienti
    residuo = float(np.sum((y_val - previsti) ** 2))
    totale = float(np.sum((y_val - np.mean(y_val)) ** 2))
    if totale == 0.0:
        return 0.0
    return 1.0 - residuo / totale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--variable", default="t2m")
    parser.add_argument(
        "--lead", type=int, default=2, help="Indice della scadenza (2 = +24 h)."
    )
    parser.add_argument("--windows", type=int, default=40)
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "FEATURES.md")
    args = parser.parse_args()

    config = Config.load(args.config)
    if not config.model.anchor_diurnal:
        raise SystemExit("Lo screening richiede model.anchor_diurnal attivo.")

    layout = InputLayout.from_config(config)
    slots_disponibili = sample_starts(config, args.fold, "train")
    stats = compute_fold_stats(config, layout, slots_disponibili)
    rng = np.random.default_rng(args.seed)

    print(f"raccolta punti (variabile {args.variable}, scadenza {args.lead}) ...")
    x_train, y_train = raccogli_punti(
        config, layout, stats, args.fold, "train",
        args.windows, args.cells, args.variable, args.lead, rng,
    )
    x_val, y_val = raccogli_punti(
        config, layout, stats, args.fold, "val",
        max(args.windows // 2, 4), args.cells, args.variable, args.lead, rng,
    )
    print(f"  train {x_train.shape[0]} punti, validazione {x_val.shape[0]} punti")
    print(f"  deviazione del residuo in validazione: {np.std(y_val):.3f}")

    gruppi: dict[str, list[int]] = defaultdict(list)
    for canale in layout.channels:
        gruppi[canale.group].append(canale.index)

    completo = varianza_spiegata(x_train, y_train, x_val, y_val)
    print(f"  tutte le famiglie insieme: {completo:+.4f}")

    righe: list[tuple[str, int, float, float]] = []
    for nome, indici in sorted(gruppi.items()):
        da_sola = varianza_spiegata(
            x_train[:, indici], y_train, x_val[:, indici], y_val
        )
        restanti = [i for i in range(layout.n_channels) if i not in set(indici)]
        senza = varianza_spiegata(
            x_train[:, restanti], y_train, x_val[:, restanti], y_val
        )
        perdita = completo - senza
        righe.append((nome, len(indici), da_sola, perdita))
        print(
            f"  {nome:12s} n={len(indici):4d}  "
            f"da sola {da_sola:+.4f}  togliendola {perdita:+.4f}"
        )

    righe.sort(key=lambda r: r[3], reverse=True)
    scrivi_relazione(
        args, layout, completo, float(np.std(y_val)),
        righe, x_train.shape[0], x_val.shape[0],
    )
    print(f"\nrelazione scritta in {args.output}")


def scrivi_relazione(
    args, layout: InputLayout, completo: float, dispersione: float,
    righe: list[tuple[str, int, float, float]], n_train: int, n_val: int,
) -> None:
    testo = [
        "# Screening delle famiglie di canali",
        "",
        "Generato da `scripts/screen_features.py`.",
        "",
        "## Che cosa misura",
        "",
        "Il bersaglio **non** e' il valore futuro ma il **residuo rispetto alla",
        "persistenza diurna**, cioe' quanto il tempo cambia rispetto a ieri alla stessa",
        "ora. E' la quantita' che il modello ancorato deve davvero produrre.",
        "",
        "La distinzione non e' formale. La temperatura di domani a mezzogiorno somiglia",
        "moltissimo a quella di oggi a mezzogiorno: qualunque canale che porti traccia",
        "della temperatura attuale mostrerebbe una correlazione altissima col valore",
        "futuro pur non aggiungendo nulla a un riferimento gratuito. Misurare sul",
        "residuo elimina quel merito apparente.",
        "",
        "## Protocollo",
        "",
        "| | |",
        "|---|---|",
        f"| variabile | `{args.variable}` |",
        f"| scadenza | indice {args.lead} |",
        f"| fold | {args.fold} |",
        f"| punti di train | {n_train} |",
        f"| punti di validazione | {n_val} |",
        f"| canali totali | {layout.n_channels} |",
        f"| deviazione del residuo | {dispersione:.3f} |",
        "",
        "Sonda lineare con Ridge, adattata sul blocco di train e misurata su quello di",
        "validazione. Le celle sono campionate sparse e non a ritagli contigui: celle",
        "adiacenti portano quasi la stessa informazione, e contarle come punti",
        "indipendenti gonfierebbe la fiducia nel risultato.",
        "",
        "## Risultati",
        "",
        f"Tutte le famiglie insieme spiegano **{completo:+.4f}** della varianza del residuo.",
        "",
        "| famiglia | canali | da sola | togliendola |",
        "|---|---:|---:|---:|",
    ]
    for nome, quanti, da_sola, perdita in righe:
        testo.append(f"| `{nome}` | {quanti} | {da_sola:+.4f} | {perdita:+.4f} |")
    testo += [
        "",
        "**Come si leggono le due colonne.** *Da sola* dice se in quella famiglia il",
        "segnale esiste. *Togliendola* dice se quel segnale e' suo o gia' disponibile",
        "altrove. Una famiglia forte da sola ma con perdita nulla in ablazione e'",
        "ridondante, e rimuoverla non costa nulla.",
        "",
        "## Limiti dichiarati",
        "",
        "1. **La sonda e' lineare.** Misura il segnale accessibile linearmente, che e' un",
        "   limite inferiore. Una famiglia debole qui puo' comunque servire alla rete,",
        "   che non lo e'; una famiglia forte qui serve di sicuro.",
        "2. **Una sola variabile e una sola scadenza per esecuzione.** Le conclusioni non",
        "   si trasferiscono automaticamente a pioggia e neve, che sono discontinue.",
        "3. **Un solo fold.** La stabilita' fra fold non e' verificata qui.",
        "4. La sonda opera per cella indipendente e quindi **non vede la struttura",
        "   spaziale**, che e' invece cio' che la rete convoluzionale sfrutta. Le famiglie",
        "   utili solo attraverso il contesto locale risultano sottostimate.",
    ]
    args.output.write_text("\n".join(testo) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
