"""Quanto ancorare la probabilita' di pioggia alla persistenza diurna.

Sul modello a scala piena la probabilita' di pioggia prevista varia fra la prima e
l'ultima scadenza circa **nove volte meno** di quanto vari la realta', e alla scadenza
piu' breve perde contro la persistenza. La testa non e' collassata: nessuna coppia di
scadenze e' identica bit a bit. Cio' che manca e' il riferimento. Le teste gaussiane
prevedono lo scarto dall'ultima osservazione alla stessa ora del giorno; le teste di
occorrenza, fino ad ora, no.

Questo banco misura se dare loro lo stesso riferimento aiuti, e con quale ampiezza.
L'ampiezza e' un logit sommato al logit di occorrenza, con segno positivo dove ieri
alla stessa ora pioveva e negativo dove non pioveva. A 0 il modello e' quello attuale,
quindi la prima riga della tabella e' il termine di paragone e non una prova a parte.

Non si guarda solo il Brier medio. Un modello puo' migliorare il Brier restando piatto,
e la piattezza e' il difetto da cui siamo partiti: la tabella riporta percio' anche
quanto la probabilita' prevista si muove fra le scadenze rispetto a quanto si muove
l'osservazione. Un rapporto vicino a 1 vuol dire che il modello segue la giornata.

Uso:
    python scripts/screen_occurrence_anchor.py --amplitudes 0 0.6 1.1 1.8 --seeds 1234 101
"""

from __future__ import annotations

import argparse
import statistics
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:  # pragma: no cover - avvio da riga di comando
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dwf.config import Config  # noqa: E402
from dwf.data.dataset import WeatherWindowDataset, build_reader, sample_starts  # noqa: E402
from dwf.data.features import InputLayout  # noqa: E402
from dwf.evaluate import collect_predictions, persistence_baseline  # noqa: E402
from dwf.train import compute_fold_stats, load_checkpoint, train_fold, training_slots  # noqa: E402


@dataclass
class Misura:
    """Cosa si osserva su una singola passata addestrata."""

    brier: float
    brier_per_scadenza: list[float]
    rapporto_movimento: float


@dataclass
class Esito:
    ampiezza: float
    misure: list[Misura] = field(default_factory=list)
    errore: str = ""

    @property
    def brier(self) -> float:
        return statistics.fmean(m.brier for m in self.misure) if self.misure else float("nan")

    @property
    def rapporto(self) -> float:
        return (
            statistics.fmean(m.rapporto_movimento for m in self.misure)
            if self.misure
            else float("nan")
        )

    def brier_medio_per_scadenza(self) -> list[float]:
        if not self.misure:
            return []
        return [
            statistics.fmean(m.brier_per_scadenza[i] for m in self.misure)
            for i in range(len(self.misure[0].brier_per_scadenza))
        ]


def _misura(probabilita: np.ndarray, occorrenza: np.ndarray, scadenza: np.ndarray) -> Misura:
    """Brier complessivo, Brier per scadenza e quanto il modello segue la giornata."""
    scadenze = sorted(set(int(v) for v in np.unique(scadenza)))
    per_scadenza: list[float] = []
    previste: list[float] = []
    osservate: list[float] = []
    for lead in scadenze:
        dove = scadenza == lead
        per_scadenza.append(float(np.mean((probabilita[dove] - occorrenza[dove]) ** 2)))
        previste.append(float(np.mean(probabilita[dove])))
        osservate.append(float(np.mean(occorrenza[dove])))

    # Quanto si muove la previsione fra le scadenze, diviso quanto si muove la realta'.
    # Serve la stessa quantita' sulle due, altrimenti si confronterebbe un errore con
    # una variabilita'. Se la realta' e' piatta il rapporto non e' definito.
    movimento_vero = max(osservate) - min(osservate)
    movimento_previsto = max(previste) - min(previste)
    rapporto = movimento_previsto / movimento_vero if movimento_vero > 1e-9 else float("nan")
    return Misura(
        brier=float(np.mean((probabilita - occorrenza) ** 2)),
        brier_per_scadenza=per_scadenza,
        rapporto_movimento=rapporto,
    )


def _configura(base: Config, ampiezza: float, seme: int, args: argparse.Namespace) -> Config:
    modello = base.model.model_copy(update={"occurrence_anchor_logit": ampiezza})
    allenamento = base.training.model_copy(
        update={
            "epochs": args.epochs,
            "crop_size": args.crop,
            "samples_per_epoch": args.samples,
            "seed": seme,
        }
    )
    etichetta = f"ancora{ampiezza:g}".replace(".", "")
    percorsi = base.paths.model_copy(
        update={"artifacts_subdir": f"bench/{etichetta}_seed{seme}"}
    )
    return base.model_copy(
        update={"model": modello, "training": allenamento, "paths": percorsi}
    )


def _dataset_validazione(config: Config, fold: int, stats, input_layout: InputLayout):
    inizi = sample_starts(config, fold, "val")
    if not inizi:
        raise RuntimeError("Nessuna finestra di validazione")
    return WeatherWindowDataset(
        config,
        input_layout,
        stats,
        inizi,
        build_reader(config, input_layout),
        crop_size=None,
        crops_per_window=1,
    )


def riferimento_persistenza(config: Config, fold: int, finestre: int) -> Misura:
    """La stessa misura sulla persistenza diurna, sulle stesse finestre."""
    input_layout = InputLayout.from_config(config)
    stats = compute_fold_stats(
        config,
        input_layout,
        training_slots(
            sample_starts(config, fold, "train"),
            config.windows.input_slots + config.windows.output_slots,
        ),
    )
    dataset = _dataset_validazione(config, fold, stats, input_layout)
    previsioni = persistence_baseline(dataset, max_windows=finestre, mode="diurnal")
    return _misura(previsioni.tp_probability, previsioni.tp_occurrence, previsioni.lead)


def esegui(ampiezza: float, semi: list[int], base: Config, args: argparse.Namespace) -> Esito:
    esito = Esito(ampiezza=ampiezza)
    for seme in semi:
        config = _configura(base, ampiezza, seme, args)
        try:
            train_fold(config, args.fold, verbose=False)
            rete, stats, input_layout, output_layout = load_checkpoint(config, args.fold)
            dataset = _dataset_validazione(config, args.fold, stats, input_layout)
            previsioni = collect_predictions(
                rete, dataset, output_layout, config, max_windows=args.eval_windows
            )
        except Exception as errore:  # pragma: no cover - percorso diagnostico
            traceback.print_exc()
            esito.errore = str(errore)[:120]
            return esito
        misura = _misura(
            previsioni.tp_probability, previsioni.tp_occurrence, previsioni.lead
        )
        esito.misure.append(misura)
        print(
            f"    seme {seme}: Brier {misura.brier:.4f}, "
            f"movimento {misura.rapporto_movimento:.2f}x il vero",
            flush=True,
        )
    return esito


def scrivi_tabella(
    esiti: list[Esito], riferimento: Misura, args: argparse.Namespace, destinazione: Path
) -> None:
    validi = [e for e in esiti if e.misure]
    righe = [
        "# Ancoraggio della probabilita' di pioggia alla persistenza diurna",
        "",
        "Generato da `scripts/screen_occurrence_anchor.py`.",
        "",
        "> **Esito finale: l'ancoraggio e' spento.** Quanto segue e' la misura a scala",
        "> ridotta, che indicava ampiezza 1,1. A piena scala il verdetto si e' rovesciato:",
        "> Brier di test 0,181 senza ancoraggio contro 0,192 con, peggio in otto scadenze",
        "> su nove. L'ancoraggio e' una conoscenza a priori, e una conoscenza a priori vale",
        "> tanto di piu' quanto meno il modello ha imparato da solo: un banco da poche",
        "> passate misura chi impara in fretta, quindi premia per costruzione i termini a",
        "> priori. Questa pagina resta come prova di come il metodo puo' sbagliare.",
        "",
        "## Cosa si sta misurando",
        "",
        "La probabilita' di pioggia del modello a scala piena varia fra la prima e",
        "l'ultima scadenza circa nove volte meno di quanto vari la realta'. La testa non",
        "e' collassata: nessuna coppia di scadenze e' identica. Quello che manca e' il",
        "riferimento, che le teste gaussiane hanno e quelle di occorrenza no.",
        "",
        "L'ampiezza e' un logit sommato al logit di occorrenza: positivo dove ieri alla",
        "stessa ora pioveva, negativo dove non pioveva. **Ampiezza 0 e' il modello",
        "attuale**, quindi la prima riga e' il termine di paragone, non una prova.",
        "",
        "La colonna *movimento* e' l'escursione della probabilita' prevista fra le",
        "scadenze divisa per l'escursione osservata: 1,00 vuol dire che il modello segue",
        "la giornata come la realta', 0,10 che la appiattisce di dieci volte. Va letta",
        "insieme al Brier, perche' un modello puo' migliorare il Brier restando piatto.",
        "",
        "## Protocollo",
        "",
        f"Fold {args.fold}, {args.epochs} passate, ritaglio {args.crop}, "
        f"{args.samples} campioni per passata, semi {args.seeds}.",
        "",
        "## Risultati",
        "",
        "| ampiezza | Brier | movimento | Brier prima scadenza | Brier ultima scadenza |",
        "|---:|---:|---:|---:|---:|",
        f"| persistenza diurna | {riferimento.brier:.4f} | "
        f"{riferimento.rapporto_movimento:.2f} | "
        f"{riferimento.brier_per_scadenza[0]:.4f} | "
        f"{riferimento.brier_per_scadenza[-1]:.4f} |",
    ]
    for esito in esiti:
        if not esito.misure:
            righe.append(
                f"| {esito.ampiezza:g} | fallita | - | - | - | <!-- {esito.errore} -->"
            )
            continue
        per_scadenza = esito.brier_medio_per_scadenza()
        etichetta = f"{esito.ampiezza:g}" + (" (attuale)" if esito.ampiezza == 0 else "")
        righe.append(
            f"| {etichetta} | {esito.brier:.4f} | {esito.rapporto:.2f} | "
            f"{per_scadenza[0]:.4f} | {per_scadenza[-1]:.4f} |"
        )

    righe += ["", "## Lettura", ""]
    if len(validi) < 2:
        righe.append("Prove insufficienti per un confronto.")
    else:
        base = next((e for e in validi if e.ampiezza == 0), None)
        migliore = min(validi, key=lambda e: e.brier)
        dispersioni = [
            max(m.brier for m in e.misure) - min(m.brier for m in e.misure)
            for e in validi
            if len(e.misure) > 1
        ]
        rumore = statistics.fmean(dispersioni) if dispersioni else float("nan")
        if base is None:
            righe.append("Manca l'ampiezza 0: senza termine di paragone non si conclude.")
        elif rumore != rumore:
            righe.append("Un solo seme per ampiezza: l'incertezza non e' stimabile.")
        else:
            guadagno = base.brier - migliore.brier
            righe += [
                f"Brier migliore: **ampiezza {migliore.ampiezza:g}** "
                f"({migliore.brier:.4f} contro {base.brier:.4f} del modello attuale, "
                f"guadagno {guadagno:+.4f}).",
                "",
                f"Dispersione fra semi: {rumore:.4f}. ",
            ]
            if guadagno > rumore:
                righe.append(
                    "Il guadagno supera la dispersione fra semi: l'ancoraggio "
                    "dell'occorrenza va adottato, con questa ampiezza."
                )
            else:
                righe.append(
                    "Il guadagno **non** supera la dispersione fra semi: a questo budget "
                    "l'ancoraggio non e' dimostrato utile e resta disattivato. "
                    "Il difetto di piattezza resta quindi aperto, e la sua causa non e' "
                    "la mancanza del riferimento."
                )
            righe += [
                "",
                f"Movimento fra le scadenze: modello attuale {base.rapporto:.2f}x, "
                f"migliore {migliore.rapporto:.2f}x, persistenza diurna "
                f"{riferimento.rapporto_movimento:.2f}x il vero.",
            ]

    righe += [
        "",
        "## Limiti",
        "",
        "1. Protocollo ridotto e un solo fold, come per gli altri banchi.",
        "2. Il Brier e' misurato sulla validazione, non sul test: serve a scegliere, non",
        "   a dichiarare la bravura del modello finale.",
        "3. L'ampiezza e' fissa e uguale per tutte le scadenze. Alla scadenza piu' lunga",
        "   la persistenza vale meno, quindi il valore migliore qui e' un compromesso.",
        "",
    ]
    destinazione.write_text("\n".join(righe), encoding="utf-8")
    print(f"relazione scritta in {destinazione}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--crop", type=int, default=64)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--eval-windows", type=int, default=24)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.0, 0.6, 1.1, 1.8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 101])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/OCCURRENCE_ANCHOR.md")
    args = parser.parse_args()

    base = Config.load(args.config)
    if 0.0 not in args.amplitudes:
        raise SystemExit(
            "Serve l'ampiezza 0 come termine di paragone: senza, il confronto non dice "
            "se l'ancoraggio aiuti o peggiori."
        )

    riferimento = riferimento_persistenza(base, args.fold, args.eval_windows)
    print(
        f"persistenza diurna: Brier {riferimento.brier:.4f}, "
        f"movimento {riferimento.rapporto_movimento:.2f}x il vero",
        flush=True,
    )

    esiti: list[Esito] = []
    for ampiezza in args.amplitudes:
        print(f"\n[ampiezza {ampiezza:g}]", flush=True)
        esiti.append(esegui(ampiezza, args.seeds, base, args))

    scrivi_tabella(esiti, riferimento, args, args.output)


if __name__ == "__main__":
    main()
