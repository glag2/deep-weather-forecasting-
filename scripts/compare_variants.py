"""Banco di prova comparativo fra varianti di modello e scelte di preparazione del dato.

Il confronto ha valore solo se **cambia una cosa sola per volta**: stessi dati, stesse
finestre, stesso seme, stesso numero di passate, stessa perdita, stesso protocollo di
valutazione. Qui cambia la sola configurazione in prova, e tutto il resto e' tenuto
fermo per costruzione, riusando lo stesso `train_fold` della pipeline vera.

Il protocollo e' volutamente **ridotto** rispetto all'addestramento finale: ritagli piu'
piccoli e poche passate. Non serve a produrre il modello da consegnare, serve a
ordinare le alternative spendendo ore invece che giorni. Un protocollo ridotto puo'
favorire i modelli che convergono in fretta, ed e' un limite che va tenuto presente
leggendo la tabella: la variante vincente viene poi riaddestrata a scala piena.

Il riferimento da battere non e' la persistenza ingenua ma quella **diurna**, cioe'
l'ultima osservazione alla stessa ora del giorno, che sui dati di questo progetto
raggiunge gia' un errore quadratico di 3,16 gradi.

Uso:
    python scripts/compare_variants.py --list
    python scripts/compare_variants.py --epochs 3 --crop 64 --samples 192
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:  # pragma: no cover - avvio da riga di comando
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dwf.config import Config  # noqa: E402
from dwf.data.dataset import build_reader, sample_starts  # noqa: E402
from dwf.data.features import InputLayout  # noqa: E402
from dwf.evaluate import collect_predictions, persistence_baseline  # noqa: E402
from dwf.models import variants  # noqa: E402
from dwf.models.heads import OutputLayout  # noqa: E402
from dwf.train import build_network, load_checkpoint, train_fold  # noqa: E402


@dataclass
class Prova:
    """Una configurazione in gara, descritta dagli scarti rispetto a quella di base."""

    nome: str
    descrizione: str
    modello: dict[str, object] = field(default_factory=dict)
    perdita: dict[str, object] = field(default_factory=dict)


@dataclass
class Esito:
    nome: str
    descrizione: str
    parametri: int
    perdita_val: float
    epoca_migliore: int
    rmse_celsius: float
    secondi_per_epoca: float
    errore: str = ""


def prove_predefinite() -> list[Prova]:
    """Le configurazioni confrontate.

    Le prime due isolano le scelte di preparazione del dato, le altre le architetture.
    Sono nella stessa gara di proposito: serve sapere se conti di piu' cambiare il
    modello o cambiare cosa gli si chiede di prevedere.
    """
    elenco = [
        Prova(
            "base_senza_ancoraggio",
            "Convoluzione, bersaglio assoluto: com'era prima dell'analisi dei dati.",
            modello={"variant": "conv", "anchor_diurnal": False},
        ),
        Prova(
            "conv",
            "Convoluzione con bersaglio ancorato alla persistenza diurna.",
            modello={"variant": "conv", "anchor_diurnal": True},
        ),
        Prova(
            "conv_spettrale",
            "Come sopra, piu' il termine spettrale contro lo smorzamento.",
            modello={"variant": "conv", "anchor_diurnal": True},
            perdita={"spectral": 0.1},
        ),
    ]
    elenco += [
        Prova(
            nome,
            variants.describe(nome),
            modello={"variant": nome, "anchor_diurnal": True},
        )
        for nome in variants.available()
        if nome != "conv"
    ]
    return elenco


def configura(base: Config, prova: Prova, args: argparse.Namespace) -> Config:
    """Applica alla configurazione di base gli scarti della prova e i limiti del banco."""
    modello = base.model.model_copy(update=prova.modello)
    pesi = base.training.loss_weights.model_copy(update=prova.perdita)
    allenamento = base.training.model_copy(
        update={
            "loss_weights": pesi,
            "epochs": args.epochs,
            "crop_size": args.crop,
            "samples_per_epoch": args.samples,
            # Seme identico per tutte: le finestre estratte e l'inizializzazione
            # partono dallo stesso stato, quindi la differenza misurata e' la variante.
            # Ripetere la stessa prova cambiando solo questo valore misura invece la
            # dispersione fra ripetizioni, cioe' quanto scarto e' rumore.
            "seed": args.seed if args.seed is not None else base.training.seed,
        }
    )
    suffisso = "" if args.seed is None else f"_seed{args.seed}"
    percorsi = base.paths.model_copy(
        update={"artifacts_subdir": f"bench/{prova.nome}{suffisso}"}
    )
    return base.model_copy(
        update={"model": modello, "training": allenamento, "paths": percorsi}
    )


def rmse_su_validazione(config: Config, fold: int, finestre: int) -> float:
    """Errore quadratico in gradi sul blocco di validazione, in unita' fisiche."""
    rete, stats, input_layout, output_layout = load_checkpoint(config, fold)
    lettore = build_reader(config, input_layout)
    inizi = sample_starts(config, fold, "val")
    if not inizi:
        raise RuntimeError("Nessuna finestra di validazione")
    from dwf.data.dataset import WeatherWindowDataset

    dataset = WeatherWindowDataset(
        config, input_layout, stats, inizi, lettore, crop_size=None, crops_per_window=1
    )
    previsioni = collect_predictions(
        rete, dataset, output_layout, config, max_windows=finestre
    )
    scarto = (previsioni.t2m_mean - previsioni.t2m_target) * stats.std["t2m"]
    return float(np.sqrt(np.mean(scarto**2)))


def riferimento_diurno(config: Config, fold: int, finestre: int) -> float:
    """Errore della persistenza diurna sulle stesse finestre, come metro di paragone."""
    from dwf.data.dataset import WeatherWindowDataset
    from dwf.train import compute_fold_stats, training_slots

    input_layout = InputLayout.from_config(config)
    lettore = build_reader(config, input_layout)
    inizi = sample_starts(config, fold, "val")
    stats = compute_fold_stats(
        config,
        input_layout,
        training_slots(
            sample_starts(config, fold, "train"),
            config.windows.input_slots + config.windows.output_slots,
        ),
    )
    dataset = WeatherWindowDataset(
        config, input_layout, stats, inizi, lettore, crop_size=None, crops_per_window=1
    )
    previsioni = persistence_baseline(dataset, max_windows=finestre, mode="diurnal")
    scarto = (previsioni.t2m_mean - previsioni.t2m_target) * stats.std["t2m"]
    return float(np.sqrt(np.mean(scarto**2)))


def esegui(prova: Prova, base: Config, args: argparse.Namespace) -> Esito:
    config = configura(base, prova, args)
    layout = OutputLayout.from_targets(config.targets, config.windows.output_slots)
    parametri = build_network(
        config, layout, InputLayout.from_config(config).n_channels
    ).n_parameters

    avvio = time.perf_counter()
    try:
        risultato = train_fold(config, args.fold, verbose=False)
        durata = (time.perf_counter() - avvio) / max(1, len(risultato.history))
        rmse = rmse_su_validazione(config, args.fold, args.eval_windows)
    except Exception as errore:
        traceback.print_exc()
        return Esito(prova.nome, prova.descrizione, parametri, float("nan"), -1,
                     float("nan"), float("nan"), str(errore)[:120])

    return Esito(
        nome=prova.nome,
        descrizione=prova.descrizione,
        parametri=parametri,
        perdita_val=risultato.best_val_loss,
        epoca_migliore=risultato.best_epoch,
        rmse_celsius=rmse,
        secondi_per_epoca=durata,
    )


def scrivi_tabella(esiti: list[Esito], riferimento: float, args: argparse.Namespace,
                   destinazione: Path) -> None:
    righe = [
        "# Confronto fra varianti",
        "",
        f"Protocollo identico per tutte: fold {args.fold}, {args.epochs} passate, "
        f"ritagli {args.crop}x{args.crop}, {args.samples} campioni per passata, "
        f"seme unico, stesse finestre.",
        "",
        f"Riferimento da battere, **persistenza diurna**: RMSE {riferimento:.3f} gradi "
        f"sulle stesse finestre di validazione.",
        "",
        "| variante | parametri | perdita val | RMSE degC | vs riferimento | s/passata |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for esito in sorted(esiti, key=lambda e: (np.isnan(e.rmse_celsius), e.rmse_celsius)):
        if esito.errore:
            righe.append(f"| {esito.nome} | {esito.parametri:,} | fallita: {esito.errore} | | | |")
            continue
        guadagno = 1.0 - esito.rmse_celsius / riferimento
        righe.append(
            f"| {esito.nome} | {esito.parametri:,} | {esito.perdita_val:.4f} | "
            f"{esito.rmse_celsius:.3f} | {guadagno:+.1%} | {esito.secondi_per_epoca:.0f} |"
        )
    righe += ["", "## Descrizione delle varianti", ""]
    righe += [f"- **{e.nome}**: {e.descrizione}" for e in esiti]
    righe += [
        "",
        "Le varianti che perdono non vengono rimosse: restano in "
        "`src/dwf/models/variants/` e si selezionano da configurazione con "
        "`model.variant`, cosi' il confronto e' ripetibile e una scelta diversa resta "
        "a portata di mano se i dati cambiano.",
        "",
    ]
    destinazione.write_text("\n".join(righe), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--crop", type=int, default=64)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--eval-windows", type=int, default=40)
    parser.add_argument("--only", nargs="*", help="Esegue solo le prove indicate.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sovrascrive il seme: ripetere una prova con semi diversi misura il rumore.",
    )
    parser.add_argument("--list", action="store_true", help="Elenca le prove e termina.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/VARIANTS.md")
    args = parser.parse_args()

    prove = prove_predefinite()
    if args.only:
        prove = [p for p in prove if p.nome in set(args.only)]
    if args.list:
        for prova in prove:
            print(f"{prova.nome:24s} {prova.descrizione}")
        return

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    base = Config.load(args.config, project_root=PROJECT_ROOT)

    riferimento = riferimento_diurno(base, args.fold, args.eval_windows)
    print(f"persistenza diurna sulla validazione: RMSE {riferimento:.3f} gradi\n")

    esiti: list[Esito] = []
    for indice, prova in enumerate(prove, start=1):
        print(f"[{indice}/{len(prove)}] {prova.nome} ...", flush=True)
        esito = esegui(prova, base, args)
        esiti.append(esito)
        if esito.errore:
            print(f"    fallita: {esito.errore}")
        else:
            print(
                f"    parametri {esito.parametri:,}, perdita val {esito.perdita_val:.4f}, "
                f"RMSE {esito.rmse_celsius:.3f} degC, {esito.secondi_per_epoca:.0f} s/passata",
                flush=True,
            )

    scrivi_tabella(esiti, riferimento, args, args.output)
    print(f"\ntabella scritta in {args.output}")


if __name__ == "__main__":
    main()
