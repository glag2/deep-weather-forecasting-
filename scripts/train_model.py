"""Addestra uno o piu' fold della validazione a finestra mobile.

I fold vanno addestrati separatamente: ognuno ha le proprie statistiche e il proprio
checkpoint. Con `--fold` se ne sceglie uno, senza vengono addestrati tutti quelli che
hanno abbastanza dati ingeriti.

Uso:
    python scripts/train_model.py --fold 0 --epochs 5
    python scripts/train_model.py --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dwf.config import Config
from dwf.data.dataset import sample_starts
from dwf.train import TrainingError, train_fold

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def override_training(config: Config, **valori: object) -> Config:
    """Applica override alla sezione di addestramento, rivalidando la configurazione.

    La configurazione e' immutabile per scelta, cosi' nessun modulo puo' cambiarla di
    nascosto a meta' esecuzione: gli override passano da qui e vengono rivalidati.
    """
    filtrati = {chiave: valore for chiave, valore in valori.items() if valore is not None}
    if not filtrati:
        return config
    dati = config.model_dump()
    dati["training"].update(filtrati)
    # `project_root` e' un campo della configurazione, quindi il dump lo conserva e i
    # percorsi restano ancorati alla stessa radice.
    return Config.model_validate(dati)


def fold_availability(config: Config) -> list[tuple[int, int, int, int]]:
    """Per ogni fold, quante finestre sono ammesse in train, validazione e test."""
    righe = []
    for indice in range(len(config.build_folds())):
        righe.append(
            (
                indice,
                len(sample_starts(config, indice, "train")),
                len(sample_starts(config, indice, "val")),
                len(sample_starts(config, indice, "test")),
            )
        )
    return righe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--fold", type=int, action="append", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--list", action="store_true", help="Mostra i fold e le finestre disponibili."
    )
    args = parser.parse_args()

    config = Config.load(args.config, project_root=PROJECT_ROOT)
    config = override_training(
        config,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        learning_rate=args.learning_rate,
    )

    disponibilita = fold_availability(config)
    if args.list:
        print(f"{'fold':>5} {'train':>8} {'val':>8} {'test':>8}  addestrabile")
        for indice, n_train, n_val, n_test in disponibilita:
            pronto = "si" if n_train and n_val else "no"
            print(f"{indice:5d} {n_train:8d} {n_val:8d} {n_test:8d}  {pronto}")
        return

    if args.fold is not None:
        richiesti = args.fold
    else:
        richiesti = [
            indice for indice, n_train, n_val, _ in disponibilita if n_train and n_val
        ]
        if not richiesti:
            print(
                "Nessun fold ha insieme finestre di train e di validazione. "
                "Ingerire piu' mesi, oppure usare --list per vedere lo stato."
            )
            raise SystemExit(1)

    print(
        f"addestramento di {len(richiesti)} fold, {config.training.epochs} epoche, "
        f"{config.training.samples_per_epoch} campioni per epoca, "
        f"crop {config.training.crop_size}"
    )

    esiti = []
    for indice in richiesti:
        try:
            esiti.append(train_fold(config, indice))
        except TrainingError as errore:
            print(f"fold {indice}: {errore}")
            continue

    if not esiti:
        raise SystemExit(1)

    print("\nriepilogo:")
    for esito in esiti:
        print(
            f"  fold {esito.fold}: migliore all'epoca {esito.best_epoch}, "
            f"validazione {esito.best_val_loss:.4f} -> {esito.checkpoint}"
        )


if __name__ == "__main__":
    main()
