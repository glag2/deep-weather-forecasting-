"""Valuta un fold addestrato contro la persistenza e la climatologia.

Il numero che conta non e' l'errore del modello ma la differenza rispetto a una
previsione banale: un RMSE di 2 K puo' essere ottimo a tre giorni e pessimo a sei ore.

Uso:
    python scripts/evaluate_model.py --fold 0 --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from dwf.config import Config
from dwf.data.dataset import WeatherWindowDataset, build_reader, sample_starts
from dwf.evaluate import (
    collect_predictions,
    metrics_table,
    persistence_baseline,
    reliability_table,
)
from dwf.tables import METRICS, RELIABILITY, write_table
from dwf.train import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--max-windows", type=int, default=None,
        help="Limita le finestre valutate, utile per una verifica rapida.",
    )
    args = parser.parse_args()

    config = Config.load(args.config, project_root=PROJECT_ROOT)
    network, stats, input_layout, output_layout = load_checkpoint(config, args.fold)

    starts = sample_starts(config, args.fold, args.split)
    if not starts:
        print(f"Nessuna finestra ammessa per fold {args.fold} split {args.split}.")
        raise SystemExit(1)

    reader = build_reader(config, input_layout)
    dataset = WeatherWindowDataset(
        config, input_layout, stats, starts, reader,
        crop_size=None, crops_per_window=1, seed=config.training.seed,
    )
    print(
        f"fold {args.fold}, split {args.split}: {len(starts)} finestre, "
        f"dominio {reader.shape[0]} x {reader.shape[1]}"
    )

    previsioni = collect_predictions(
        network, dataset, output_layout, config, max_windows=args.max_windows
    )
    riferimento = persistence_baseline(dataset, max_windows=args.max_windows)

    metriche = pl.concat(
        [
            metrics_table(
                previsioni, stats, model="dwf", split=args.split, fold=args.fold
            ),
            metrics_table(
                riferimento, stats, model="persistence", split=args.split, fold=args.fold
            ),
        ]
    )
    affidabilita = reliability_table(
        previsioni.tp_probability, previsioni.tp_occurrence,
        model="dwf", split=args.split, fold=args.fold, variable="tp",
    )

    destinazione = config.artifacts_dir / f"fold_{args.fold:02d}"
    destinazione.mkdir(parents=True, exist_ok=True)
    print(f"metriche    : {write_table(metriche, METRICS, destinazione)}")
    print(f"affidabilita: {write_table(affidabilita, RELIABILITY, destinazione)}")

    confronto = (
        metriche.filter((pl.col("lead_slot") >= 0) & (pl.col("month") == -1))
        .pivot(on="model", index=["variable", "metric", "lead_slot"], values="value")
        .sort("variable", "metric", "lead_slot")
    )
    print("\nmodello contro persistenza, per scadenza:")
    with pl.Config(tbl_rows=40, tbl_width_chars=120):
        print(confronto)

    print("\naffidabilita' della probabilita' di pioggia:")
    with pl.Config(tbl_rows=12):
        print(
            affidabilita.select(
                "bin_lower", "bin_upper", "forecast_mean", "observed_frequency", "count"
            )
        )


if __name__ == "__main__":
    main()
