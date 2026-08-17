"""Valuta un fold addestrato contro la persistenza, con calibrazione onesta.

Il numero che conta non e' l'errore del modello ma la differenza rispetto a una
previsione banale: un RMSE di 2 K puo' essere ottimo a tre giorni e pessimo a sei ore.

**Protocollo.** La mappa di calibrazione e la soglia di decisione si stimano sulla
**validazione** e si applicano al **test**. Stimarle e misurarle sugli stessi dati
darebbe un guadagno apparente che sparirebbe al primo dato nuovo.

Uso:
    python scripts/evaluate_model.py --fold 0 --split test
    python scripts/evaluate_model.py --fold 0 --split val --no-calibration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from dwf.calibration import CalibrationError, calibration_error, fit_calibrator
from dwf.config import Config
from dwf.data.dataset import WeatherWindowDataset, build_reader, sample_starts
from dwf.evaluate import (
    best_f1_threshold,
    brier_score,
    collect_predictions,
    metrics_table,
    persistence_baseline,
    reliability_table,
)
from dwf.tables import CALIBRATION, METRICS, RELIABILITY, write_table
from dwf.train import fold_dir, load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def costruisci_dataset(config, input_layout, stats, fold, split, reader):
    starts = sample_starts(config, fold, split)
    if not starts:
        return None
    return WeatherWindowDataset(
        config, input_layout, stats, starts, reader,
        crop_size=None, crops_per_window=1, seed=config.training.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--max-windows", type=int, default=None,
        help="Limita le finestre valutate, utile per una verifica rapida.",
    )
    parser.add_argument(
        "--calibration-windows", type=int, default=None,
        help="Limita le finestre di validazione usate per stimare la calibrazione.",
    )
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="Salta la calibrazione e usa le probabilita' grezze della rete.",
    )
    args = parser.parse_args()

    config = Config.load(args.config, project_root=PROJECT_ROOT)
    network, stats, input_layout, output_layout = load_checkpoint(config, args.fold)
    reader = build_reader(config, input_layout)

    dataset = costruisci_dataset(config, input_layout, stats, args.fold, args.split, reader)
    if dataset is None:
        print(f"Nessuna finestra ammessa per fold {args.fold} split {args.split}.")
        raise SystemExit(1)

    print(
        f"fold {args.fold}, split {args.split}: {len(dataset.starts)} finestre, "
        f"dominio {reader.shape[0]} x {reader.shape[1]}"
    )

    previsioni = collect_predictions(
        network, dataset, output_layout, config, max_windows=args.max_windows
    )
    riferimento = persistence_baseline(dataset, max_windows=args.max_windows)

    # ---------------------------------------------------------------- #
    # Calibrazione stimata sulla validazione
    # ---------------------------------------------------------------- #
    calibratore = None
    soglia_pioggia = 0.5
    soglia_neve = 0.5
    tabella_calibrazione = None

    if not args.no_calibration:
        if args.split == "val":
            print(
                "\nAttenzione: stimare e misurare la calibrazione sulla stessa "
                "validazione sovrastima il guadagno. Usare --split test."
            )
        taratura = costruisci_dataset(config, input_layout, stats, args.fold, "val", reader)
        if taratura is None:
            print("\nNessuna finestra di validazione: calibrazione saltata.")
        else:
            print(f"\ncalibrazione stimata su {len(taratura.starts)} finestre di validazione")
            su_val = collect_predictions(
                network, taratura, output_layout, config,
                max_windows=args.calibration_windows,
            )
            try:
                calibratore = fit_calibrator(
                    su_val.tp_probability, su_val.tp_occurrence,
                    variable="tp", fitted_on_split="val",
                )
            except CalibrationError as errore:
                print(f"  calibrazione non stimabile: {errore}")
            else:
                soglia_pioggia, f1_val = best_f1_threshold(
                    calibratore.apply(su_val.tp_probability), su_val.tp_occurrence
                )
                print(
                    f"  nodi: {calibratore.knots_in.size}, "
                    f"soglia pioggia scelta: {soglia_pioggia:.2f} (F1 {f1_val:.3f} su val)"
                )
                tabella_calibrazione = calibratore.to_table(fold=args.fold)

            # La neve ha una soglia propria. La probabilita' di neve e' il prodotto di
            # due probabilita', quindi vive su una scala molto piu' bassa: lasciarla a
            # 0,5 significherebbe non prevedere quasi mai neve, e misurare quello.
            if su_val.snow_probability is not None:
                calibrata = (
                    su_val.with_calibrated_tp(calibratore)
                    if calibratore is not None
                    else su_val
                )
                soglia_neve, f1_neve = best_f1_threshold(
                    calibrata.snow_probability, calibrata.snow_occurrence
                )
                print(
                    f"  soglia neve scelta: {soglia_neve:.2f} (F1 {f1_neve:.3f} su val)"
                )

    if calibratore is not None:
        grezze = previsioni
        previsioni = previsioni.with_calibrated_tp(calibratore)
        esiti = grezze.tp_occurrence
        prima_ece = calibration_error(grezze.tp_probability, esiti)
        dopo_ece = calibration_error(previsioni.tp_probability, esiti)
        prima_brier = brier_score(grezze.tp_probability, esiti)
        dopo_brier = brier_score(previsioni.tp_probability, esiti)
        print(f"\neffetto della calibrazione sul {args.split}:")
        print(f"  errore di calibrazione {prima_ece:.4f} -> {dopo_ece:.4f}")
        print(f"  Brier                  {prima_brier:.4f} -> {dopo_brier:.4f}")

    # ---------------------------------------------------------------- #
    # Metriche e salvataggio
    # ---------------------------------------------------------------- #
    metriche = pl.concat(
        [
            metrics_table(
                previsioni, stats, model="dwf", split=args.split, fold=args.fold,
                rain_threshold=soglia_pioggia, snow_threshold=soglia_neve,
            ),
            # La persistenza e' gia' binaria: la sua soglia non ha nulla da ottimizzare.
            metrics_table(
                riferimento, stats, model="persistence", split=args.split, fold=args.fold,
                rain_threshold=0.5, snow_threshold=0.5,
            ),
        ]
    )
    affidabilita = reliability_table(
        previsioni.tp_probability, previsioni.tp_occurrence,
        model="dwf", split=args.split, fold=args.fold, variable="tp",
    )

    destinazione = fold_dir(config, args.fold)
    destinazione.mkdir(parents=True, exist_ok=True)
    print(f"\nmetriche    : {write_table(metriche, METRICS, destinazione)}")
    print(f"affidabilita: {write_table(affidabilita, RELIABILITY, destinazione)}")
    if tabella_calibrazione is not None:
        print(f"calibrazione: {write_table(tabella_calibrazione, CALIBRATION, destinazione)}")

    aggregate = metriche.filter((pl.col("lead_slot") == -1) & (pl.col("month") == -1))
    confronto = (
        aggregate.pivot(on="model", index=["variable", "metric"], values="value")
        .sort("variable", "metric")
    )
    print("\nsintesi su tutte le scadenze:")
    with pl.Config(tbl_rows=40, tbl_width_chars=110):
        print(confronto)

    per_scadenza = (
        metriche.filter(
            (pl.col("lead_slot") >= 0)
            & (pl.col("month") == -1)
            & (pl.col("metric").is_in(["rmse_celsius", "f1", "brier_skill_score"]))
        )
        .pivot(on="model", index=["variable", "metric", "lead_slot"], values="value")
        .sort("variable", "metric", "lead_slot")
    )
    print("\nper scadenza:")
    with pl.Config(tbl_rows=40, tbl_width_chars=110):
        print(per_scadenza)

    print("\naffidabilita' della probabilita' di pioggia:")
    with pl.Config(tbl_rows=12):
        print(
            affidabilita.select(
                "bin_lower", "forecast_mean", "observed_frequency", "count"
            )
        )


if __name__ == "__main__":
    main()
