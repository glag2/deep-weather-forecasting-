"""Produce una previsione a tre giorni sull'ultima finestra disponibile.

ERA5 pubblica con circa sei giorni di ritardo, quindi la previsione riguarda giorni
gia' trascorsi ed e' verificabile contro l'osservato. E' un limite della sorgente, non
del modello.

Uso:
    python scripts/predict_forecast.py --fold 0
    python scripts/predict_forecast.py --fold 0 --save-table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from dwf.config import Config
from dwf.data.dataset import build_reader
from dwf.predict import (
    forecast_to_table,
    latest_usable_start,
    load_calibrator,
    predict_window,
    summarize,
)
from dwf.tables import FORECAST, SLOTS, read_table, write_table
from dwf.train import fold_dir, load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--start", type=int, default=None,
        help="Slot iniziale della finestra di input; senza, si usa la piu' recente.",
    )
    parser.add_argument(
        "--save-table", action="store_true",
        help="Scrive la previsione completa in forma lunga su Parquet.",
    )
    parser.add_argument(
        "--stride", type=int, default=4,
        help="Sottocampionamento della griglia nella tabella salvata.",
    )
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="Usa le probabilita' grezze della rete, senza correggerne la scala.",
    )
    args = parser.parse_args()

    config = Config.load(args.config, project_root=PROJECT_ROOT)
    network, stats, input_layout, output_layout = load_checkpoint(config, args.fold)
    reader = build_reader(config, input_layout)

    catalogo = read_table(SLOTS, config.tables_dir)
    utilizzabili = catalogo.sort("slot_index").get_column("usable").to_numpy()
    inizio = args.start if args.start is not None else latest_usable_start(config, utilizzabili)

    calibratore = None if args.no_calibration else load_calibrator(config, args.fold)
    previsione = predict_window(
        config, network, input_layout, output_layout, stats, reader, inizio,
        calibrator=calibratore,
    )

    if calibratore is None:
        print("probabilita' non calibrate: eseguire prima scripts/evaluate_model.py")
    else:
        print(f"probabilita' calibrate su {calibratore.n_samples:,} casi di validazione")
    print(f"fold {args.fold}, finestra che inizia allo slot {inizio}")
    print(f"ultimo istante osservato: {previsione.init_time}")
    print(f"dominio: {previsione.t2m_mean.shape[1]} x {previsione.t2m_mean.shape[2]}")
    print()
    with pl.Config(tbl_rows=12, tbl_width_chars=140):
        print(summarize(previsione))

    if args.save_table:
        tabella = forecast_to_table(previsione, stride=args.stride)
        destinazione = fold_dir(config, args.fold)
        destinazione.mkdir(parents=True, exist_ok=True)
        percorso = write_table(tabella, FORECAST, destinazione)
        print(f"\nprevisione salvata: {percorso} ({tabella.height:,} righe)")


if __name__ == "__main__":
    main()
