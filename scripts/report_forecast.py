"""Produce il report PDF della previsione piu' recente disponibile.

Le mappe e gli andamenti sono la forma in cui la previsione diventa leggibile; la
pagina finale isola Vigo di Cadore e dichiara lo scarto di quota della cella, che e' la
principale causa di errore sistematico in montagna.

Uso:
    python scripts/report_forecast.py --fold 0
    python scripts/report_forecast.py --fold 0 --output report.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dwf.config import Config

# Le metriche vivono nella cartella del fold, non in `tables/`: questa funzione le cerca
# in entrambi i posti, e riusarla evita di riscrivere qui la ricerca sbagliata.
from dwf.dashboard import metriche
from dwf.data.dataset import ZarrWindowReader, build_reader
from dwf.predict import latest_usable_start, load_calibrator, predict_window
from dwf.report import report_path, write_report
from dwf.tables import SLOTS, read_table
from dwf.thermo import surface_pressure_from_msl
from dwf.train import fold_dir, load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def humidity_fields(
    reader: ZarrWindowReader, start: int, input_slots: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Rugiada e pressione al suolo dell'ultimo istante osservato, se disponibili.

    La rete non prevede l'umidita': per il calore latente si persiste l'ultimo campo
    osservato, che e' un'ipotesi debole ma dichiarata, e comunque piu' informativa
    dell'aria satura. La pressione al suolo va ricostruita dalla pressione al livello
    del mare, altrimenti in quota l'umidita' specifica sarebbe sbagliata.
    """
    finestra = reader.read_window(start, input_slots)
    if "d2m" not in finestra or "msl" not in finestra:
        return None, None

    rugiada = finestra["d2m"][-1]
    temperatura = finestra["t2m"][-1] if "t2m" in finestra else rugiada
    geopotenziale = reader.static.get("z")
    if geopotenziale is None:
        return rugiada, finestra["msl"][-1]
    return rugiada, surface_pressure_from_msl(
        finestra["msl"][-1], geopotenziale, temperatura
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--start", type=int, default=None,
        help="Slot iniziale della finestra di input; senza, si usa la piu' recente.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Percorso del PDF; senza, si scrive nella cartella del fold.",
    )
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="Usa le probabilita' grezze della rete, senza correggerne la scala.",
    )
    parser.add_argument(
        "--no-documentation", action="store_true",
        help="Solo mappe e andamenti, senza le pagine su modello, pipeline e lettura.",
    )
    parser.add_argument(
        "--saturated", action="store_true",
        help="Calore latente con aria satura, senza persistere la rugiada osservata.",
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
    print(f"fold {args.fold}, finestra che inizia allo slot {inizio}")
    print(f"ultimo istante osservato: {previsione.init_time}")
    print(f"dominio: {previsione.t2m_mean.shape[1]} x {previsione.t2m_mean.shape[2]}")
    if calibratore is None:
        print("probabilita' non calibrate: eseguire prima scripts/evaluate_model.py")

    rugiada, pressione = (None, None)
    if not args.saturated:
        rugiada, pressione = humidity_fields(reader, inizio, config.windows.input_slots)
        if rugiada is None:
            print("rugiada osservata non disponibile: calore latente con aria satura")

    destinazione = args.output
    if destinazione is None:
        cartella = fold_dir(config, args.fold)
        cartella.mkdir(parents=True, exist_ok=True)
        destinazione = report_path(cartella, previsione)

    percorso = write_report(
        previsione,
        destinazione,
        fold=args.fold,
        n_parameters=network.n_parameters,
        dewpoint_celsius=rugiada,
        pressure_pa=pressione,
        architecture=config.model.architecture,
        metrics=metriche(config, args.fold, "test"),
        documentation=not args.no_documentation,
    )
    print(f"\nreport scritto: {percorso} ({percorso.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
