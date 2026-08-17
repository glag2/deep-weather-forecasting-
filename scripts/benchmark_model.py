"""Misura il costo su CPU della rete, per decidere risoluzione e dimensioni.

Il training a piena risoluzione su CPU e' l'ipotesi piu' rischiosa del progetto:
questo script la misura invece di stimarla. Riporta il tempo di un passo di
training su crop e il tempo di una inferenza sul dominio intero.

Uso:
    python scripts/benchmark_model.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dwf.config import Config
from dwf.models.heads import OutputLayout
from dwf.models.network import DeepWeatherNet, NetworkSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def estimate_input_channels(config: Config) -> int:
    """Conta i canali di input previsti dal piano di feature engineering."""
    n_dynamic = len(config.variables.dynamic_short_names)
    input_slots = config.windows.input_slots

    channels = n_dynamic * input_slots
    channels += len(config.features.tendency_lags) * n_dynamic
    if config.features.include_wind_speed:
        channels += input_slots
    if config.features.include_static:
        channels += len(config.variables.static_short_names)
    if config.features.include_latitude_encoding:
        channels += 2
    if config.features.include_time_encoding:
        channels += 4
    return channels


def time_training_step(
    network: DeepWeatherNet, batch_size: int, crop: int, repeats: int
) -> float:
    """Secondi medi per un passo completo forward + backward + aggiornamento."""
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    features = torch.randn(batch_size, network.spec.in_channels, crop, crop)
    network.train()

    # Un passo a vuoto: la prima chiamata paga allocazioni e scelta dei kernel.
    optimizer.zero_grad(set_to_none=True)
    network(features).square().mean().backward()
    optimizer.step()

    started = time.perf_counter()
    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        network(features).square().mean().backward()
        optimizer.step()
    return (time.perf_counter() - started) / repeats


def time_full_domain_inference(network: DeepWeatherNet, height: int, width: int) -> float:
    """Secondi per una previsione sull'intero dominio."""
    features = torch.randn(1, network.spec.in_channels, height, width)
    network.eval()
    with torch.no_grad():
        network(features)  # riscaldamento
        started = time.perf_counter()
        network(features)
        return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--skip-full-domain",
        action="store_true",
        help="Salta l'inferenza a piena risoluzione, che puo' richiedere molta memoria.",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    layout = OutputLayout.from_targets(config.targets, config.windows.output_slots)
    in_channels = estimate_input_channels(config)
    crop = config.training.crop_size or 96

    network = DeepWeatherNet(
        NetworkSpec(
            in_channels=in_channels,
            base_channels=config.model.base_channels,
            depth=config.model.depth,
            blocks_per_level=config.model.blocks_per_level,
            dropout=config.model.dropout,
        ),
        layout,
    )

    height, width = config.region.n_lat, config.region.n_lon
    n_train_samples = int(2190 * config.split.train_fraction)

    print(f"thread torch          : {torch.get_num_threads()}")
    print(f"canali di input       : {in_channels}")
    print(f"canali di uscita      : {layout.total_channels}")
    print(f"parametri             : {network.n_parameters:,}")
    print(f"dominio               : {height} x {width}")
    print(f"crop di training      : {crop} x {crop}")
    print()

    step_seconds = time_training_step(
        network, config.training.batch_size, crop, args.repeats
    )
    per_sample = step_seconds / config.training.batch_size
    epoch_seconds = per_sample * config.training.samples_per_epoch

    print(f"passo di training     : {step_seconds:.2f} s (batch {config.training.batch_size})")
    print(f"per campione          : {per_sample:.2f} s")
    print(
        f"epoca configurata     : {epoch_seconds / 60:.1f} min "
        f"({config.training.samples_per_epoch} campioni)"
    )
    print(
        f"training completo     : {epoch_seconds * config.training.epochs / 3600:.1f} h "
        f"({config.training.epochs} epoche)"
    )
    print(
        f"epoca su tutto il train: {per_sample * n_train_samples / 3600:.1f} h "
        f"({n_train_samples} campioni)"
    )

    if not args.skip_full_domain:
        print()
        inference_seconds = time_full_domain_inference(network, height, width)
        print(f"inferenza dominio pieno: {inference_seconds:.2f} s")
        print(f"previsione a 3 giorni  : {inference_seconds:.2f} s (un solo passaggio)")


if __name__ == "__main__":
    main()
