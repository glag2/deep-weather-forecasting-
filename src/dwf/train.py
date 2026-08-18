"""Addestramento di un fold della validazione a finestra mobile.

Ogni fold e' un esperimento indipendente: le statistiche di normalizzazione si
calcolano **solo** sui suoi slot di train, il modello riparte da zero e il checkpoint
migliore e' scelto sulla sua validazione. Riusare statistiche o pesi fra fold farebbe
filtrare informazione dal futuro e le metriche non misurerebbero piu' la capacita' di
prevedere, ma la memoria dei dati gia' visti.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from dwf.data.dataset import (
    KEY_FEATURES,
    WeatherWindowDataset,
    WindowBatchSampler,
    build_reader,
    data_fingerprint,
    sample_starts,
    split_baselines,
)
from dwf.data.features import InputLayout, NormStats, SlotReader, compute_norm_stats
from dwf.models.global_network import GlobalContextNet, GlobalNetworkSpec
from dwf.models.heads import OutputLayout
from dwf.models.losses import CompositeLoss
from dwf.models.network import DeepWeatherNet, NetworkSpec
from dwf.persistence import PersistenceError, load_model, save_model
from dwf.tables import NORM_STATS, write_table

if TYPE_CHECKING:  # pragma: no cover - solo per i tipi
    from dwf.config import Config

HISTORY_NAME = "history.json"


class TrainingError(RuntimeError):
    """Errore nella preparazione o nell'esecuzione dell'addestramento."""


@dataclass
class EpochRecord:
    """Esito di un'epoca, per la cronologia salvata accanto al modello."""

    epoch: int
    train_loss: float
    val_loss: float
    seconds: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class FoldResult:
    fold: int
    best_epoch: int
    best_val_loss: float
    checkpoint: Path
    history: list[EpochRecord]


def fold_dir(config: Config, fold: int) -> Path:
    return config.fold_dir(fold)


def build_network(
    config: Config, layout: OutputLayout, in_channels: int
) -> DeepWeatherNet | GlobalContextNet:
    """Costruisce l'architettura scelta in configurazione.

    Le due architetture non sono una l'evoluzione dell'altra: ricevono gli stessi
    canali, producono lo stesso layout di uscita e si addestrano con la stessa perdita,
    quindi sono confrontabili. La scelta sta in `model.architecture`.
    """
    if config.model.architecture == "global":
        return GlobalContextNet(
            GlobalNetworkSpec(
                in_channels=in_channels,
                base_channels=config.model.base_channels,
                patch=config.model.patch,
                embed_channels=config.model.embed_channels,
                blocks=config.model.global_blocks,
                heads=config.model.heads,
                dropout=config.model.dropout,
            ),
            layout,
        )
    return DeepWeatherNet(
        NetworkSpec(
            in_channels=in_channels,
            base_channels=config.model.base_channels,
            depth=config.model.depth,
            blocks_per_level=config.model.blocks_per_level,
            dropout=config.model.dropout,
            variant=config.model.variant,
        ),
        layout,
    )


def training_slots(starts: list[int], total_slots: int) -> list[int]:
    """Slot effettivamente toccati dalle finestre di train, senza duplicati.

    Le statistiche vanno calcolate su questi e non su tutto il blocco di train: gli
    slot esclusi dalle finestre ammesse non entrano mai nel modello.
    """
    tocchi = {inizio + passo for inizio in starts for passo in range(total_slots)}
    return sorted(tocchi)


def compute_fold_stats(
    config: Config, layout: InputLayout, starts: list[int]
) -> NormStats:
    """Statistiche di normalizzazione sugli slot di train del fold."""
    store = xr.open_zarr(config.zarr_path, consolidated=True)
    arrays: dict[str, np.ndarray] = {
        nome: store[nome] for nome in layout.dynamic_variables
    }
    if layout.static_variables:
        statico = xr.open_zarr(config.static_path, consolidated=True)
        for nome in layout.static_variables:
            arrays[nome] = np.asarray(statico[nome].values, dtype=np.float32)

    totale = config.windows.input_slots + config.windows.output_slots
    slot = training_slots(starts, totale)
    return compute_norm_stats(
        SlotReader(arrays), layout.normalized_variables, slot, split_name="train"
    )


def make_loader(
    dataset: WeatherWindowDataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    max_batches: int | None,
    num_workers: int,
) -> DataLoader:
    sampler = WindowBatchSampler(
        n_windows=len(dataset.starts),
        crops_per_window=dataset.crops_per_window,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        max_batches=max_batches,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=torch.utils.data.default_collate,
    )


def run_epoch(
    network: DeepWeatherNet,
    loader: DataLoader,
    criterion: CompositeLoss,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float | None,
    layout: OutputLayout,
) -> tuple[float, dict[str, float]]:
    """Una passata completa; se `optimizer` e' None esegue solo la valutazione."""
    allena = optimizer is not None
    network.train(allena)

    somma = 0.0
    conteggio = 0
    componenti: dict[str, float] = {}

    with torch.set_grad_enabled(allena):
        for batch in loader:
            caratteristiche = batch[KEY_FEATURES]
            bersagli, riferimenti = split_baselines(batch)
            previsione = layout.apply_anchor(network(caratteristiche), riferimenti)
            perdita = criterion(previsione, bersagli)

            if allena:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                perdita.total.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
                optimizer.step()

            somma += float(perdita.total.detach())
            conteggio += 1
            for nome, valore in perdita.components.items():
                componenti[nome] = componenti.get(nome, 0.0) + float(valore.detach())

    if conteggio == 0:
        raise TrainingError("Nessun batch prodotto: controllare le finestre ammesse")
    medie = {nome: valore / conteggio for nome, valore in componenti.items()}
    return somma / conteggio, medie


def train_fold(
    config: Config,
    fold: int,
    *,
    epochs: int | None = None,
    verbose: bool = True,
) -> FoldResult:
    """Addestra un fold e salva il checkpoint migliore secondo la validazione."""
    torch.manual_seed(config.training.seed + fold)
    np.random.seed(config.training.seed + fold)

    input_layout = InputLayout.from_config(config)
    output_layout = OutputLayout.from_targets(config.targets, config.windows.output_slots)

    starts_train = sample_starts(config, fold, "train")
    starts_val = sample_starts(config, fold, "val")
    if not starts_train:
        raise TrainingError(
            f"Fold {fold}: nessuna finestra di train ammessa. Ingerire piu' mesi."
        )
    if not starts_val:
        raise TrainingError(
            f"Fold {fold}: nessuna finestra di validazione ammessa. Ingerire piu' mesi."
        )

    stats = compute_fold_stats(config, input_layout, starts_train)
    reader = build_reader(config, input_layout)

    crop = config.training.crop_size
    # Piu' ritagli per finestra ammortizzano la lettura, che e' il costo dominante
    # dell'accesso ai dati; il batch li consuma tutti insieme.
    crops = max(config.training.batch_size, 1)

    dataset_train = WeatherWindowDataset(
        config, input_layout, stats, starts_train, reader,
        crop_size=crop, crops_per_window=crops, seed=config.training.seed + fold,
    )
    dataset_val = WeatherWindowDataset(
        config, input_layout, stats, starts_val, reader,
        crop_size=crop, crops_per_window=crops, seed=config.training.seed + 1000 + fold,
    )

    batch_per_epoca = max(1, config.training.samples_per_epoch // config.training.batch_size)
    loader_train = make_loader(
        dataset_train, config.training.batch_size, shuffle=True,
        seed=config.training.seed + fold, max_batches=batch_per_epoca,
        num_workers=config.training.num_workers,
    )
    loader_val = make_loader(
        dataset_val, config.training.batch_size, shuffle=False,
        seed=config.training.seed, max_batches=max(1, batch_per_epoca // 4),
        num_workers=config.training.num_workers,
    )

    network = build_network(config, output_layout, input_layout.n_channels)
    criterion = CompositeLoss(output_layout, config.training.loss_weights)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    destinazione = fold_dir(config, fold)
    destinazione.mkdir(parents=True, exist_ok=True)
    write_table(stats.to_table(), NORM_STATS, destinazione)

    n_epoche = epochs if epochs is not None else config.training.epochs
    cronologia: list[EpochRecord] = []
    migliore = float("inf")
    epoca_migliore = -1
    checkpoint = destinazione
    # Calcolata una volta sola: descrive i dati di questa corsa, che non cambiano
    # mentre la corsa e' in atto.
    impronta_dati = data_fingerprint(config, fold)

    if verbose:
        print(
            f"fold {fold}: {len(starts_train)} finestre di train, "
            f"{len(starts_val)} di validazione, {input_layout.n_channels} canali, "
            f"{network.n_parameters:,} parametri"
        )

    for epoca in range(n_epoche):
        avvio = time.perf_counter()
        perdita_train, componenti = run_epoch(
            network, loader_train, criterion, optimizer,
            config.training.grad_clip_norm, output_layout,
        )
        perdita_val, _ = run_epoch(
            network, loader_val, criterion, None, None, output_layout
        )
        durata = time.perf_counter() - avvio

        cronologia.append(
            EpochRecord(
                epoch=epoca,
                train_loss=perdita_train,
                val_loss=perdita_val,
                seconds=durata,
                components=componenti,
            )
        )

        # Una perdita non finita non migliora mai il minimo corrente, quindi senza
        # questo controllo l'addestramento arriverebbe in fondo, non salverebbe alcun
        # checkpoint e non direbbe perche': il difetto si manifesterebbe molto piu'
        # tardi, come un file mancante durante la valutazione.
        if not math.isfinite(perdita_val) or not math.isfinite(perdita_train):
            raise TrainingError(
                f"Perdita non finita all'epoca {epoca} "
                f"(train {perdita_train}, validazione {perdita_val}): "
                "l'addestramento non puo' proseguire. Cause tipiche sono slot mancanti "
                "dentro la finestra, statistiche di normalizzazione degeneri o un passo "
                "di apprendimento troppo grande."
            )

        if perdita_val < migliore:
            migliore = perdita_val
            epoca_migliore = epoca
            save_model(
                checkpoint,
                {
                    nome: valori.detach().cpu().numpy()
                    for nome, valori in network.state_dict().items()
                },
                {
                    "in_channels": input_layout.n_channels,
                    "fold": fold,
                    "epoch": epoca,
                    "val_loss": perdita_val,
                    "channels": input_layout.describe(),
                    "outputs": output_layout.describe(),
                    "data": impronta_dati,
                },
            )

        if verbose:
            marcatore = " *" if epoca == epoca_migliore else ""
            print(
                f"  epoca {epoca:3d}  train {perdita_train:8.4f}  "
                f"val {perdita_val:8.4f}  {durata:6.1f} s{marcatore}"
            )

    (destinazione / HISTORY_NAME).write_text(
        json.dumps([asdict(record) for record in cronologia], indent=2),
        encoding="utf-8",
    )

    return FoldResult(
        fold=fold,
        best_epoch=epoca_migliore,
        best_val_loss=migliore,
        checkpoint=checkpoint,
        history=cronologia,
    )


def load_checkpoint(
    config: Config, fold: int
) -> tuple[DeepWeatherNet, NormStats, InputLayout, OutputLayout]:
    """Ricostruisce rete e normalizzazione salvate per un fold."""
    from dwf.tables import read_table

    destinazione = fold_dir(config, fold)
    input_layout = InputLayout.from_config(config)
    output_layout = OutputLayout.from_targets(config.targets, config.windows.output_slots)
    try:
        pesi, metadati = load_model(destinazione)
    except PersistenceError as errore:
        raise TrainingError(str(errore)) from errore

    if metadati["in_channels"] != input_layout.n_channels:
        raise TrainingError(
            f"Il checkpoint attende {metadati['in_channels']} canali, la configurazione "
            f"ne produce {input_layout.n_channels}: configurazione e modello non "
            f"corrispondono"
        )

    network = build_network(config, output_layout, input_layout.n_channels)
    network.load_state_dict(
        {nome: torch.from_numpy(valori) for nome, valori in pesi.items()}
    )
    network.eval()

    stats = NormStats.from_table(read_table(NORM_STATS, destinazione))
    return network, stats, input_layout, output_layout


__all__ = [

    "EpochRecord",
    "FoldResult",
    "TrainingError",
    "build_network",
    "compute_fold_stats",
    "fold_dir",
    "load_checkpoint",
    "run_epoch",
    "train_fold",
    "training_slots",
]
