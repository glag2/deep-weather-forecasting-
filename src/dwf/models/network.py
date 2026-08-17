"""Rete convoluzionale encoder-decoder per la previsione a 3 giorni.

Architettura scritta da zero, senza pesi preaddestrati.

Il tempo di input non e' un asse di convoluzione ma viene impilato sui canali: con
21 slot di input, 8 variabili e i canali derivati si arriva a oltre 200 canali in
ingresso, e una convoluzione 3D su CPU costerebbe un ordine di grandezza in piu'
senza un guadagno dimostrato a questa scala. Analogamente i 9 slot previsti sono
prodotti in un unico passaggio ("in bulk"), non autoregressivamente: evita
l'accumulo di errore tipico del rollout e permette alla rete di imporre coerenza
tra lead time diversi.

La rete e' interamente convoluzionale: si allena su crop 96 x 96 e si applica al
dominio intero 261 x 401 senza modifiche, previa estensione a multiplo di 2^depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dwf.models.blocks import (
    DecoderStage,
    EncoderStage,
    ResidualBlock,
    conv3x3,
    crop_padding,
    group_count,
    pad_to_multiple,
)
from dwf.models.heads import OutputLayout


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    """Iperparametri strutturali della rete."""

    in_channels: int
    base_channels: int = 48
    depth: int = 3
    blocks_per_level: int = 2
    dropout: float = 0.0
    max_channels: int = 384

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError(f"in_channels deve essere positivo: {self.in_channels}")
        if self.base_channels < 1:
            raise ValueError(f"base_channels deve essere positivo: {self.base_channels}")
        if not 1 <= self.depth <= 5:
            raise ValueError(f"depth fuori range 1..5: {self.depth}")
        if self.blocks_per_level < 1:
            raise ValueError(f"blocks_per_level deve essere positivo: {self.blocks_per_level}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout fuori range [0, 1): {self.dropout}")

    def channels_at(self, level: int) -> int:
        return min(self.base_channels * 2**level, self.max_channels)

    @property
    def size_multiple(self) -> int:
        """Multiplo richiesto per le dimensioni spaziali, dato il numero di riduzioni."""
        return 2**self.depth


class DeepWeatherNet(nn.Module):
    """Encoder-decoder residuo con teste probabilistiche multiple.

    L'uscita e' un unico tensore ``(B, C, H, W)``; la mappatura dei canali su
    (variabile, componente, lead time) e' definita da :class:`OutputLayout`.
    """

    def __init__(self, spec: NetworkSpec, layout: OutputLayout) -> None:
        super().__init__()
        self.spec = spec
        self.layout = layout

        self.stem = conv3x3(spec.in_channels, spec.base_channels)

        self.encoders = nn.ModuleList()
        channels = spec.base_channels
        skip_channels: list[int] = []
        for level in range(spec.depth):
            out_channels = spec.channels_at(level)
            self.encoders.append(
                EncoderStage(channels, out_channels, spec.blocks_per_level, spec.dropout)
            )
            skip_channels.append(out_channels)
            channels = out_channels

        bottleneck_channels = spec.channels_at(spec.depth)
        self.bottleneck = nn.Sequential(
            *[
                ResidualBlock(
                    channels if index == 0 else bottleneck_channels,
                    bottleneck_channels,
                    spec.dropout,
                )
                for index in range(spec.blocks_per_level)
            ]
        )
        channels = bottleneck_channels

        self.decoders = nn.ModuleList()
        for level in reversed(range(spec.depth)):
            out_channels = skip_channels[level]
            self.decoders.append(
                DecoderStage(
                    channels, out_channels, out_channels, spec.blocks_per_level, spec.dropout
                )
            )
            channels = out_channels

        self.output_norm = nn.GroupNorm(group_count(channels), channels)
        # Convoluzione 1x1 finale: ogni canale di uscita e' una combinazione locale
        # delle feature, senza ulteriore mescolamento spaziale.
        self.output_conv = nn.Conv2d(channels, layout.total_channels, kernel_size=1)
        self._initialize_output_bias()

    def _initialize_output_bias(self) -> None:
        """Azzera i pesi finali per partire da una previsione neutra.

        Con pesi finali nulli l'uscita iniziale e' il solo bias: le componenti sono
        quindi costanti e prevedibili, e la rete parte da uno stato equivalente a una
        climatologia piatta invece che da rumore. La log-varianza parte da 0
        (deviazione standard unitaria in spazio normalizzato).
        """
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    @property
    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                f"Atteso un input (B, C, H, W), ricevuta forma {tuple(features.shape)}"
            )
        if features.shape[1] != self.spec.in_channels:
            raise ValueError(
                f"L'input ha {features.shape[1]} canali, la rete ne attende "
                f"{self.spec.in_channels}"
            )

        padded, padding = pad_to_multiple(features, self.spec.size_multiple)

        out = self.stem(padded)
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            out, skip = encoder(out)
            skips.append(skip)

        out = self.bottleneck(out)

        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            out = decoder(out, skip)

        out = self.output_conv(nn.functional.gelu(self.output_norm(out)))
        return crop_padding(out, padding)


def build_network(
    in_channels: int, layout: OutputLayout, model_config: object
) -> DeepWeatherNet:
    """Istanzia la rete dai parametri di ``ModelConfig``.

    Accetta un oggetto qualunque con gli attributi attesi per non far dipendere il
    package dei modelli dal modulo di configurazione.
    """
    spec = NetworkSpec(
        in_channels=in_channels,
        base_channels=model_config.base_channels,
        depth=model_config.depth,
        blocks_per_level=model_config.blocks_per_level,
        dropout=model_config.dropout,
    )
    return DeepWeatherNet(spec, layout)
