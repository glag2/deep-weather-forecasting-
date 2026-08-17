"""Blocchi elementari della rete.

Due scelte non ovvie, motivate dal dominio:

- **padding riflesso** invece di zeri. Un padding a zero ai bordi del dominio
  inietterebbe una temperatura di 0 K e una pressione di 0 Pa appena fuori
  dall'area, creando gradienti artificiali lungo tutto il perimetro. La riflessione
  e' la condizione al contorno meno dannosa senza dati esterni al ritaglio.
- **GroupNorm** invece di BatchNorm. Il training su CPU usa batch molto piccoli
  (4 campioni), regime in cui le statistiche di batch sono troppo rumorose.
  GroupNorm normalizza dentro il singolo campione e non dipende dal batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


def group_count(channels: int, preferred: int = 8) -> int:
    """Numero di gruppi per GroupNorm che divida esattamente i canali."""
    if channels < 1:
        raise ValueError(f"channels deve essere positivo: {channels}")
    for candidate in range(min(preferred, channels), 0, -1):
        if channels % candidate == 0:
            return candidate
    return 1  # pragma: no cover - irraggiungibile, 1 divide sempre


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    """Convoluzione 3x3 che preserva la risoluzione, con padding riflesso."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        padding_mode="reflect",
    )


class ResidualBlock(nn.Module):
    """Blocco residuo pre-attivato a due convoluzioni."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        # La scorciatoia serve solo quando il blocco cambia numero di canali.
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(features)
        out = self.conv1(functional.gelu(self.norm1(features)))
        out = self.dropout(out)
        out = self.conv2(functional.gelu(self.norm2(out)))
        return out + residual


class EncoderStage(nn.Module):
    """Blocchi residui seguiti da dimezzamento della risoluzione."""

    def __init__(
        self, in_channels: int, out_channels: int, n_blocks: int, dropout: float
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        channels = in_channels
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(channels, out_channels, dropout))
            channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode="reflect"
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Restituisce (uscita ridotta, skip a piena risoluzione dello stadio)."""
        skip = self.blocks(features)
        return self.downsample(skip), skip


class DecoderStage(nn.Module):
    """Upsampling bilineare, fusione con lo skip e blocchi residui.

    L'upsampling e' interpolazione seguita da convoluzione, non `ConvTranspose2d`:
    quest'ultima produce artefatti a scacchiera che su campi meteo verrebbero letti
    come struttura fisica inesistente.
    """

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, n_blocks: int, dropout: float
    ) -> None:
        super().__init__()
        self.reduce = conv3x3(in_channels, out_channels)
        blocks: list[nn.Module] = []
        channels = out_channels + skip_channels
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(channels, out_channels, dropout))
            channels = out_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = functional.interpolate(
            features, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        merged = torch.cat([self.reduce(upsampled), skip], dim=1)
        return self.blocks(merged)


def pad_to_multiple(features: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Estende in basso e a destra fino a un multiplo di ``multiple``, per riflessione.

    La griglia reale e' 261 x 401, non divisibile per 8: senza questa estensione la
    rete non potrebbe essere applicata al dominio intero pur essendo stata allenata
    su crop. Restituisce anche il padding applicato, per ritagliarlo dopo.
    """
    if multiple < 1:
        raise ValueError(f"multiple deve essere positivo: {multiple}")
    height, width = features.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if pad_height == 0 and pad_width == 0:
        return features, (0, 0)
    # `reflect` richiede padding strettamente minore della dimensione corrispondente.
    mode = "reflect" if pad_height < height and pad_width < width else "replicate"
    padded = functional.pad(features, (0, pad_width, 0, pad_height), mode=mode)
    return padded, (pad_height, pad_width)


def crop_padding(features: torch.Tensor, padding: tuple[int, int]) -> torch.Tensor:
    """Rimuove il padding aggiunto da :func:`pad_to_multiple`."""
    pad_height, pad_width = padding
    if pad_height == 0 and pad_width == 0:
        return features
    height = features.shape[-2] - pad_height
    width = features.shape[-1] - pad_width
    return features[..., :height, :width]
