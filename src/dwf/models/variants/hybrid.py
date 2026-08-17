"""Variante ibrida: ramo convolutivo e ramo spettrale nello stesso blocco.

Le due famiglie sbagliano in modi diversi. La convoluzione vede bene il dettaglio e la
discontinuita' netta, come la linea di costa o il bordo di una catena montuosa, ma per
collegare punti lontani deve impilare molti strati. L'operatore spettrale collega tutto
con tutto in un colpo solo, ma tenendo i soli modi bassi non sa rappresentare un
gradino.

Sommare i due rami costa poco piu' del piu' caro dei due e li lascia specializzare: se
l'ipotesi e' giusta, l'ibrido deve battere entrambi i puri. Se non lo fa, e' una
informazione altrettanto utile, perche' significa che il collo di bottiglia non e' la
portata spaziale del blocco.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from dwf.models.blocks import conv3x3, group_count
from dwf.models.variants.fourier import DEFAULT_MODES, SpectralConv2d


class HybridBlock(nn.Module):
    """Blocco residuo con ramo locale 3x3 e ramo globale spettrale."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0,
        modes: int = DEFAULT_MODES,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(group_count(in_channels), in_channels)
        self.local = conv3x3(in_channels, out_channels)
        self.spectral = SpectralConv2d(in_channels, out_channels, modes)
        self.norm_out = nn.GroupNorm(group_count(out_channels), out_channels)
        self.project = conv3x3(out_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        # Peso appreso del ramo globale, inizializzato basso: il blocco parte come una
        # convoluzione e apre il ramo spettrale solo se serve davvero.
        self.spectral_gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalizzato = functional.gelu(self.norm(features))
        misto = self.local(normalizzato) + self.spectral_gain * self.spectral(normalizzato)
        misto = self.dropout(misto)
        return self.project(functional.gelu(self.norm_out(misto))) + self.shortcut(features)


__all__ = ["HybridBlock"]
