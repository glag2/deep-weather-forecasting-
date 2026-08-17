"""Variante ad autoattenzione su finestre locali.

L'attenzione globale su una griglia di 261 per 401 celle richiederebbe una matrice di
circa 10^10 elementi per testa: impraticabile. Limitandola a finestre di 8 per 8 il
costo torna lineare nel numero di celle, e la comunicazione fra finestre e' affidata
agli stadi di sottocampionamento della U, dove una finestra di 8 celle copre un'area
doppia a ogni livello.

Cosa aggiunge rispetto alla convoluzione: i pesi non sono fissi ma **dipendono dal
contenuto**. Una convoluzione applica lo stesso nucleo su mare aperto e su un fronte;
l'attenzione puo' pesare diversamente i vicini a seconda di cio' che vede, il che e'
plausibilmente utile dove il campo cambia regime.

Il bias di posizione relativa non e' un dettaglio: senza, l'attenzione sarebbe
invariante a permutazioni dentro la finestra e perderebbe del tutto la nozione di
"sopra" e "a sinistra", che in meteorologia e' l'informazione principale.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from dwf.models.blocks import group_count

DEFAULT_WINDOW = 8
DEFAULT_HEADS = 4


def _pad_to_window(features: torch.Tensor, window: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Riempie in basso e a destra fino a un multiplo della finestra."""
    altezza, larghezza = features.shape[-2:]
    giu = (window - altezza % window) % window
    destra = (window - larghezza % window) % window
    if giu or destra:
        features = functional.pad(features, (0, destra, 0, giu), mode="replicate")
    return features, (giu, destra)


class WindowAttention(nn.Module):
    """Autoattenzione a piu' teste dentro finestre disgiunte."""

    def __init__(
        self, channels: int, window: int = DEFAULT_WINDOW, heads: int = DEFAULT_HEADS
    ) -> None:
        super().__init__()
        if channels % heads != 0:
            heads = 1
        self.window = window
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.project = nn.Conv2d(channels, channels, kernel_size=1)

        # Una voce per ogni scarto relativo possibile dentro la finestra.
        self.relative_bias = nn.Parameter(
            torch.zeros(heads, (2 * window - 1) * (2 * window - 1))
        )
        self.register_buffer("relative_index", self._relative_index(window), persistent=False)

    @staticmethod
    def _relative_index(window: int) -> torch.Tensor:
        coordinate = torch.stack(
            torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")
        ).flatten(1)
        scarti = coordinate[:, :, None] - coordinate[:, None, :]
        scarti = scarti.permute(1, 2, 0) + (window - 1)
        return (scarti[..., 0] * (2 * window - 1) + scarti[..., 1]).reshape(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        riempito, (giu, destra) = _pad_to_window(features, self.window)
        lotti, canali, altezza, larghezza = riempito.shape
        finestra = self.window
        n_v, n_o = altezza // finestra, larghezza // finestra

        qkv = self.qkv(riempito)
        # (B, 3, teste, finestre, celle nella finestra, canali per testa)
        qkv = qkv.reshape(lotti, 3, self.heads, canali // self.heads, n_v, finestra, n_o, finestra)
        qkv = qkv.permute(0, 1, 2, 4, 6, 5, 7, 3).reshape(
            lotti, 3, self.heads, n_v * n_o, finestra * finestra, canali // self.heads
        )
        query, key, value = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        punteggi = (query @ key.transpose(-2, -1)) * self.scale
        bias = self.relative_bias[:, self.relative_index].reshape(
            1, self.heads, 1, finestra * finestra, finestra * finestra
        )
        pesi = torch.softmax(punteggi + bias, dim=-1)
        uscita = pesi @ value

        uscita = uscita.reshape(
            lotti, self.heads, n_v, n_o, finestra, finestra, canali // self.heads
        )
        uscita = uscita.permute(0, 1, 6, 2, 4, 3, 5).reshape(lotti, canali, altezza, larghezza)
        uscita = self.project(uscita)
        if giu or destra:
            uscita = uscita[..., : altezza - giu, : larghezza - destra]
        return uscita


class WindowAttentionBlock(nn.Module):
    """Blocco residuo: attenzione a finestre seguita da una rete punto a punto."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0,
        window: int = DEFAULT_WINDOW, heads: int = DEFAULT_HEADS,
    ) -> None:
        super().__init__()
        self.entry = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm1 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.attention = WindowAttention(out_channels, window, heads)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        # Una convoluzione 3x3 nella parte punto a punto ricuce i bordi fra finestre
        # adiacenti, che l'attenzione da sola lascerebbe scollegati.
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1,
                      padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base = self.entry(features)
        base = base + self.dropout(self.attention(self.norm1(base)))
        return base + self.mlp(self.norm2(base))


__all__ = ["DEFAULT_HEADS", "DEFAULT_WINDOW", "WindowAttention", "WindowAttentionBlock"]
