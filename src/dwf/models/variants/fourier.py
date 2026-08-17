"""Variante a operatore di Fourier: convoluzione spettrale sui modi bassi.

Una convoluzione 3x3 vede tre celle: per collegare due punti distanti mille chilometri
servono decine di strati. La moltiplicazione nello spazio di Fourier equivale a una
convoluzione con nucleo grande quanto il dominio, quindi collega **ogni punto con ogni
altro in una sola operazione**, e costa O(N log N) invece di O(N^2).

Tenere solo i modi bassi non e' una semplificazione ma la scelta di fondo: i modi alti
sono il rumore di piccola scala, che nessun modello puo' prevedere a tre giorni, e
scartarli agisce da regolarizzatore. Il numero di parametri dipende dai modi tenuti,
non dalla dimensione del dominio, quindi lo stesso blocco funziona su un ritaglio di
96 celle e sulla griglia intera.

Il ramo convolutivo 1x1 affiancato a quello spettrale e' l'accorgimento standard di
questa famiglia: da solo lo spettro troncato non saprebbe rappresentare discontinuita'
nette come la linea di costa.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from dwf.models.blocks import group_count

# Modi conservati per asse. Otto modi descrivono le strutture piu' grandi di un ottavo
# del dominio, oltre i mille chilometri: la scala sinottica che governa tre giorni di
# previsione. Il dettaglio piu' fine resta al ramo convolutivo.
#
# Il valore non e' scelto per gusto ma per un vincolo misurato: il costo in parametri
# cresce con il **quadrato** dei modi, e con sedici modi la rete arrivava a 228 milioni
# di parametri, in larga parte inutilizzati perche' al collo di bottiglia la griglia si
# riduce a dodici celle e i modi oltre il sesto vengono troncati a ogni passata.
DEFAULT_MODES = 8

# Larghezza del ramo spettrale. I pesi spettrali costano in_canali x out_canali x
# modi^2, quindi applicarli direttamente ai 384 canali del collo di bottiglia sarebbe
# sproporzionato: si proietta su uno spazio piu' stretto, si opera li' e si torna
# indietro. E' la stessa logica del collo di bottiglia dei blocchi residui profondi.
DEFAULT_WIDTH = 64


class SpectralConv2d(nn.Module):
    """Moltiplicazione per pesi complessi appresi sui modi bassi della trasformata."""

    def __init__(
        self, in_channels: int, out_channels: int, modes: int = DEFAULT_MODES,
        width: int = DEFAULT_WIDTH,
    ) -> None:
        super().__init__()
        if modes < 1:
            raise ValueError(f"Servono almeno un modo per asse, ricevuto {modes}")
        larghezza = min(width, in_channels, out_channels)
        self.enter = (
            nn.Conv2d(in_channels, larghezza, kernel_size=1)
            if in_channels != larghezza
            else nn.Identity()
        )
        self.leave = (
            nn.Conv2d(larghezza, out_channels, kernel_size=1)
            if out_channels != larghezza
            else nn.Identity()
        )
        in_channels = out_channels = larghezza
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # Due blocchi di pesi: i modi verticali positivi e quelli negativi. La
        # trasformata reale comprime il solo asse orizzontale, quindi il verticale
        # conserva entrambi i segni e vanno trattati separatamente.
        scala = 1.0 / (in_channels * out_channels)
        forma = (2, in_channels, out_channels, modes, modes)
        self.weight = nn.Parameter(scala * torch.randn(*forma, dtype=torch.cfloat))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.enter(features)
        altezza, larghezza = features.shape[-2:]
        spettro = torch.fft.rfft2(features, norm="ortho")
        # Su ritagli piccoli i modi disponibili possono essere meno di quelli appresi:
        # troncare qui rende il blocco indipendente dalla dimensione dell'ingresso.
        modi_v = min(self.modes, altezza // 2)
        modi_o = min(self.modes, larghezza // 2 + 1)

        uscita = torch.zeros(
            features.shape[0],
            self.out_channels,
            altezza,
            larghezza // 2 + 1,
            dtype=torch.cfloat,
            device=features.device,
        )
        uscita[..., :modi_v, :modi_o] = torch.einsum(
            "bixy,ioxy->boxy",
            spettro[..., :modi_v, :modi_o],
            self.weight[0, :, :, :modi_v, :modi_o],
        )
        uscita[..., -modi_v:, :modi_o] = torch.einsum(
            "bixy,ioxy->boxy",
            spettro[..., -modi_v:, :modi_o],
            self.weight[1, :, :, :modi_v, :modi_o],
        )
        return self.leave(torch.fft.irfft2(uscita, s=(altezza, larghezza), norm="ortho"))


class FourierBlock(nn.Module):
    """Blocco residuo con ramo spettrale e ramo locale."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0,
        modes: int = DEFAULT_MODES,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(group_count(in_channels), in_channels)
        self.spectral = SpectralConv2d(in_channels, out_channels, modes)
        self.local = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm_out = nn.GroupNorm(group_count(out_channels), out_channels)
        self.project = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalizzato = functional.gelu(self.norm(features))
        misto = self.spectral(normalizzato) + self.local(normalizzato)
        misto = self.dropout(misto)
        misto = self.project(functional.gelu(self.norm_out(misto)))
        return misto + self.shortcut(features)


__all__ = ["DEFAULT_MODES", "DEFAULT_WIDTH", "FourierBlock", "SpectralConv2d"]
