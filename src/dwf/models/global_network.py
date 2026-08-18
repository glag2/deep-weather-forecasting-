"""Rete rivale con nucleo globale e denso, alternativa allo scheletro a U.

Architettura scritta da zero, senza pesi preaddestrati.

**Perche' esiste.** Il campo recettivo *efficace* della rete convoluzionale e' stato
misurato sul modello addestrato derivando l'uscita in un punto rispetto all'ingresso:
il 50% dell'influenza su una previsione arriva da un raggio di appena 130 km, il 90% da
2189 km. Una struttura sinottica alle medie latitudini viaggia fra i 500 e i 1000 km al
giorno, quindi a tre giorni l'informazione che conta parte da 1500-3000 km di distanza.
Una convoluzione puo' in teoria arrivarci impilando strati, ma i pesi che ha imparato
dicono che di fatto decide guardando vicino. Questa rete mette il ragionamento
principale dove serve: su una griglia grossolana dove ogni posizione vede **tutte** le
altre in una sola operazione.

**Come tiene basso il costo.** L'attenzione globale su tutti i pixel sarebbe quadratica
in 261x401, cioe' impensabile. Qui l'attenzione lavora su token ottenuti riducendo di
un fattore ``patch`` per lato: il dominio intero diventa 33x51 token, e il costo torna
trattabile. Il dettaglio fine non passa dal nucleo globale ma da una scorciatoia
convolutiva a piena risoluzione, che e' il posto giusto per il dettaglio locale.

**Perche' resta applicabile al dominio intero.** Ci si addestra su ritagli 96x96 e si
prevede su 261x401, quindi nessun peso puo' dipendere dal *numero* di token.
L'attenzione non ne dipende per costruzione; la posizione e' iniettata con una
convoluzione depthwise invece che con una tabella di posizioni assolute, che sarebbe
legata a una dimensione fissa e renderebbe la rete inutilizzabile fuori dal ritaglio.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dwf.models.blocks import conv3x3, crop_padding, group_count, pad_to_multiple
from dwf.models.heads import OutputLayout


@dataclass(frozen=True, slots=True)
class GlobalNetworkSpec:
    """Iperparametri strutturali della rete a nucleo globale."""

    in_channels: int
    base_channels: int = 48
    # Riduzione per lato prima del nucleo globale. Il costo dell'attenzione scala con
    # il quadrato del numero di token, quindi con il quarto inverso di questo valore:
    # portarlo da 8 a 4 costa sedici volte tanto.
    patch: int = 8
    embed_channels: int = 192
    blocks: int = 4
    heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError(f"in_channels deve essere positivo: {self.in_channels}")
        if self.patch < 2 or self.patch & (self.patch - 1):
            raise ValueError(f"patch deve essere una potenza di due >= 2: {self.patch}")
        if self.embed_channels % self.heads:
            raise ValueError(
                f"embed_channels ({self.embed_channels}) deve essere divisibile per "
                f"heads ({self.heads})"
            )
        if self.blocks < 1:
            raise ValueError(f"blocks deve essere positivo: {self.blocks}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout fuori range [0, 1): {self.dropout}")

    @property
    def size_multiple(self) -> int:
        return self.patch


class GlobalBlock(nn.Module):
    """Attenzione su tutti i token piu' un percorso denso sui canali.

    L'attenzione decide *da dove* prendere informazione, il percorso denso decide *cosa*
    farne. Sono le due operazioni che una convoluzione non fa: la prima perche' i suoi
    pesi non dipendono dal contenuto, la seconda perche' i suoi canali si mescolano solo
    entro un intorno.
    """

    def __init__(self, channels: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        # Posizione iniettata da una convoluzione invece che da una tabella assoluta:
        # una tabella sarebbe legata al numero di token visto in addestramento, e la
        # rete non potrebbe piu' essere applicata al dominio intero.
        self.position = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm_attention = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.norm_dense = nn.LayerNorm(channels)
        nascosti = int(channels * mlp_ratio)
        self.dense = nn.Sequential(
            nn.Linear(channels, nascosti),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(nascosti, channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features + self.position(features)

        lotto, canali, altezza, larghezza = features.shape
        token = features.flatten(2).transpose(1, 2)

        normalizzati = self.norm_attention(token)
        attesi, _ = self.attention(
            normalizzati, normalizzati, normalizzati, need_weights=False
        )
        token = token + attesi
        token = token + self.dense(self.norm_dense(token))

        return token.transpose(1, 2).reshape(lotto, canali, altezza, larghezza)


class GlobalContextNet(nn.Module):
    """Scorciatoia convolutiva locale piu' nucleo globale su griglia grossolana.

    L'uscita e' un unico tensore ``(B, C, H, W)`` con lo stesso significato dei canali
    della rete a U: le due sono quindi confrontabili a parita' di tutto il resto.
    """

    def __init__(self, spec: GlobalNetworkSpec, layout: OutputLayout) -> None:
        super().__init__()
        self.spec = spec
        self.layout = layout

        self.stem = conv3x3(spec.in_channels, spec.base_channels)
        self.stem_norm = nn.GroupNorm(group_count(spec.base_channels), spec.base_channels)

        self.to_tokens = nn.Conv2d(
            spec.base_channels, spec.embed_channels, spec.patch, stride=spec.patch
        )
        self.blocks = nn.ModuleList(
            GlobalBlock(spec.embed_channels, spec.heads, spec.mlp_ratio, spec.dropout)
            for _ in range(spec.blocks)
        )
        self.token_norm = nn.LayerNorm(spec.embed_channels)
        self.from_tokens = nn.Conv2d(spec.embed_channels, spec.base_channels, 1)

        self.merge = conv3x3(2 * spec.base_channels, spec.base_channels)
        self.output_norm = nn.GroupNorm(group_count(spec.base_channels), spec.base_channels)
        self.output_conv = nn.Conv2d(spec.base_channels, layout.total_channels, kernel_size=1)
        self._initialize_output_bias()

    def _initialize_output_bias(self) -> None:
        """Uscita iniziale neutra, come nella rete a U, perche' il confronto sia equo.

        Se una delle due partisse da rumore e l'altra da una previsione piatta, la
        differenza misurata comprenderebbe anche il punto di partenza.
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

        locale = nn.functional.gelu(self.stem_norm(self.stem(padded)))

        token = self.to_tokens(locale)
        for blocco in self.blocks:
            token = blocco(token)
        lotto, canali, altezza, larghezza = token.shape
        token = (
            self.token_norm(token.flatten(2).transpose(1, 2))
            .transpose(1, 2)
            .reshape(lotto, canali, altezza, larghezza)
        )

        globale = nn.functional.interpolate(
            self.from_tokens(token), size=locale.shape[-2:], mode="bilinear",
            align_corners=False,
        )

        unito = self.merge(torch.cat([locale, globale], dim=1))
        out = self.output_conv(nn.functional.gelu(self.output_norm(unito)))
        return crop_padding(out, padding)


__all__ = ["GlobalBlock", "GlobalContextNet", "GlobalNetworkSpec"]
