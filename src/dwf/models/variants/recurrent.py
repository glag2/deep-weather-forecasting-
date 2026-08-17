"""Variante a ricorrenza convoluzionale con pesi condivisi.

Avvertenza sull'onesta' del confronto: **questo non e' il ConvLSTM temporale** della
letteratura. Quello consuma una sequenza ``(B, T, C, H, W)`` e aggiorna uno stato slot
dopo slot; qui il contratto comune impone un ingresso ``(B, C, H, W)`` in cui i 21 slot
osservati sono gia' impacchettati nei canali insieme a tendenze, vento, campi statici e
codifiche temporali. Trasformarlo in una sequenza richiederebbe di cambiare il layout di
ingresso, cioe' proprio la variabile che il confronto tiene fissa.

Cio' che si misura qui e' l'altra proprieta' interessante della ricorrenza: la
**profondita' effettiva a parametri costanti**. La stessa cella convoluzionale viene
applicata piu' volte, quindi il campo ricettivo e il numero di trasformazioni crescono
senza aggiungere un solo peso. E' l'ipotesi che vale la pena testare visto che la
ricerca riporta un peggioramento delle architetture ricorrenti quando si aggiungono
parametri.

La cella e' una GRU convoluzionale: rispetto a un LSTM ha un gate in meno, quindi meno
parametri a parita' di canali, e sulle sequenze corte come questa la differenza di
capacita' non e' osservabile.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from dwf.models.blocks import conv3x3, group_count

# Iterazioni della cella. Tre bastano a triplicare il campo ricettivo del blocco; oltre,
# il costo cresce linearmente mentre il guadagno atteso si appiattisce.
DEFAULT_STEPS = 3


class ConvGRUCell(nn.Module):
    """Cella GRU in cui i prodotti matriciali sono convoluzioni 3x3."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Un'unica convoluzione produce entrambi i gate: e' equivalente a due separate
        # e dimezza il numero di chiamate.
        self.gates = conv3x3(channels * 2, channels * 2)
        self.candidate = conv3x3(channels * 2, channels)

    def forward(self, state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        unito = torch.cat([state, inputs], dim=1)
        aggiornamento, azzeramento = torch.chunk(torch.sigmoid(self.gates(unito)), 2, dim=1)
        proposta = torch.tanh(self.candidate(torch.cat([azzeramento * state, inputs], dim=1)))
        return (1.0 - aggiornamento) * state + aggiornamento * proposta


class RecurrentBlock(nn.Module):
    """Blocco residuo che itera una cella ricorrente a pesi condivisi."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0,
        steps: int = DEFAULT_STEPS,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError(f"Servono almeno un'iterazione, ricevute {steps}")
        self.steps = steps
        self.norm = nn.GroupNorm(group_count(in_channels), in_channels)
        self.entry = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.cell = ConvGRUCell(out_channels)
        self.norm_out = nn.GroupNorm(group_count(out_channels), out_channels)
        self.project = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ingresso = self.entry(functional.gelu(self.norm(features)))
        # Lo stato parte dall'ingresso proiettato: partire da zero sprecherebbe la prima
        # iterazione a ricostruire cio' che e' gia' disponibile.
        stato = ingresso
        for _ in range(self.steps):
            stato = self.cell(stato, ingresso)
        stato = self.dropout(stato)
        return self.project(functional.gelu(self.norm_out(stato))) + self.shortcut(features)


__all__ = ["DEFAULT_STEPS", "ConvGRUCell", "RecurrentBlock"]
