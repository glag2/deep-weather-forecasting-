"""Funzioni di perdita, una per tipo di testa probabilistica.

Il modello non prevede un numero ma una distribuzione, quindi la perdita e' la
log-verosimiglianza negativa di quella distribuzione. E' questa scelta che rende
misurabile l'affidabilita' richiesta: una previsione sicura e sbagliata viene punita
piu' di una previsione incerta e sbagliata, e il modello impara a dichiarare quando
non sa.

Ogni perdita e' mediata solo sui punti che la riguardano: la quantita' di pioggia si
addestra dove piove, la frazione di neve dove la precipitazione e' misurabile.
Mediare sui punti esclusi spingerebbe il modello verso lo zero ovunque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dwf.models.heads import OutputLayout

# Limiti della log-varianza prevista. Senza di essi la rete puo' minimizzare la NLL
# facendo esplodere la varianza sui punti difficili invece di migliorare la media,
# e l'esponenziale va in overflow.
MIN_LOG_VAR = -10.0
MAX_LOG_VAR = 10.0

LOG_TWO_PI = math.log(2.0 * math.pi)

# Sotto questo numero di punti validi il contributo viene ignorato: una media su
# pochissimi punti e' rumore che destabilizza il gradiente.
MIN_VALID_POINTS = 1.0


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    """Perdita totale e sue componenti, per diagnosticare quale testa non impara."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]

    def detached(self) -> dict[str, float]:
        valori = {nome: float(valore.detach()) for nome, valore in self.components.items()}
        valori["total"] = float(self.total.detach())
        return valori


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Media sui soli punti ammessi; zero se non ce ne sono abbastanza."""
    if mask is None:
        return values.mean()
    peso = mask.sum()
    if float(peso) < MIN_VALID_POINTS:
        return values.sum() * 0.0
    return (values * mask).sum() / peso


def gaussian_nll(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-verosimiglianza negativa gaussiana, a meno di costanti additive."""
    limitata = log_var.clamp(MIN_LOG_VAR, MAX_LOG_VAR)
    residuo = target - mean
    punto = 0.5 * (LOG_TWO_PI + limitata + residuo.square() * torch.exp(-limitata))
    return masked_mean(punto, mask)


def hurdle_occurrence_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Entropia incrociata sulla probabilita' di superare la soglia."""
    punto = F.binary_cross_entropy_with_logits(
        logit, target, reduction="none", pos_weight=pos_weight
    )
    return masked_mean(punto, mask)


def hurdle_amount_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Errore sulla quantita', valutato solo dove la soglia e' superata.

    Si usa la perdita di Huber e non l'errore quadratico: la distribuzione della
    precipitazione ha code lunghe e pochi eventi estremi dominerebbero il gradiente.
    """
    punto = F.smooth_l1_loss(prediction, target, reduction="none", beta=1.0)
    return masked_mean(punto, mask)


def fraction_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Entropia incrociata su una frazione continua in [0, 1].

    Il bersaglio non e' binario ma una proporzione, e l'entropia incrociata con
    bersaglio continuo e' la scelta naturale: e' minimizzata quando la probabilita'
    prevista uguaglia la frazione osservata.
    """
    punto = F.binary_cross_entropy_with_logits(
        logit, target.clamp(0.0, 1.0), reduction="none"
    )
    return masked_mean(punto, mask)


class CompositeLoss:
    """Somma pesata delle perdite di tutte le teste dichiarate nel layout."""

    def __init__(self, layout: OutputLayout, weights: object) -> None:
        self.layout = layout
        self.weight_gaussian = float(getattr(weights, "gaussian", 1.0))
        self.weight_occurrence = float(getattr(weights, "precip_occurrence", 1.0))
        self.weight_amount = float(getattr(weights, "precip_amount", 1.0))
        self.weight_fraction = float(getattr(weights, "snow_fraction", 1.0))

    def __call__(
        self, prediction: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> LossBreakdown:
        componenti: dict[str, torch.Tensor] = {}
        totale = prediction.sum() * 0.0

        for variabile in self.layout.variables:
            testa = self.layout.head_of(variabile)

            if testa == "gaussian":
                bersaglio = _require(batch, f"target_{variabile}")
                perdita = gaussian_nll(
                    self.layout.select(prediction, variabile, "mean"),
                    self.layout.select(prediction, variabile, "log_var"),
                    bersaglio,
                    batch.get(f"mask_{variabile}"),
                )
                componenti[f"{variabile}_nll"] = perdita
                totale = totale + self.weight_gaussian * perdita

            elif testa == "hurdle":
                occorrenza = _require(batch, f"target_{variabile}_occurrence")
                perdita_occ = hurdle_occurrence_loss(
                    self.layout.select(prediction, variabile, "occurrence_logit"),
                    occorrenza,
                    batch.get(f"mask_{variabile}_occurrence"),
                )
                componenti[f"{variabile}_occurrence"] = perdita_occ
                totale = totale + self.weight_occurrence * perdita_occ

                quantita = _require(batch, f"target_{variabile}_amount")
                maschera = _require(batch, f"mask_{variabile}_amount")
                perdita_amt = hurdle_amount_loss(
                    self.layout.select(prediction, variabile, "amount"),
                    quantita,
                    maschera,
                )
                componenti[f"{variabile}_amount"] = perdita_amt
                totale = totale + self.weight_amount * perdita_amt

            elif testa == "fraction_of":
                frazione = _require(batch, f"target_{variabile}_fraction")
                maschera = _require(batch, f"mask_{variabile}_fraction")
                perdita_fr = fraction_loss(
                    self.layout.select(prediction, variabile, "fraction_logit"),
                    frazione,
                    maschera,
                )
                componenti[f"{variabile}_fraction"] = perdita_fr
                totale = totale + self.weight_fraction * perdita_fr

            else:  # pragma: no cover - le teste ammesse sono validate a monte
                raise ValueError(f"Testa non supportata nella perdita: {testa!r}")

        return LossBreakdown(total=totale, components=componenti)


def _require(batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in batch:
        raise KeyError(
            f"Il batch non contiene {key!r}. Presenti: {sorted(batch)}"
        )
    return batch[key]


__all__ = [
    "MAX_LOG_VAR",
    "MIN_LOG_VAR",
    "CompositeLoss",
    "LossBreakdown",
    "fraction_loss",
    "gaussian_nll",
    "hurdle_amount_loss",
    "hurdle_occurrence_loss",
    "masked_mean",
]
