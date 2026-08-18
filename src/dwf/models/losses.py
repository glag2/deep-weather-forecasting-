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

# Chiave con cui il campione trasporta il peso spaziale del proprio ritaglio.
KEY_SPATIAL_WEIGHT = "spatial_weight"


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


def soft_clamp(
    values: torch.Tensor, minimum: float = MIN_LOG_VAR, maximum: float = MAX_LOG_VAR
) -> torch.Tensor:
    """Limita `values` all'intervallo dato **senza annullare il gradiente ai bordi**.

    Perche' non `clamp`. Il taglio secco ha gradiente esattamente nullo fuori
    dall'intervallo: misurato, con log_var 15 il gradiente vale 0,000000. Un canale
    spinto oltre il limite da un solo passo troppo lungo non riceve piu' alcuna forza che
    lo riporti dentro, e resta muto per tutto il resto dell'addestramento. La zona non e'
    attrattiva (appena sotto il limite il gradiente punta verso l'interno), quindi il
    difetto e' raro, ma quando accade e' permanente e silenzioso.

    Questa versione, costruita con due `softplus` speculari, e' misurata cosi': lo scarto
    dall'identita' vale 0,0009 a sette unita' dal limite, 0,007 a cinque, 0,049 a tre. Le
    log-varianze utili stanno fra -3 e +3, dove la distorsione e' sotto il millesimo.

    Il gradiente decade in modo esponenziale ma non si annulla: 1,3e-1 a log_var 11,
    3,3e-3 a 15, 1,5e-7 a 25. Un canale finito la' fuori rientra lentamente, invece di
    restare fermo per sempre.
    """
    if maximum <= minimum:
        raise ValueError(f"Intervallo vuoto: [{minimum}, {maximum}]")
    dal_basso = minimum + F.softplus(values - minimum)
    return maximum - F.softplus(maximum - dal_basso)


def gaussian_nll(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-verosimiglianza negativa gaussiana, a meno di costanti additive."""
    limitata = soft_clamp(log_var)
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


def spectral_amplitude_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Scarto fra gli spettri di ampiezza di previsione e osservazione.

    Serve a contrastare la doppia penalizzazione. Con un errore quadratico puro, se la
    correlazione fra previsione e realta' vale rho, il minimo si ottiene producendo un
    campo con ampiezza rho volte quella vera: sfumare conviene, perche' una struttura
    nel posto sbagliato viene punita due volte, dove c'e' e dove manca. Il risultato e'
    una previsione troppo liscia, ed e' esattamente il difetto misurato su questo
    modello, che sottostimava di 4,8 gradi l'escursione a mezzogiorno.

    Confrontare i moduli della trasformata di Fourier bidimensionale misura quanta
    energia c'e' a ogni scala **senza guardare dove si trova**, quindi premia l'ampiezza
    corretta senza reintrodurre la penalizzazione di posizione. Non sostituisce
    l'errore quadratico, lo affianca: da solo sarebbe soddisfatto da un campo con lo
    spettro giusto e la fase sbagliata.

    La normalizzazione per il numero di celle rende il valore confrontabile con una
    varianza per punto, cosi' il peso del termine non dipende dalla dimensione del
    ritaglio.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"Forme incompatibili: previsione {tuple(prediction.shape)}, "
            f"osservazione {tuple(target.shape)}"
        )
    if prediction.ndim < 2:
        raise ValueError("Servono almeno due dimensioni spaziali")
    altezza, larghezza = prediction.shape[-2:]
    spettro_previsto = torch.fft.rfft2(prediction, norm="backward").abs()
    spettro_vero = torch.fft.rfft2(target, norm="backward").abs()
    return (spettro_previsto - spettro_vero).square().mean() / (altezza * larghezza)


class CompositeLoss:
    """Somma pesata delle perdite di tutte le teste dichiarate nel layout."""

    # Un peso letto con un default silenzioso e' un difetto in attesa: se un campo della
    # configurazione viene rinominato, il termine corrispondente prende il valore di
    # comodo invece di quello scelto, e la corsa sembra riuscita. I nomi vanno quindi
    # dichiarati e la loro assenza deve fermare tutto.
    NOMI_DEI_PESI = (
        ("weight_gaussian", "gaussian"),
        ("weight_occurrence", "precip_occurrence"),
        ("weight_amount", "precip_amount"),
        ("weight_fraction", "snow_fraction"),
        ("weight_spectral", "spectral"),
    )

    def __init__(self, layout: OutputLayout, weights: object) -> None:
        self.layout = layout
        mancanti = [
            nome for _, nome in self.NOMI_DEI_PESI if not hasattr(weights, nome)
        ]
        if mancanti:
            raise ValueError(
                f"Pesi della perdita incompleti: mancano {mancanti}. Dichiararli "
                f"esplicitamente, anche a zero, invece di lasciarli al caso"
            )
        for attributo, nome in self.NOMI_DEI_PESI:
            setattr(self, attributo, float(getattr(weights, nome)))

    def _combina(
        self, mask: torch.Tensor | None, spatial: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Fonde maschera di validita' e peso spaziale in un unico peso per punto."""
        if spatial is None:
            return mask
        return spatial if mask is None else mask * spatial

    def __call__(
        self, prediction: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> LossBreakdown:
        componenti: dict[str, torch.Tensor] = {}
        totale = prediction.sum() * 0.0

        peso_spaziale = batch.get(KEY_SPATIAL_WEIGHT)
        if peso_spaziale is not None:
            # Arriva come (B, H, W) e va trasmesso su tutte le scadenze.
            peso_spaziale = peso_spaziale.unsqueeze(1)

        for variabile in self.layout.variables:
            testa = self.layout.head_of(variabile)

            if testa == "gaussian":
                bersaglio = _require(batch, f"target_{variabile}")
                media = self.layout.select(prediction, variabile, "mean")
                perdita = gaussian_nll(
                    media,
                    self.layout.select(prediction, variabile, "log_var"),
                    bersaglio,
                    self._combina(batch.get(f"mask_{variabile}"), peso_spaziale),
                )
                componenti[f"{variabile}_nll"] = perdita
                totale = totale + self.weight_gaussian * perdita

                if self.weight_spectral > 0.0:
                    # Solo sulle variabili continue: il termine spettrale su un campo a
                    # code lunghe come la precipitazione peggiora le metriche di
                    # occorrenza, come documentato in RESEARCH.md.
                    spettrale = spectral_amplitude_loss(media, bersaglio)
                    componenti[f"{variabile}_spectral"] = spettrale
                    totale = totale + self.weight_spectral * spettrale

            elif testa == "hurdle":
                occorrenza = _require(batch, f"target_{variabile}_occurrence")
                perdita_occ = hurdle_occurrence_loss(
                    self.layout.select(prediction, variabile, "occurrence_logit"),
                    occorrenza,
                    self._combina(
                        batch.get(f"mask_{variabile}_occurrence"), peso_spaziale
                    ),
                )
                componenti[f"{variabile}_occurrence"] = perdita_occ
                totale = totale + self.weight_occurrence * perdita_occ

                quantita = _require(batch, f"target_{variabile}_amount")
                maschera = _require(batch, f"mask_{variabile}_amount")
                perdita_amt = hurdle_amount_loss(
                    self.layout.select(prediction, variabile, "amount"),
                    quantita,
                    self._combina(maschera, peso_spaziale),
                )
                componenti[f"{variabile}_amount"] = perdita_amt
                totale = totale + self.weight_amount * perdita_amt

            elif testa == "fraction_of":
                frazione = _require(batch, f"target_{variabile}_fraction")
                maschera = _require(batch, f"mask_{variabile}_fraction")
                perdita_fr = fraction_loss(
                    self.layout.select(prediction, variabile, "fraction_logit"),
                    frazione,
                    self._combina(maschera, peso_spaziale),
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
    "KEY_SPATIAL_WEIGHT",
    "MAX_LOG_VAR",
    "MIN_LOG_VAR",
    "CompositeLoss",
    "LossBreakdown",
    "fraction_loss",
    "gaussian_nll",
    "hurdle_amount_loss",
    "hurdle_occurrence_loss",
    "masked_mean",
    "soft_clamp",
    "spectral_amplitude_loss",
]
