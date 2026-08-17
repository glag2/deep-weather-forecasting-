"""Registro delle varianti di blocco confrontabili.

Il confronto fra architetture ha senso solo se cambia **una cosa sola**. Qui la cosa
sola e' il blocco elementare: lo scheletro a U, il numero di canali di ingresso, il
layout di uscita, i dati, la perdita e il protocollo di addestramento restano identici
per tutte. Ogni variante espone la stessa firma di costruzione

    blocco(in_channels, out_channels, dropout) -> nn.Module

e trasforma un tensore ``(B, C_in, H, W)`` in ``(B, C_out, H, W)`` senza cambiare la
risoluzione. Sotto questo contratto le varianti sono intercambiabili e i loro risultati
sono direttamente confrontabili.

Le varianti che perdono il confronto **non vengono cancellate**: restano qui,
documentate e selezionabili da configurazione con ``model.variant``, sia perche' il
risultato potrebbe ribaltarsi con piu' dati, sia perche' la misura che le ha scartate
resta riproducibile.

Le ragioni della scelta di queste quattro famiglie, e dell'esclusione di altre, sono in
`RESEARCH.md`.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from dwf.models.variants.attention import WindowAttentionBlock
from dwf.models.variants.conv import ConvBlock
from dwf.models.variants.fourier import FourierBlock
from dwf.models.variants.hybrid import HybridBlock
from dwf.models.variants.recurrent import RecurrentBlock

BlockFactory = Callable[[int, int, float], nn.Module]


class VariantError(KeyError):
    """Variante di blocco non registrata."""


VARIANTS: dict[str, BlockFactory] = {
    "conv": ConvBlock,
    "attention": WindowAttentionBlock,
    "fourier": FourierBlock,
    "recurrent": RecurrentBlock,
    "hybrid": HybridBlock,
}

DESCRIPTIONS: dict[str, str] = {
    "conv": "Convoluzione residua 3x3. Riferimento: ricettivo locale, costo lineare.",
    "attention": (
        "Autoattenzione a finestre di 8x8 con bias di posizione relativa. Pesi che "
        "dipendono dal contenuto, entro la finestra."
    ),
    "fourier": (
        "Convoluzione spettrale sui modi bassi della trasformata di Fourier. Ricettivo "
        "globale in una sola operazione, costo O(N log N)."
    ),
    "recurrent": (
        "Ricorrenza convoluzionale a pesi condivisi, iterata piu' volte. Profondita' "
        "effettiva senza parametri aggiuntivi."
    ),
    "hybrid": (
        "Somma di ramo convolutivo e ramo spettrale: dettaglio locale e struttura "
        "globale nello stesso blocco."
    ),
}


def available() -> tuple[str, ...]:
    """Nomi delle varianti registrate, in ordine stabile."""
    return tuple(VARIANTS)


def get(name: str) -> BlockFactory:
    """Fabbrica di blocchi corrispondente al nome, con errore esplicito se assente."""
    if name not in VARIANTS:
        raise VariantError(
            f"Variante {name!r} sconosciuta. Disponibili: {', '.join(available())}"
        )
    return VARIANTS[name]


def describe(name: str) -> str:
    """Descrizione sintetica della variante, usata nelle tabelle di confronto."""
    return DESCRIPTIONS.get(name, "")


__all__ = [
    "DESCRIPTIONS",
    "VARIANTS",
    "BlockFactory",
    "ConvBlock",
    "FourierBlock",
    "HybridBlock",
    "RecurrentBlock",
    "VariantError",
    "WindowAttentionBlock",
    "available",
    "describe",
    "get",
]
