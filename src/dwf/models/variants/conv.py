"""Variante di riferimento: convoluzione residua.

E' il blocco con cui e' stato addestrato il primo modello, quindi funge da metro di
paragone per tutte le altre. Non e' registrato come "vecchio": e' il riferimento
rispetto al quale una variante piu' elaborata deve dimostrare di valere il proprio
costo.
"""

from __future__ import annotations

from torch import nn

from dwf.models.blocks import ResidualBlock


class ConvBlock(ResidualBlock):
    """Blocco residuo pre-attivato a due convoluzioni 3x3.

    Sottoclasse senza modifiche: serve solo a dare alla variante di riferimento un nome
    proprio nel registro, cosi' che nelle tabelle di confronto compaia come una scelta
    esplicita e non come l'assenza di una scelta.
    """


def _check_contract() -> None:  # pragma: no cover - eseguito dai test
    assert issubclass(ConvBlock, nn.Module)


__all__ = ["ConvBlock"]
