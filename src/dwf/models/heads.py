"""Mappa dei canali di uscita della rete.

La rete produce un solo tensore ``(B, C, H, W)`` che contiene, per ogni variabile
prevista, ogni componente della sua testa e ogni lead time. Indicizzare quei canali
a mano e' l'errore piu' facile e piu' silenzioso possibile: scambiare media e
log-varianza non fa fallire nulla, produce solo previsioni sbagliate. Questo modulo
rende la mappatura esplicita, verificabile e condivisa tra modello, loss,
valutazione e inferenza.

Layout dei canali: blocchi contigui di ``output_slots`` canali, uno per ciascuna
coppia (variabile, componente), nell'ordine di dichiarazione dei target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

# Componenti prodotte da ciascun tipo di testa, nell'ordine in cui occupano i canali.
HEAD_COMPONENTS: dict[str, tuple[str, ...]] = {
    "gaussian": ("mean", "log_var"),
    "hurdle": ("occurrence_logit", "amount"),
    "fraction_of": ("fraction_logit",),
}


@dataclass(frozen=True, slots=True)
class ChannelBlock:
    """Un blocco contiguo di canali dedicato a una coppia (variabile, componente)."""

    variable: str
    head: str
    component: str
    start: int
    stop: int

    @property
    def n_channels(self) -> int:
        return self.stop - self.start

    @property
    def key(self) -> tuple[str, str]:
        return (self.variable, self.component)


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Disposizione dei canali di uscita, derivata dai target e dall'orizzonte."""

    blocks: tuple[ChannelBlock, ...]
    output_slots: int

    @property
    def total_channels(self) -> int:
        return sum(block.n_channels for block in self.blocks)

    @property
    def variables(self) -> tuple[str, ...]:
        seen: list[str] = []
        for block in self.blocks:
            if block.variable not in seen:
                seen.append(block.variable)
        return tuple(seen)

    def head_of(self, variable: str) -> str:
        for block in self.blocks:
            if block.variable == variable:
                return block.head
        raise KeyError(f"Variabile non presente nel layout di uscita: {variable!r}")

    def block(self, variable: str, component: str) -> ChannelBlock:
        for candidate in self.blocks:
            if candidate.key == (variable, component):
                return candidate
        available = sorted(candidate.key for candidate in self.blocks)
        raise KeyError(
            f"Componente ({variable!r}, {component!r}) assente. Disponibili: {available}"
        )

    def components_of(self, variable: str) -> tuple[str, ...]:
        return HEAD_COMPONENTS[self.head_of(variable)]

    def select(self, prediction: torch.Tensor, variable: str, component: str) -> torch.Tensor:
        """Estrae ``(B, output_slots, H, W)`` per una coppia (variabile, componente)."""
        if prediction.ndim != 4:
            raise ValueError(
                f"Attesa una previsione (B, C, H, W), ricevuta forma {tuple(prediction.shape)}"
            )
        if prediction.shape[1] != self.total_channels:
            raise ValueError(
                f"La previsione ha {prediction.shape[1]} canali, il layout ne richiede "
                f"{self.total_channels}"
            )
        target_block = self.block(variable, component)
        return prediction[:, target_block.start : target_block.stop]

    def split(self, prediction: torch.Tensor) -> dict[tuple[str, str], torch.Tensor]:
        """Scompone la previsione in un dizionario indicizzato per (variabile, componente)."""
        return {
            block.key: self.select(prediction, block.variable, block.component)
            for block in self.blocks
        }

    @classmethod
    def from_targets(
        cls, targets: Sequence[object], output_slots: int
    ) -> OutputLayout:
        """Costruisce il layout dai ``TargetConfig`` dichiarati in configurazione.

        Accetta qualunque oggetto con attributi ``name`` e ``head`` per non far
        dipendere il modello dal modulo di configurazione.
        """
        if output_slots < 1:
            raise ValueError(f"output_slots deve essere positivo: {output_slots}")
        if not targets:
            raise ValueError("Serve almeno un target per costruire il layout di uscita")

        blocks: list[ChannelBlock] = []
        cursor = 0
        for target in targets:
            name = target.name
            head = target.head
            if head not in HEAD_COMPONENTS:
                raise ValueError(
                    f"Testa non supportata per {name!r}: {head!r}. "
                    f"Ammesse: {sorted(HEAD_COMPONENTS)}"
                )
            for component in HEAD_COMPONENTS[head]:
                blocks.append(
                    ChannelBlock(
                        variable=name,
                        head=head,
                        component=component,
                        start=cursor,
                        stop=cursor + output_slots,
                    )
                )
                cursor += output_slots
        return cls(blocks=tuple(blocks), output_slots=output_slots)

    def apply_anchor(
        self, prediction: torch.Tensor, baselines: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Somma un riferimento noto alla media prevista, rendendo la rete un correttore.

        Senza ancoraggio la rete deve ricostruire da zero anche la parte di segnale che
        si ottiene gratis ripetendo l'osservazione piu' recente alla stessa ora del
        giorno, che sui dati di questo progetto vale gia' un errore quadratico di circa
        3,2 gradi contro i 4,7 della persistenza ingenua. Ancorando, la rete impara solo
        lo scarto da quel riferimento: un bersaglio di ampiezza molto minore e centrato
        su zero, quindi meglio condizionato.

        Tocca la sola componente ``mean``: la log-varianza descrive l'incertezza dello
        scarto e non va traslata.
        """
        if not baselines:
            return prediction
        corretta = prediction.clone()
        for variabile, riferimento in baselines.items():
            if self.head_of(variabile) != "gaussian":
                raise ValueError(
                    f"L'ancoraggio vale solo per le teste gaussiane, {variabile!r} ha "
                    f"testa {self.head_of(variabile)!r}"
                )
            blocco = self.block(variabile, "mean")
            attesa = prediction[:, blocco.start : blocco.stop].shape
            if tuple(riferimento.shape) != tuple(attesa):
                raise ValueError(
                    f"Il riferimento di {variabile!r} ha forma {tuple(riferimento.shape)}, "
                    f"attesa {tuple(attesa)}"
                )
            corretta[:, blocco.start : blocco.stop] = (
                prediction[:, blocco.start : blocco.stop] + riferimento
            )
        return corretta

    def describe(self) -> list[dict[str, object]]:
        """Descrizione tabellare del layout, per la tabella dei canali e i notebook."""
        return [
            {
                "variable": block.variable,
                "head": block.head,
                "component": block.component,
                "start": block.start,
                "stop": block.stop,
                "n_channels": block.n_channels,
            }
            for block in self.blocks
        ]
