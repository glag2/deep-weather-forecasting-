"""CMuon: Muon con le matrici fuse spezzate prima dell'ortogonalizzazione.

Perche' esiste questo modulo. Il collo di bottiglia misurato del progetto non e' la
capacita' della rete ma il numero di passi di ottimizzazione che una CPU a quattro thread
riesce a fare (2.560 in totale, contro i 15.000 di MET Norway): un ottimizzatore che
impara di piu' per passo attacca il difetto vero. AdamW, fino a oggi, era un default mai
messo in discussione.

Muon sostituisce l'aggiornamento coordinata per coordinata con la direzione
**ortogonalizzata** del momento: dato il gradiente G di una matrice, si applica
l'approssimazione di U V^T (la parte ortogonale della decomposizione ai valori singolari)
con poche iterazioni quintiche di Newton-Schulz. L'aggiornamento risultante ha valori
singolari tutti prossimi a 1, cioe' aggiorna con la stessa forza tutte le direzioni dello
spazio dei pesi invece di privilegiare quelle con gradiente grande.

La parte "C" viene da CMuon (arXiv:2608.02502): applicare l'ortogonalizzazione a una
matrice **fusa** provoca interferenza di sottospazi, perche' costruisce un solo
precondizionatore per blocchi con statistiche di gradiente diverse. Da noi il caso e'
concreto e non ipotetico: `nn.MultiheadAttention` tiene query, chiave e valore in un unico
`in_proj_weight` (576x192), quindi Muon ingenuo ricadrebbe esattamente nel caso che il
paper documenta come dannoso. Qui la matrice viene spezzata nei suoi blocchi funzionali
prima di Newton-Schulz.

Avvertenza. I numeri del paper sono a 675M parametri e lotto 1024; noi abbiamo 1,9M
parametri e lotto 4, e le iterazioni di Newton-Schulz su CPU costano tempo. Per questo il
default della configurazione resta AdamW: si adotta solo se vince sul banco.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch import Tensor, nn

# Coefficienti quintici di Newton-Schulz. Non approssimano la funzione segno con
# precisione: sono scelti per portare rapidamente tutti i valori singolari in un intorno
# di 1, che e' l'unica cosa che serve all'aggiornamento.
COEFFICIENTI_NEWTON_SCHULZ = (3.4445, -4.7750, 2.0315)
PASSI_NEWTON_SCHULZ = 5
# Riscalatura di CMuon: rende la RMS dell'aggiornamento pari a circa 0,2, cioe' quella
# tipica di AdamW, cosi' il learning rate mantiene lo stesso ordine di grandezza.
FATTORE_RISCALATURA = 0.2

__all__ = [
    "COEFFICIENTI_NEWTON_SCHULZ",
    "PASSI_NEWTON_SCHULZ",
    "CMuon",
    "MediaEsponenziale",
    "OptimError",
    "build_optimizer",
    "ortogonalizza",
    "split_parameters",
]


class OptimError(RuntimeError):
    """Configurazione dell'ottimizzatore incoerente con i parametri della rete."""


class MediaEsponenziale:
    """Media esponenziale dei pesi, mantenuta a fianco della rete che si addestra.

    A cosa serve qui. L'ultima epoca non e' necessariamente la migliore posizione dei
    pesi: con pochi passi e lotti piccoli l'ottimizzatore oscilla attorno al minimo, e la
    media delle posizioni recenti cade piu' vicino al centro della conca di quanto ci cada
    l'ultimo passo. E' un guadagno che non costa calcolo aggiuntivo nel passo, solo una
    copia dei pesi in memoria.

    Rampa iniziale. Con un coefficiente fisso di 0,999 e circa 250 passi per epoca la
    media resterebbe ancorata all'inizializzazione per gran parte della corsa, cioe'
    sarebbe peggiore dei pesi veri senza che se ne capisca il motivo. Il coefficiente
    parte quindi da valori bassi e cresce, come si fa con la correzione di bias di Adam:
    al passo n non si usa mai piu' di (1 + n) / (10 + n).

    Cosa non fa. Non decide nulla da sola: chi la usa deve valutare *entrambe* le copie
    sulla validazione e conservare quella che vince. E' l'unico modo di aggiungerla senza
    scommettere, dato che il suo effetto in questo regime non e' misurato.
    """

    def __init__(self, network: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise OptimError(f"Il coefficiente della media deve stare in (0, 1): {decay}")
        self.decay = decay
        self.passi = 0
        # `detach().clone()` e non un riferimento: la media deve essere una posizione
        # distinta, altrimenti seguirebbe i pesi invece di mediarli.
        self.ombra: dict[str, Tensor] = {
            nome: valore.detach().clone().float()
            for nome, valore in network.state_dict().items()
            if valore.is_floating_point()
        }

    def coefficiente(self) -> float:
        return min(self.decay, (1.0 + self.passi) / (10.0 + self.passi))

    @torch.no_grad()
    def update(self, network: nn.Module) -> None:
        self.passi += 1
        coefficiente = self.coefficiente()
        stato = network.state_dict()
        for nome, media in self.ombra.items():
            media.mul_(coefficiente).add_(stato[nome].detach().float(), alpha=1.0 - coefficiente)

    def state_dict(self) -> dict[str, Tensor]:
        """I pesi medi, nella forma attesa da `load_state_dict` della rete."""
        return {nome: valore.clone() for nome, valore in self.ombra.items()}

    @contextlib.contextmanager
    def applicata(self, network: nn.Module) -> Iterator[None]:
        """Installa temporaneamente i pesi medi, per valutarli, e poi rimette i veri.

        Il ripristino sta in `finally` perche' un'eccezione durante la validazione
        lascerebbe altrimenti la rete con i pesi medi e l'addestramento proseguirebbe da
        una posizione diversa da quella raggiunta, senza alcun segnale.
        """
        originali = {
            nome: valore.detach().clone() for nome, valore in network.state_dict().items()
        }
        try:
            network.load_state_dict(
                {
                    nome: self.ombra[nome].to(valore.dtype)
                    if nome in self.ombra
                    else valore
                    for nome, valore in originali.items()
                }
            )
            yield
        finally:
            network.load_state_dict(originali)


def ortogonalizza(matrice: Tensor, passi: int = PASSI_NEWTON_SCHULZ) -> Tensor:
    """Approssima la parte ortogonale di `matrice` con iterazioni quintiche.

    La normalizzazione di Frobenius iniziale serve a portare il valore singolare massimo
    sotto 1, condizione senza la quale l'iterazione diverge. La trasposizione quando le
    righe superano le colonne mantiene il prodotto intermedio sul lato piu' piccolo.

    L'iterazione **non converge all'identita'**: misurato qui, il punto fisso dei valori
    singolari sta fra 0,68 e 1,14. E' voluto, i coefficienti sono tarati per la velocita'.
    Sui gradienti molto degeneri, invece, cinque passi non bastano: un condizionamento
    iniziale di 1e4 scende a 37 con cinque passi e a 1,7 con dieci.
    """
    if matrice.ndim != 2:
        raise OptimError(f"Newton-Schulz richiede una matrice, ricevute {matrice.ndim} dimensioni")

    a, b, c = COEFFICIENTI_NEWTON_SCHULZ
    trasposta = matrice.shape[0] > matrice.shape[1]
    corrente = matrice.T if trasposta else matrice
    corrente = corrente / (corrente.norm() + 1e-7)

    for _ in range(passi):
        prodotto = corrente @ corrente.T
        corrente = a * corrente + (b * prodotto + c * prodotto @ prodotto) @ corrente

    return corrente.T if trasposta else corrente


class CMuon(torch.optim.Optimizer):
    """Muon con momento di Nesterov, weight decay disaccoppiato e matrici spezzate.

    Ogni gruppo di parametri accetta `chunks`: il numero di blocchi funzionali in cui la
    matrice va divisa lungo la prima dimensione prima di ortogonalizzare. Con 1 il
    comportamento e' quello di Muon.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = PASSI_NEWTON_SCHULZ,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise OptimError(f"momentum {momentum} fuori da [0, 1)")
        if ns_steps < 1:
            raise OptimError(f"ns_steps {ns_steps} deve essere almeno 1")
        super().__init__(
            params,  # type: ignore[arg-type]
            {
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
                "chunks": 1,
            },
        )
        for gruppo in self.param_groups:
            for parametro in gruppo["params"]:
                if parametro.ndim < 2:
                    raise OptimError(
                        "CMuon accetta solo tensori con almeno due dimensioni: norme, bias "
                        "ed embedding vanno affidati ad AdamW"
                    )
                if parametro.shape[0] % gruppo["chunks"]:
                    raise OptimError(
                        f"la prima dimensione {parametro.shape[0]} non e' divisibile per "
                        f"chunks {gruppo['chunks']}: i blocchi funzionali sarebbero disuguali"
                    )

    @torch.no_grad()
    def step(self, closure: Any = None) -> float | None:
        perdita = None
        if closure is not None:
            with torch.enable_grad():
                perdita = closure()

        for gruppo in self.param_groups:
            for parametro in gruppo["params"]:
                if parametro.grad is None:
                    continue
                stato = self.state[parametro]
                if "momento" not in stato:
                    stato["momento"] = torch.zeros_like(parametro)

                momento = stato["momento"]
                momento.lerp_(parametro.grad, 1.0 - gruppo["momentum"])
                direzione = (
                    parametro.grad.lerp(momento, gruppo["momentum"])
                    if gruppo["nesterov"]
                    else momento.clone()
                )

                aggiornamento = self._direzione_ortogonale(
                    direzione, gruppo["chunks"], gruppo["ns_steps"]
                )
                if gruppo["weight_decay"]:
                    parametro.mul_(1.0 - gruppo["lr"] * gruppo["weight_decay"])
                parametro.add_(aggiornamento.view_as(parametro), alpha=-gruppo["lr"])

        return perdita

    @staticmethod
    def _direzione_ortogonale(direzione: Tensor, chunks: int, passi: int) -> Tensor:
        """Ortogonalizza ogni blocco funzionale separatamente e riscala.

        Le convoluzioni vengono viste come matrici (uscite, ingressi x nucleo): e' la
        forma su cui l'ortogonalizzazione ha il significato di trattare allo stesso modo
        tutte le direzioni dello spazio dei filtri.
        """
        piatta = direzione.reshape(direzione.shape[0], -1)
        blocchi = piatta.chunk(chunks, dim=0) if chunks > 1 else (piatta,)
        risultato = torch.cat([ortogonalizza(blocco, passi) for blocco in blocchi], dim=0)
        righe, colonne = piatta.shape
        return risultato * (FATTORE_RISCALATURA * math.sqrt(max(righe, colonne)))


# --------------------------------------------------------------------------- #
# Ripartizione dei parametri
# --------------------------------------------------------------------------- #

# Moduli i cui pesi restano ad AdamW. `stem` legge i canali grezzi e `output_conv` emette
# i parametri delle distribuzioni: sono i due estremi della rete, dove la letteratura su
# Muon lascia sempre AdamW perche' la scala del gradiente non e' quella dei blocchi
# interni. `to_tokens` e' la proiezione delle patch, cioe' l'equivalente dell'embedding di
# un modello linguistico, che Muon esclude per convenzione; e' anche la matrice piu' larga
# della rete (192x3072) e da sola costava 26 dei 219 ms di ortogonalizzazione misurati.
# `position` e' la convoluzione posizionale depthwise: i suoi "filtri" hanno una sola
# coordinata d'ingresso, quindi non c'e' alcun sottospazio da equalizzare.
MODULI_ADAMW = ("stem", "to_tokens", "output_conv", "position")
# Matrici fuse: quante funzioni distinte contengono. `in_proj_weight` di
# nn.MultiheadAttention impila query, chiave e valore.
MATRICI_FUSE = {"in_proj_weight": 3}


def split_parameters(network: nn.Module) -> tuple[list[dict[str, Any]], list[Tensor]]:
    """Divide i parametri fra i gruppi di CMuon e la lista per AdamW.

    Restituisce i gruppi ortogonalizzabili, ciascuno con il proprio numero di blocchi, e i
    parametri che restano ad AdamW: tensori a una dimensione (norme, bias) e i due estremi
    della rete.
    """
    per_chunks: dict[int, list[Tensor]] = {}
    adamw: list[Tensor] = []

    for nome, parametro in network.named_parameters():
        if not parametro.requires_grad:
            continue
        pezzi = nome.split(".")
        escluso = any(pezzo in MODULI_ADAMW for pezzo in pezzi)
        if parametro.ndim < 2 or escluso:
            adamw.append(parametro)
            continue
        chunks = MATRICI_FUSE.get(pezzi[-1], 1)
        per_chunks.setdefault(chunks, []).append(parametro)

    gruppi = [
        {"params": parametri, "chunks": chunks}
        for chunks, parametri in sorted(per_chunks.items())
    ]
    return gruppi, adamw


class HybridOptimizer(torch.optim.Optimizer):
    """CMuon sulle matrici interne e AdamW su tutto il resto, come un solo oggetto.

    Serve perche' il ciclo di addestramento e lo scheduler parlano con un ottimizzatore
    solo: tenerne due separati significherebbe duplicare ogni chiamata e ogni salvataggio.
    """

    def __init__(self, muon: CMuon, adamw: torch.optim.AdamW) -> None:
        self.muon = muon
        self.adamw = adamw
        self.defaults = dict(adamw.defaults)
        # Nessuna copia: i gruppi sono gli stessi oggetti dei due ottimizzatori, cosi' uno
        # scheduler che scrive `lr` nei gruppi agisce davvero su entrambi.
        self.param_groups = muon.param_groups + adamw.param_groups

    @property  # type: ignore[override]
    def state(self) -> dict[Tensor, Any]:
        return {**self.muon.state, **self.adamw.state}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> float | None:
        perdita = None
        if closure is not None:
            with torch.enable_grad():
                perdita = closure()
        self.muon.step()
        self.adamw.step()
        return perdita

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def build_optimizer(
    network: nn.Module, *, kind: str, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Costruisce l'ottimizzatore richiesto dalla configurazione."""
    if kind == "adamw":
        return torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    if kind != "cmuon":
        raise OptimError(f"Ottimizzatore sconosciuto: {kind}")

    gruppi, resto = split_parameters(network)
    if not gruppi:
        raise OptimError(
            "Nessuna matrice ortogonalizzabile: con questa rete CMuon coinciderebbe con AdamW"
        )
    muon = CMuon(gruppi, lr=lr, weight_decay=weight_decay)
    # Lo stesso learning rate su entrambi i rami e' voluto: la riscalatura di CMuon serve
    # proprio a rendere le due ampiezze di aggiornamento comparabili.
    adamw = torch.optim.AdamW(resto, lr=lr, weight_decay=weight_decay)
    return HybridOptimizer(muon, adamw)
