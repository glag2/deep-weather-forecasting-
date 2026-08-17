"""Algebra degli slot temporali.

Uno *slot* e' un istante previsto (es. 06, 12, 18 UTC di un giorno). Tutto il
progetto indicizza il tempo per posizione nella sequenza contigua di slot, quindi
la conversione indice <-> istante e la definizione degli split vivono qui, in un
unico posto, per evitare che training, valutazione e inferenza divergano.

Convenzione ERA5 sulle cumulate: il valore di ``tp`` all'ora H rappresenta
l'accumulo sull'intervallo (H-1, H]. Sommare le ore H1..H2 copre quindi (H1-1, H2].

Il modulo lavora su tipi primitivi e non importa ``dwf.config``: e' la
configurazione a dipendere da qui per validare i propri parametri temporali.
"""

from __future__ import annotations

import calendar
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import numpy as np

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
GAP_LABEL = "gap"


# --------------------------------------------------------------------------- #
# Finestre di accumulo
# --------------------------------------------------------------------------- #


def accumulation_offsets(window_hours: int) -> list[int]:
    """Scarti orari, rispetto all'ora dello slot, da sommare per le variabili cumulate.

    Restituisce esattamente ``window_hours`` offset, il piu' possibile centrati sullo
    slot. Con finestra pari il centro cade tra due ore e la finestra viene spostata
    di mezz'ora in avanti: scelta arbitraria, fissata qui una volta per tutte.
    """
    if not 1 <= window_hours <= 24:
        raise ValueError(f"window_hours fuori range 1..24: {window_hours}")
    half = window_hours // 2
    offsets = (
        list(range(-half, half + 1))
        if window_hours % 2 == 1
        else list(range(-half + 1, half + 1))
    )
    if len(offsets) != window_hours:  # pragma: no cover - invariante aritmetica
        raise AssertionError(f"attesi {window_hours} offset, ottenuti {len(offsets)}")
    return offsets


def accumulation_hours(slot_hour: int, window_hours: int) -> list[int]:
    """Ore del giorno da sommare per lo slot dato.

    Solleva se la finestra sconfina fuori dal giorno: l'ingestione lavora mese per
    mese e una finestra a cavallo di mezzanotte del primo o ultimo giorno
    richiederebbe dati di un altro file, complicazione evitata per costruzione.
    """
    if not 0 <= slot_hour <= 23:
        raise ValueError(f"slot_hour fuori range 0..23: {slot_hour}")
    hours = [slot_hour + offset for offset in accumulation_offsets(window_hours)]
    if hours[0] < 0 or hours[-1] > 23:
        raise ValueError(
            f"La finestra di accumulo di {window_hours} h centrata sull'ora {slot_hour} "
            f"coprirebbe le ore {hours[0]}..{hours[-1]}, fuori dall'intervallo 0..23. "
            f"Ridurre accum_window_hours oppure spostare lo slot verso il centro del giorno."
        )
    return hours


def validate_accumulation_fits(slot_hours: Sequence[int], window_hours: int) -> None:
    """Verifica che ogni slot abbia la propria finestra interamente dentro il giorno."""
    for slot_hour in slot_hours:
        accumulation_hours(slot_hour, window_hours)


def required_hours(slot_hours: Sequence[int], window_hours: int) -> list[int]:
    """Ore che il download delle variabili cumulate deve coprire, ordinate e univoche."""
    needed: set[int] = set()
    for slot_hour in slot_hours:
        needed.update(accumulation_hours(slot_hour, window_hours))
    return sorted(needed)


def accumulation_coverage(slot_hours: Sequence[int], window_hours: int) -> tuple[int, int]:
    """Ore del giorno coperte dalle finestre e numero di conteggi ridondanti.

    Rende esplicito il compromesso della finestra: con slot 06/12/18 e finestra di
    8 h si coprono 20 delle 24 ore, con 4 conteggi in sovrapposizione. Le ore non
    coperte non compaiono in nessun target di precipitazione.
    """
    counts: dict[int, int] = {}
    for slot_hour in slot_hours:
        for hour in accumulation_hours(slot_hour, window_hours):
            counts[hour] = counts.get(hour, 0) + 1
    return len(counts), sum(count - 1 for count in counts.values())


# --------------------------------------------------------------------------- #
# Sequenze di slot
# --------------------------------------------------------------------------- #


def month_slot_times(year: int, month: int, slot_hours: Sequence[int]) -> list[datetime]:
    """Istanti di slot attesi per un singolo mese, in ordine crescente."""
    n_days = calendar.monthrange(year, month)[1]
    return [
        datetime(year, month, day, hour, tzinfo=UTC)
        for day in range(1, n_days + 1)
        for hour in sorted(slot_hours)
    ]


def expected_slot_times(
    months: Sequence[tuple[int, int]], slot_hours: Sequence[int]
) -> list[datetime]:
    """Tutti gli istanti di slot attesi nei mesi indicati, in ordine crescente."""
    out: list[datetime] = []
    for year, month in months:
        out.extend(month_slot_times(year, month, slot_hours))
    return out


def slot_of_day(moment: datetime, slot_hours: Sequence[int]) -> int:
    """Posizione dello slot dentro la giornata (0 = primo slot del giorno)."""
    ordered = sorted(slot_hours)
    try:
        return ordered.index(moment.hour)
    except ValueError:
        raise ValueError(
            f"L'ora {moment.hour} di {moment.isoformat()} non e' uno slot configurato "
            f"({ordered})"
        ) from None


def expected_steps(slot_hours: Sequence[int]) -> list[int]:
    """Distanza in ore da ciascuno slot al successivo, chiudendo il giro sul giorno dopo."""
    ordered = sorted(slot_hours)
    steps: list[int] = []
    for index, hour in enumerate(ordered):
        if index + 1 < len(ordered):
            steps.append(ordered[index + 1] - hour)
        else:
            steps.append(ordered[0] + 24 - hour)
    return steps


def find_gaps(
    times: Sequence[datetime], slot_hours: Sequence[int]
) -> list[tuple[datetime, datetime]]:
    """Coppie (precedente, successivo) tra cui manca almeno uno slot atteso.

    Le finestre scorrevoli presuppongono cadenza regolare: un buco non rilevato
    produrrebbe campioni che saltano nel tempo senza che nulla lo segnali, e il
    modello imparerebbe transizioni inesistenti.
    """
    if len(times) < 2:
        return []
    ordered = sorted(times)
    steps = expected_steps(slot_hours)
    gaps: list[tuple[datetime, datetime]] = []
    for previous, following in pairwise(ordered):
        expected_delta = timedelta(hours=steps[slot_of_day(previous, slot_hours)])
        if following - previous != expected_delta:
            gaps.append((previous, following))
    return gaps


# --------------------------------------------------------------------------- #
# Split temporali
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SplitLayout:
    """Blocchi temporali contigui e indici di campione ammessi per ciascuno."""

    bounds: dict[str, tuple[int, int]]
    sample_starts: dict[str, tuple[int, ...]]
    input_slots: int
    output_slots: int

    @property
    def total_window(self) -> int:
        return self.input_slots + self.output_slots

    def n_samples(self, split: str) -> int:
        return len(self.sample_starts[split])


def build_split_layout(
    n_slots: int,
    *,
    train_fraction: float,
    val_fraction: float,
    gap_slots: int,
    input_slots: int,
    output_slots: int,
) -> SplitLayout:
    """Divide l'asse temporale in train/val/test contigui e ne elenca i campioni.

    Il train precede sempre validation e test: valutare su dati precedenti a quelli
    di addestramento non misurerebbe capacita' previsionale. Tra i blocchi si
    scartano ``gap_slots`` slot per attenuare l'autocorrelazione, e un campione e'
    ammesso solo se l'intera finestra input+target ricade nel proprio blocco, cosi'
    nessun target di train puo' comparire tra gli input di validation.
    """
    if n_slots < 0:
        raise ValueError(f"n_slots negativo: {n_slots}")
    total_window = input_slots + output_slots
    if n_slots < total_window:
        raise ValueError(
            f"Servono almeno {total_window} slot per formare un campione, disponibili {n_slots}"
        )

    n_train = int(n_slots * train_fraction)
    n_val = int(n_slots * val_fraction)

    train_end = n_train
    val_start = train_end + gap_slots
    val_end = val_start + n_val
    test_start = val_end + gap_slots

    if test_start >= n_slots:
        raise ValueError(
            f"Con {n_slots} slot, frazioni ({train_fraction}, {val_fraction}) e gap "
            f"{gap_slots} il blocco di test resterebbe vuoto. Allungare il periodo, "
            f"ridurre gap_slots o abbassare le frazioni."
        )

    bounds = {
        "train": (0, train_end),
        "val": (val_start, val_end),
        "test": (test_start, n_slots),
    }

    sample_starts: dict[str, tuple[int, ...]] = {}
    for name, (start, end) in bounds.items():
        last_start = end - total_window
        sample_starts[name] = tuple(range(start, last_start + 1)) if last_start >= start else ()

    empty = sorted(name for name, starts in sample_starts.items() if not starts)
    if empty:
        raise ValueError(
            f"Gli split {empty} non contengono alcun campione completo di {total_window} "
            f"slot. Blocchi calcolati: {bounds}. Allungare il periodo oppure ridurre "
            f"input_slots/output_slots."
        )

    return SplitLayout(
        bounds=bounds,
        sample_starts=sample_starts,
        input_slots=input_slots,
        output_slots=output_slots,
    )


def split_labels(n_slots: int, layout: SplitLayout) -> np.ndarray:
    """Etichetta di split per ogni slot; ``"gap"`` per gli slot scartati alle giunzioni."""
    labels = np.full(n_slots, GAP_LABEL, dtype=object)
    for name, (start, end) in layout.bounds.items():
        labels[start:end] = name
    return labels


# --------------------------------------------------------------------------- #
# Codifica temporale
# --------------------------------------------------------------------------- #


def time_encoding(times: Sequence[datetime], slot_hours: Sequence[int]) -> np.ndarray:
    """Codifica ciclica del tempo: ``(n_slots, 4)`` con seno/coseno di anno e giorno.

    Il modello e' convoluzionale sullo spazio e non ha altro modo di sapere in che
    stagione o a che ora del giorno si trovi: senza questi canali confonderebbe il
    raffreddamento notturno con il passaggio di un fronte freddo.
    """
    if not times:
        return np.zeros((0, 4), dtype=np.float32)
    slots_per_day = len(slot_hours)
    encoded = np.empty((len(times), 4), dtype=np.float32)
    for index, moment in enumerate(times):
        days_in_year = 366 if calendar.isleap(moment.year) else 365
        year_angle = 2.0 * np.pi * (moment.timetuple().tm_yday - 1) / days_in_year
        day_angle = 2.0 * np.pi * slot_of_day(moment, slot_hours) / slots_per_day
        encoded[index] = (
            np.sin(year_angle),
            np.cos(year_angle),
            np.sin(day_angle),
            np.cos(day_angle),
        )
    return encoded
