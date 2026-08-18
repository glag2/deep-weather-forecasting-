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
from datetime import UTC, date, datetime, timedelta
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


def parse_month(value: str) -> tuple[int, int]:
    """Interpreta una stringa `YYYY-MM` come coppia (anno, mese).

    Usata dalle interfacce a riga di comando per delimitare le ondate di download e di
    ingestione, dove un mese scritto male deve fallire subito e non a metà di un
    trasferimento da ore.
    """
    parti = value.split("-")
    if len(parti) != 2:
        raise ValueError(f"Mese non valido: {value!r}, atteso YYYY-MM")
    try:
        year, month = int(parti[0]), int(parti[1])
    except ValueError:
        raise ValueError(f"Mese non valido: {value!r}, atteso YYYY-MM") from None
    if not 1 <= month <= 12:
        raise ValueError(f"Mese fuori intervallo: {value!r}")
    return year, month


def months_between(start: date, end: date) -> list[tuple[int, int]]:
    """Coppie (anno, mese) toccate dall'intervallo, estremi inclusi."""
    if start > end:
        raise ValueError(f"start ({start}) successivo a end ({end})")
    out: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        out.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def days_for_month(year: int, month: int, start: date, end: date) -> list[int]:
    """Giorni del mese che ricadono nell'intervallo richiesto.

    Serve perche' il periodo non comincia ne' finisce necessariamente a confine di
    mese: il primo e l'ultimo mese sono parziali, e chiedere al CDS giorni fuori
    dall'intervallo scaricherebbe dati che poi verrebbero scartati, oppure giorni non
    ancora pubblicati.
    """
    n_days = calendar.monthrange(year, month)[1]
    # Intersezione fra il mese e l'intervallo: gestisce in un colpo i mesi parziali,
    # quelli interni interi e quelli completamente fuori intervallo.
    first_day = max(start, date(year, month, 1))
    last_day = min(end, date(year, month, n_days))
    if first_day > last_day:
        return []
    return list(range(first_day.day, last_day.day + 1))


def slot_times_between(
    start: date, end: date, slot_hours: Sequence[int]
) -> list[datetime]:
    """Tutti gli istanti di slot nell'intervallo di date, estremi inclusi."""
    out: list[datetime] = []
    for year, month in months_between(start, end):
        for day in days_for_month(year, month, start, end):
            for hour in sorted(slot_hours):
                out.append(datetime(year, month, day, hour, tzinfo=UTC))
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


def advance_slots(moment: datetime, steps: int, slot_hours: Sequence[int]) -> datetime:
    """L'istante che si raggiunge avanzando di `steps` slot da `moment`.

    Gli slot non sono equidistanti: con 06, 12 e 18 UTC gli intervalli sono di 6, 6 e 12
    ore, quindi moltiplicare per una durata media sbaglierebbe. Serve perche' gli istanti
    di una previsione **non sono nello store**: sono nel futuro, e leggerli da li' e'
    possibile solo finche' si verifica il passato.
    """
    ordinate = sorted(slot_hours)
    if steps < 0:
        raise ValueError(f"steps deve essere non negativo: {steps}")
    posizione = slot_of_day(moment, ordinate)
    totale = posizione + steps
    giorni, resto = divmod(totale, len(ordinate))
    return datetime(
        moment.year, moment.month, moment.day, ordinate[resto], tzinfo=moment.tzinfo
    ) + timedelta(days=giorni)


def diurnal_reference_index(lead_index: int, input_slots: int, slots_per_day: int) -> int:
    """Posizione, dentro la finestra, dell'ultima osservazione alla stessa ora del bersaglio.

    Gli slot non sono equidistanti: con 06Z, 12Z e 18Z le distanze sono 6, 6 e 12 ore,
    quindi solo un multiplo di ``slots_per_day`` corrisponde a un numero intero di
    giorni. Tornando indietro di giorni interi a partire dal bersaglio si ottiene
    un'osservazione alla sua stessa ora, che cade sempre nella parte gia' osservata
    della finestra ed e' quindi disponibile al momento della previsione.

    ``lead_index`` e' la scadenza contata da zero fra gli slot da prevedere.
    """
    if lead_index < 0:
        raise ValueError(f"Scadenza negativa: {lead_index}")
    if slots_per_day <= 0:
        raise ValueError(f"slots_per_day deve essere positivo, ricevuto {slots_per_day}")
    giorni_indietro = -(-(lead_index + 1) // slots_per_day)
    indice = input_slots + lead_index - giorni_indietro * slots_per_day
    if indice < 0:
        raise ValueError(
            f"La finestra osservata di {input_slots} slot non arriva abbastanza indietro "
            f"per la scadenza {lead_index}: servirebbe l'indice {indice}"
        )
    # L'aritmetica sopra garantisce indice < input_slots, perche' `giorni_indietro`
    # arrotonda per eccesso e quindi vale almeno `lead_index + 1` slot. Il controllo
    # resta perche' questa e' la riga che separa un riferimento legittimo da una
    # lettura del futuro: se qualcuno cambiasse la formula, il modello si limiterebbe
    # a diventare misteriosamente bravo invece di fallire.
    if indice >= input_slots:
        raise ValueError(
            f"Il riferimento per la scadenza {lead_index} cadrebbe all'indice {indice}, "
            f"fuori dai {input_slots} slot osservati: sarebbe il futuro"
        )
    return indice


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

    return layout_from_bounds(bounds, input_slots=input_slots, output_slots=output_slots)


def layout_from_bounds(
    bounds: dict[str, tuple[int, int]], *, input_slots: int, output_slots: int
) -> SplitLayout:
    """Enumera i campioni ammessi in ciascun blocco.

    Un campione e' ammesso solo se l'intera finestra input+target ricade nel proprio
    blocco: e' questo vincolo, non il gap, a impedire che un target di train compaia
    tra gli input di validation.
    """
    total_window = input_slots + output_slots
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


def shift_layout(layout: SplitLayout, offset: int) -> SplitLayout:
    """Sposta in avanti di ``offset`` slot tutti i blocchi di un layout.

    Serve quando la parte iniziale dell'archivio e' tenuta fuori dall'addestramento: i
    fold vanno calcolati sulla lunghezza del tratto *disponibile*, altrimenti il primo
    fold cadrebbe dentro il periodo escluso e resterebbe senza campioni, ma gli indici
    scritti nelle tabelle devono restare quelli dell'archivio intero.
    """
    if offset < 0:
        raise ValueError(f"Lo scostamento non puo' essere negativo: {offset}")
    if offset == 0:
        return layout
    return layout_from_bounds(
        {
            nome: (inizio + offset, fine + offset)
            for nome, (inizio, fine) in layout.bounds.items()
        },
        input_slots=layout.input_slots,
        output_slots=layout.output_slots,
    )


def max_rolling_folds(
    n_slots: int,
    *,
    initial_train_slots: int,
    val_slots: int,
    test_slots: int,
    gap_slots: int,
    step_slots: int,
) -> int:
    """Quanti fold entrano nel periodo disponibile."""
    if step_slots <= 0:
        raise ValueError(f"step_slots deve essere positivo: {step_slots}")
    horizon = gap_slots + val_slots + gap_slots + test_slots
    available = n_slots - initial_train_slots - horizon
    if available < 0:
        return 0
    return 1 + available // step_slots


def build_rolling_folds(
    n_slots: int,
    *,
    initial_train_slots: int,
    val_slots: int,
    test_slots: int,
    gap_slots: int,
    step_slots: int,
    input_slots: int,
    output_slots: int,
    expanding: bool = True,
    n_folds: int | None = None,
) -> list[SplitLayout]:
    """Validazione a finestra mobile (rolling origin) su fold cronologici.

    Uno split unico in tre blocchi contigui concentra il test nella coda del periodo,
    che quindi copre una sola stagione: le metriche misurerebbero il modello su un solo
    regime meteorologico. Qui l'origine avanza di ``step_slots`` a ogni fold, cosi' i
    blocchi di test scorrono nel tempo e insieme coprono tutte le stagioni, mentre
    dentro ogni fold l'ordine train -> val -> test resta rispettato.

    Con ``expanding=True`` il train cresce a ogni fold e usa tutta la storia
    disponibile; con ``expanding=False`` scorre a lunghezza costante, utile per
    verificare se il modello dipende dalla quantita' di storia o dalla sua vicinanza
    temporale.
    """
    disponibili = max_rolling_folds(
        n_slots,
        initial_train_slots=initial_train_slots,
        val_slots=val_slots,
        test_slots=test_slots,
        gap_slots=gap_slots,
        step_slots=step_slots,
    )
    if disponibili == 0:
        raise ValueError(
            f"Con {n_slots} slot non entra nemmeno un fold: servono almeno "
            f"{initial_train_slots + 2 * gap_slots + val_slots + test_slots} slot. "
            f"Allungare il periodo o ridurre initial_train/val/test."
        )
    if n_folds is None:
        n_folds = disponibili
    elif n_folds > disponibili:
        raise ValueError(
            f"Richiesti {n_folds} fold ma nel periodo ne entrano {disponibili}."
        )

    folds: list[SplitLayout] = []
    for index in range(n_folds):
        train_end = initial_train_slots + index * step_slots
        train_start = 0 if expanding else train_end - initial_train_slots
        val_start = train_end + gap_slots
        val_end = val_start + val_slots
        test_start = val_end + gap_slots
        test_end = test_start + test_slots
        folds.append(
            layout_from_bounds(
                {
                    "train": (train_start, train_end),
                    "val": (val_start, val_end),
                    "test": (test_start, test_end),
                },
                input_slots=input_slots,
                output_slots=output_slots,
            )
        )
    return folds


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
