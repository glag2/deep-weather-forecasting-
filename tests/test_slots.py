"""Test dell'algebra degli slot temporali.

Le finestre scorrevoli e gli split sono la parte piu' facile da sbagliare in
silenzio: un off-by-one sposta i target di uno slot senza far fallire nulla, e il
modello impara a prevedere l'istante sbagliato. Questi test bloccano le invarianti.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pytest

from dwf.slots import (
    GAP_LABEL,
    accumulation_coverage,
    accumulation_hours,
    accumulation_offsets,
    build_rolling_folds,
    build_split_layout,
    days_for_month,
    expected_slot_times,
    expected_steps,
    find_gaps,
    max_rolling_folds,
    month_slot_times,
    months_between,
    parse_month,
    required_hours,
    slot_of_day,
    slot_times_between,
    split_labels,
    time_encoding,
    validate_accumulation_fits,
)

SLOT_HOURS = [6, 12, 18]


# --------------------------------------------------------------------------- #
# Finestre di accumulo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("window_hours", range(1, 25))
def test_offsets_hanno_lunghezza_richiesta(window_hours: int) -> None:
    assert len(accumulation_offsets(window_hours)) == window_hours


@pytest.mark.parametrize("window_hours", range(1, 25))
def test_offsets_sono_consecutivi(window_hours: int) -> None:
    offsets = accumulation_offsets(window_hours)
    assert offsets == list(range(offsets[0], offsets[-1] + 1))


def test_offsets_dispari_sono_simmetrici() -> None:
    assert accumulation_offsets(7) == [-3, -2, -1, 0, 1, 2, 3]


def test_offsets_pari_spostati_di_mezzora_avanti() -> None:
    assert accumulation_offsets(8) == [-3, -2, -1, 0, 1, 2, 3, 4]


@pytest.mark.parametrize("window_hours", [0, -1, 25])
def test_offsets_rifiutano_finestre_fuori_range(window_hours: int) -> None:
    with pytest.raises(ValueError, match="window_hours"):
        accumulation_offsets(window_hours)


def test_ore_di_accumulo_coprono_intervallo_atteso() -> None:
    # Con la convenzione ERA5 le ore 3..10 accumulano su (2, 10], centrato su 06.
    assert accumulation_hours(6, 8) == [3, 4, 5, 6, 7, 8, 9, 10]
    assert accumulation_hours(18, 8) == [15, 16, 17, 18, 19, 20, 21, 22]


@pytest.mark.parametrize("slot_hour", [0, 1, 2, 21, 22, 23])
def test_finestra_non_puo_sconfinare_dal_giorno(slot_hour: int) -> None:
    with pytest.raises(ValueError, match="fuori dall'intervallo"):
        accumulation_hours(slot_hour, 8)


def test_validate_accumulation_fits_accetta_configurazione_di_default() -> None:
    validate_accumulation_fits(SLOT_HOURS, 8)


def test_validate_accumulation_fits_rifiuta_slot_notturno() -> None:
    with pytest.raises(ValueError):
        validate_accumulation_fits([0, 12, 18], 8)


def test_ore_richieste_sono_ordinate_e_univoche() -> None:
    hours = required_hours(SLOT_HOURS, 8)
    assert hours == sorted(set(hours))
    assert hours == list(range(3, 23))


def test_copertura_dichiara_ore_perse_e_ridondanti() -> None:
    covered, overlapping = accumulation_coverage(SLOT_HOURS, 8)
    assert covered == 20  # le ore 23, 0, 1, 2 non entrano in nessun target
    assert overlapping == 4


def test_finestra_da_sei_ore_non_ha_sovrapposizioni() -> None:
    covered, overlapping = accumulation_coverage(SLOT_HOURS, 6)
    assert overlapping == 0
    assert covered == 18


# --------------------------------------------------------------------------- #
# Sequenze di slot
# --------------------------------------------------------------------------- #


def test_slot_mensili_contano_giorni_reali() -> None:
    assert len(month_slot_times(2024, 2, SLOT_HOURS)) == 29 * 3  # bisestile
    assert len(month_slot_times(2025, 2, SLOT_HOURS)) == 28 * 3


def test_slot_mensili_sono_ordinati_e_in_utc() -> None:
    times = month_slot_times(2025, 3, SLOT_HOURS)
    assert times == sorted(times)
    assert all(moment.tzinfo is UTC for moment in times)
    assert {moment.hour for moment in times} == set(SLOT_HOURS)


def test_slot_attesi_concatenano_i_mesi_in_ordine() -> None:
    times = expected_slot_times([(2024, 12), (2025, 1)], SLOT_HOURS)
    assert len(times) == (31 + 31) * 3
    assert times == sorted(times)
    assert times[0] == datetime(2024, 12, 1, 6, tzinfo=UTC)
    assert times[-1] == datetime(2025, 1, 31, 18, tzinfo=UTC)


def test_slot_of_day_mappa_ora_su_posizione() -> None:
    assert slot_of_day(datetime(2025, 1, 1, 6, tzinfo=UTC), SLOT_HOURS) == 0
    assert slot_of_day(datetime(2025, 1, 1, 18, tzinfo=UTC), SLOT_HOURS) == 2


def test_slot_of_day_rifiuta_ora_non_configurata() -> None:
    with pytest.raises(ValueError, match="non e' uno slot configurato"):
        slot_of_day(datetime(2025, 1, 1, 7, tzinfo=UTC), SLOT_HOURS)


def test_passi_attesi_chiudono_il_giro_sulle_24_ore() -> None:
    steps = expected_steps(SLOT_HOURS)
    assert steps == [6, 6, 12]
    assert sum(steps) == 24


def test_nessun_buco_su_serie_completa() -> None:
    times = expected_slot_times([(2025, 1)], SLOT_HOURS)
    assert find_gaps(times, SLOT_HOURS) == []


def test_buco_rilevato_quando_manca_uno_slot() -> None:
    times = expected_slot_times([(2025, 1)], SLOT_HOURS)
    incomplete = times[:4] + times[5:]
    gaps = find_gaps(incomplete, SLOT_HOURS)
    assert len(gaps) == 1
    assert gaps[0] == (times[3], times[5])


def test_serie_troppo_corta_non_ha_buchi() -> None:
    assert find_gaps([], SLOT_HOURS) == []
    assert find_gaps([datetime(2025, 1, 1, 6, tzinfo=UTC)], SLOT_HOURS) == []


# --------------------------------------------------------------------------- #
# Split temporali
# --------------------------------------------------------------------------- #

SPLIT_KWARGS = {
    "train_fraction": 0.7,
    "val_fraction": 0.15,
    "gap_slots": 30,
    "input_slots": 21,
    "output_slots": 9,
}


def test_split_su_due_anni_produce_blocchi_ordinati() -> None:
    layout = build_split_layout(2190, **SPLIT_KWARGS)
    train_start, train_end = layout.bounds["train"]
    val_start, val_end = layout.bounds["val"]
    test_start, test_end = layout.bounds["test"]
    assert (train_start, train_end) == (0, 1533)
    assert val_start == train_end + 30
    assert test_start == val_end + 30
    assert test_end == 2190


def test_finestre_non_attraversano_i_confini_di_split() -> None:
    layout = build_split_layout(2190, **SPLIT_KWARGS)
    for split, (start, end) in layout.bounds.items():
        for sample_start in layout.sample_starts[split]:
            assert start <= sample_start
            assert sample_start + layout.total_window <= end


def test_nessun_indice_condiviso_tra_split() -> None:
    layout = build_split_layout(2190, **SPLIT_KWARGS)
    used: dict[str, set[int]] = {}
    for split in ("train", "val", "test"):
        covered: set[int] = set()
        for sample_start in layout.sample_starts[split]:
            covered.update(range(sample_start, sample_start + layout.total_window))
        used[split] = covered
    assert not used["train"] & used["val"]
    assert not used["val"] & used["test"]
    assert not used["train"] & used["test"]


def test_finestra_totale_e_somma_di_input_e_output() -> None:
    layout = build_split_layout(2190, **SPLIT_KWARGS)
    assert layout.total_window == 30
    assert layout.input_slots == 21
    assert layout.output_slots == 9


def test_split_rifiuta_serie_piu_corta_di_una_finestra() -> None:
    with pytest.raises(ValueError, match="almeno 30 slot"):
        build_split_layout(29, **SPLIT_KWARGS)


def test_split_rifiuta_periodo_che_azzera_il_test() -> None:
    with pytest.raises(ValueError, match="test resterebbe vuoto"):
        build_split_layout(200, **SPLIT_KWARGS)


def test_split_rifiuta_blocchi_senza_campioni_completi() -> None:
    # Val abbastanza grande da esistere ma piu' corto della finestra richiesta.
    with pytest.raises(ValueError, match="non contengono alcun campione"):
        build_split_layout(
            400,
            train_fraction=0.9,
            val_fraction=0.02,
            gap_slots=1,
            input_slots=21,
            output_slots=9,
        )


def test_etichette_marcano_i_gap() -> None:
    layout = build_split_layout(2190, **SPLIT_KWARGS)
    labels = split_labels(2190, layout)
    assert len(labels) == 2190
    assert int((labels == GAP_LABEL).sum()) == 60  # due giunzioni da 30 slot
    assert labels[0] == "train"
    assert labels[-1] == "test"


# --------------------------------------------------------------------------- #
# Codifica temporale
# --------------------------------------------------------------------------- #


def test_encoding_ha_quattro_canali_normalizzati() -> None:
    times = expected_slot_times([(2025, 6)], SLOT_HOURS)
    encoded = time_encoding(times, SLOT_HOURS)
    assert encoded.shape == (len(times), 4)
    assert encoded.dtype == np.float32
    assert np.all(np.abs(encoded) <= 1.0 + 1e-6)


def test_encoding_e_ciclico_sul_giorno() -> None:
    times = expected_slot_times([(2025, 6)], SLOT_HOURS)
    encoded = time_encoding(times, SLOT_HOURS)
    # I canali giornalieri (2, 3) si ripetono a ogni giorno.
    np.testing.assert_allclose(encoded[0, 2:], encoded[3, 2:], atol=1e-6)
    assert not np.allclose(encoded[0, 2:], encoded[1, 2:])


def test_encoding_distingue_le_stagioni() -> None:
    gennaio = time_encoding([datetime(2025, 1, 15, 12, tzinfo=UTC)], SLOT_HOURS)
    luglio = time_encoding([datetime(2025, 7, 15, 12, tzinfo=UTC)], SLOT_HOURS)
    assert not np.allclose(gennaio[0, :2], luglio[0, :2])
    # Stessa data di anni diversi deve dare quasi la stessa codifica annuale.
    altro_anno = time_encoding([datetime(2027, 1, 15, 12, tzinfo=UTC)], SLOT_HOURS)
    np.testing.assert_allclose(gennaio[0, :2], altro_anno[0, :2], atol=1e-3)


def test_encoding_su_serie_vuota() -> None:
    assert time_encoding([], SLOT_HOURS).shape == (0, 4)


# --------------------------------------------------------------------------- #
# Intervalli a granularita' di giorno
# --------------------------------------------------------------------------- #


def test_mesi_tra_due_date_includono_gli_estremi() -> None:
    mesi = months_between(date(2024, 1, 15), date(2026, 8, 11))
    assert mesi[0] == (2024, 1)
    assert mesi[-1] == (2026, 8)
    assert len(mesi) == 32
    assert mesi == sorted(mesi)


def test_mesi_tra_due_date_nello_stesso_mese() -> None:
    assert months_between(date(2025, 5, 3), date(2025, 5, 27)) == [(2025, 5)]


def test_mesi_con_start_dopo_end_sono_rifiutati() -> None:
    with pytest.raises(ValueError, match="successivo a end"):
        months_between(date(2025, 5, 1), date(2025, 4, 30))


def test_giorni_del_primo_e_ultimo_mese_sono_troncati() -> None:
    start, end = date(2024, 1, 20), date(2024, 3, 5)
    assert days_for_month(2024, 1, start, end) == list(range(20, 32))
    assert days_for_month(2024, 2, start, end) == list(range(1, 30))  # 2024 bisestile
    assert days_for_month(2024, 3, start, end) == [1, 2, 3, 4, 5]


def test_giorni_di_un_mese_fuori_intervallo_sono_vuoti() -> None:
    start, end = date(2024, 1, 20), date(2024, 1, 25)
    assert days_for_month(2024, 2, start, end) == []


def test_giorni_di_mese_unico_sono_limitati_a_entrambi_gli_estremi() -> None:
    assert days_for_month(2025, 6, date(2025, 6, 10), date(2025, 6, 12)) == [10, 11, 12]


def test_slot_nell_intervallo_rispettano_estremi_e_ordine() -> None:
    times = slot_times_between(date(2025, 3, 30), date(2025, 4, 2), [6, 12, 18])
    assert len(times) == 4 * 3
    assert times == sorted(times)
    assert times[0] == datetime(2025, 3, 30, 6, tzinfo=UTC)
    assert times[-1] == datetime(2025, 4, 2, 18, tzinfo=UTC)
    assert all(t.tzinfo is UTC for t in times)


def test_slot_nell_intervallo_coincidono_col_mese_intero() -> None:
    """Su un mese completo le due funzioni devono dare lo stesso risultato."""
    per_mese = month_slot_times(2025, 4, [6, 12, 18])
    per_intervallo = slot_times_between(date(2025, 4, 1), date(2025, 4, 30), [6, 12, 18])
    assert per_intervallo == per_mese


# --------------------------------------------------------------------------- #
# Validazione a finestra mobile
# --------------------------------------------------------------------------- #

ROLLING = {
    "initial_train_slots": 990,
    "val_slots": 180,
    "test_slots": 270,
    "gap_slots": 30,
    "step_slots": 270,
    "input_slots": 21,
    "output_slots": 9,
}


def test_numero_di_fold_coerente_col_periodo() -> None:
    args = {k: v for k, v in ROLLING.items() if k not in ("input_slots", "output_slots")}
    atteso = max_rolling_folds(2862, **args)
    folds = build_rolling_folds(2862, **ROLLING)
    assert len(folds) == atteso == 6


def test_ogni_fold_rispetta_ordine_cronologico_e_gap() -> None:
    """In ogni fold il train precede val, che precede test, con il gap richiesto."""
    for fold in build_rolling_folds(2862, **ROLLING):
        train_start, train_end = fold.bounds["train"]
        val_start, val_end = fold.bounds["val"]
        test_start, test_end = fold.bounds["test"]
        assert train_start < train_end
        assert val_start - train_end == ROLLING["gap_slots"]
        assert test_start - val_end == ROLLING["gap_slots"]
        assert val_end - val_start == ROLLING["val_slots"]
        assert test_end - test_start == ROLLING["test_slots"]


def test_i_campioni_di_un_fold_non_attraversano_i_blocchi() -> None:
    total = ROLLING["input_slots"] + ROLLING["output_slots"]
    for fold in build_rolling_folds(2862, **ROLLING):
        for nome, (start, end) in fold.bounds.items():
            for inizio in fold.sample_starts[nome]:
                assert inizio >= start
                assert inizio + total <= end


def test_nessuna_sovrapposizione_tra_split_dello_stesso_fold() -> None:
    for fold in build_rolling_folds(2862, **ROLLING):
        intervalli = [set(range(*fold.bounds[n])) for n in ("train", "val", "test")]
        assert not intervalli[0] & intervalli[1]
        assert not intervalli[1] & intervalli[2]
        assert not intervalli[0] & intervalli[2]


def test_i_test_dei_fold_si_susseguono_senza_buchi() -> None:
    """Con step pari a test_slots i blocchi di test coprono il periodo senza lacune."""
    folds = build_rolling_folds(2862, **ROLLING)
    inizi = [f.bounds["test"][0] for f in folds]
    fini = [f.bounds["test"][1] for f in folds]
    assert inizi == sorted(inizi)
    for precedente, successivo in zip(fini[:-1], inizi[1:], strict=True):
        assert successivo == precedente


def test_train_espansivo_parte_sempre_da_zero() -> None:
    folds = build_rolling_folds(2862, expanding=True, **ROLLING)
    assert all(f.bounds["train"][0] == 0 for f in folds)
    lunghezze = [f.bounds["train"][1] - f.bounds["train"][0] for f in folds]
    assert lunghezze == sorted(lunghezze)
    assert lunghezze[0] < lunghezze[-1]


def test_train_scorrevole_mantiene_lunghezza_costante() -> None:
    folds = build_rolling_folds(2862, expanding=False, **ROLLING)
    lunghezze = {f.bounds["train"][1] - f.bounds["train"][0] for f in folds}
    assert lunghezze == {ROLLING["initial_train_slots"]}
    assert folds[0].bounds["train"][0] == 0
    assert folds[-1].bounds["train"][0] > 0


def test_periodo_troppo_corto_per_un_fold_e_rifiutato() -> None:
    with pytest.raises(ValueError, match="non entra nemmeno un fold"):
        build_rolling_folds(500, **ROLLING)


def test_numero_di_fold_eccessivo_e_rifiutato() -> None:
    with pytest.raises(ValueError, match="ne entrano"):
        build_rolling_folds(2862, n_folds=99, **ROLLING)


def test_numero_di_fold_limitato_su_richiesta() -> None:
    folds = build_rolling_folds(2862, n_folds=2, **ROLLING)
    assert len(folds) == 2


def test_passo_non_positivo_e_rifiutato() -> None:
    args = {**ROLLING, "step_slots": 0}
    with pytest.raises(ValueError, match="step_slots deve essere positivo"):
        build_rolling_folds(2862, **args)


# --------------------------------------------------------------------------- #
# Interpretazione dei mesi da riga di comando
# --------------------------------------------------------------------------- #


def test_mese_valido_viene_interpretato() -> None:
    assert parse_month("2025-01") == (2025, 1)
    assert parse_month("2026-12") == (2026, 12)


@pytest.mark.parametrize("valore", ["2025", "2025-01-01", "", "-"])
def test_forma_sbagliata_e_rifiutata(valore: str) -> None:
    with pytest.raises(ValueError, match="Mese non valido"):
        parse_month(valore)


@pytest.mark.parametrize("valore", ["duemila-01", "2025-gennaio"])
def test_valori_non_numerici_sono_rifiutati(valore: str) -> None:
    with pytest.raises(ValueError, match="Mese non valido"):
        parse_month(valore)


@pytest.mark.parametrize("valore", ["2025-00", "2025-13"])
def test_mese_fuori_intervallo_e_rifiutato(valore: str) -> None:
    """Un mese assurdo deve fallire subito, non a meta' di un download da ore."""
    with pytest.raises(ValueError, match="fuori intervallo"):
        parse_month(valore)
