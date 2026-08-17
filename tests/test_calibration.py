"""Test della calibrazione delle probabilita' e delle metriche di decisione.

Una calibrazione sbagliata non fa fallire nulla: produce numeri plausibili e
sistematicamente falsi. I test verificano quindi le proprieta' che la definiscono,
non i valori: monotonia, idempotenza su dati gia' calibrati, riduzione effettiva
dell'errore su una distorsione nota, e il fatto che l'ordinamento appreso dalla rete
non venga mai invertito.
"""

from __future__ import annotations

import numpy as np
import pytest

from dwf.calibration import (
    MIN_SAMPLES,
    CalibrationError,
    ProbabilityCalibrator,
    calibration_error,
    fit_calibrator,
    pool_adjacent_violators,
)
from dwf.evaluate import best_f1_threshold, classification_score

# --------------------------------------------------------------------------- #
# Pool adjacent violators
# --------------------------------------------------------------------------- #


def test_una_sequenza_gia_crescente_non_cambia() -> None:
    valori = np.array([0.1, 0.2, 0.5, 0.9])
    pesi = np.ones(4)
    assert pool_adjacent_violators(valori, pesi) == pytest.approx(valori)


def test_una_violazione_viene_mediata() -> None:
    # 0.8 e 0.2 violano la monotonia: entrambi diventano la loro media.
    risultato = pool_adjacent_violators(np.array([0.1, 0.8, 0.2]), np.ones(3))
    assert risultato == pytest.approx([0.1, 0.5, 0.5])


def test_il_risultato_e_sempre_non_decrescente() -> None:
    generatore = np.random.default_rng(0)
    for _ in range(20):
        valori = generatore.random(50)
        risultato = pool_adjacent_violators(valori, np.ones(50))
        assert np.all(np.diff(risultato) >= -1e-12)


def test_la_media_pesata_si_conserva() -> None:
    """La regressione isotonica ridistribuisce, non crea ne' distrugge massa."""
    valori = np.array([0.9, 0.1, 0.5, 0.3])
    pesi = np.array([1.0, 3.0, 2.0, 4.0])
    risultato = pool_adjacent_violators(valori, pesi)
    assert np.average(risultato, weights=pesi) == pytest.approx(np.average(valori, weights=pesi))


def test_i_pesi_contano():
    """Un blocco con peso grande tira la media verso di se'."""
    leggero = pool_adjacent_violators(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    pesante = pool_adjacent_violators(np.array([1.0, 0.0]), np.array([1.0, 9.0]))
    assert leggero[0] == pytest.approx(0.5)
    assert pesante[0] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Stima della mappa
# --------------------------------------------------------------------------- #


def dati_distorti(n: int = 20000, seme: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Probabilita' dichiarate distorte, con esiti generati dalla probabilita' vera.

    La distorsione e' un elevamento a potenza: conserva l'ordinamento ma sbaglia la
    scala, che e' esattamente il difetto osservato nel modello reale.
    """
    generatore = np.random.default_rng(seme)
    vera = generatore.random(n)
    esiti = (generatore.random(n) < vera).astype(np.float64)
    dichiarata = vera**2
    return dichiarata, esiti


def test_la_mappa_e_monotona() -> None:
    calibratore = fit_calibrator(*dati_distorti())
    assert np.all(np.diff(calibratore.knots_out) >= -1e-12)


def test_la_calibrazione_riduce_l_errore() -> None:
    dichiarata, esiti = dati_distorti()
    prima = calibration_error(dichiarata, esiti)
    dopo = calibration_error(fit_calibrator(dichiarata, esiti).apply(dichiarata), esiti)
    assert dopo < prima / 2


def test_la_calibrazione_generalizza_a_dati_nuovi() -> None:
    """Il controllo che conta: la mappa deve valere su dati che non l'hanno prodotta."""
    addestramento = dati_distorti(seme=1)
    verifica = dati_distorti(seme=2)
    calibratore = fit_calibrator(*addestramento)
    prima = calibration_error(*verifica)
    dopo = calibration_error(calibratore.apply(verifica[0]), verifica[1])
    assert dopo < prima / 2


def test_dati_gia_calibrati_restano_quasi_invariati() -> None:
    generatore = np.random.default_rng(3)
    vera = generatore.random(20000)
    esiti = (generatore.random(20000) < vera).astype(np.float64)
    corrette = fit_calibrator(vera, esiti).apply(vera)
    # Nessuna correzione sistematica: lo scarto medio resta piccolo.
    assert np.abs(corrette - vera).mean() < 0.02


def test_l_ordinamento_non_viene_mai_invertito():
    """La rete ordina bene: la calibrazione corregge la scala, non il giudizio."""
    dichiarata, esiti = dati_distorti()
    corrette = fit_calibrator(dichiarata, esiti).apply(dichiarata)
    ordine = np.argsort(dichiarata)
    assert np.all(np.diff(corrette[ordine]) >= -1e-6)


def test_le_probabilita_corrette_restano_in_zero_uno() -> None:
    dichiarata, esiti = dati_distorti()
    corrette = fit_calibrator(dichiarata, esiti).apply(np.linspace(0.0, 1.0, 101))
    assert corrette.min() >= 0.0
    assert corrette.max() <= 1.0


def test_troppo_pochi_casi_vengono_rifiutati() -> None:
    with pytest.raises(CalibrationError, match="almeno"):
        fit_calibrator(np.full(MIN_SAMPLES - 1, 0.5), np.zeros(MIN_SAMPLES - 1))


def test_probabilita_costanti_vengono_rifiutate() -> None:
    """Senza variabilita' non c'e' mappa da stimare, e fingerne una sarebbe peggio."""
    with pytest.raises(CalibrationError, match="costanti"):
        fit_calibrator(np.full(1000, 0.4), np.zeros(1000))


def test_lunghezze_diverse_vengono_rifiutate() -> None:
    with pytest.raises(CalibrationError, match="lunghezza diversa"):
        fit_calibrator(np.zeros(200), np.zeros(100))


def test_i_valori_non_finiti_vengono_esclusi() -> None:
    dichiarata, esiti = dati_distorti(n=1000)
    dichiarata[:10] = np.nan
    calibratore = fit_calibrator(dichiarata, esiti)
    assert calibratore.n_samples == 990


# --------------------------------------------------------------------------- #
# Persistenza su tabella
# --------------------------------------------------------------------------- #


def test_la_mappa_sopravvive_al_giro_su_tabella() -> None:
    originale = fit_calibrator(*dati_distorti(n=2000))
    ricostruito = ProbabilityCalibrator.from_table(originale.to_table(fold=0))
    campione = np.linspace(0.0, 1.0, 50)
    assert ricostruito.apply(campione) == pytest.approx(originale.apply(campione))
    assert ricostruito.fitted_on_split == "val"


def test_una_variabile_assente_e_segnalata() -> None:
    tabella = fit_calibrator(*dati_distorti(n=2000)).to_table(fold=0)
    with pytest.raises(CalibrationError, match="sf"):
        ProbabilityCalibrator.from_table(tabella, variable="sf")


# --------------------------------------------------------------------------- #
# Metriche di decisione
# --------------------------------------------------------------------------- #


def test_una_previsione_perfetta_ha_f1_uno() -> None:
    esiti = np.array([0.0, 1.0, 1.0, 0.0])
    punteggio = classification_score(esiti, esiti)
    assert punteggio.f1 == pytest.approx(1.0)
    assert punteggio.accuracy == pytest.approx(1.0)


def test_precisione_e_richiamo_sono_distinti() -> None:
    # Prevede sempre l'evento: richiamo pieno, precisione pari alla frequenza di base.
    probabilita = np.ones(10)
    esiti = np.array([1.0] * 3 + [0.0] * 7)
    punteggio = classification_score(probabilita, esiti)
    assert punteggio.recall == pytest.approx(1.0)
    assert punteggio.precision == pytest.approx(0.3)


def test_non_prevedere_mai_l_evento_annulla_l_f1() -> None:
    punteggio = classification_score(np.zeros(10), np.array([1.0] * 3 + [0.0] * 7))
    assert punteggio.f1 == pytest.approx(0.0)
    assert punteggio.n_positive_predicted == 0


def test_la_soglia_sposta_il_compromesso() -> None:
    probabilita = np.linspace(0.0, 1.0, 100)
    esiti = (probabilita > 0.7).astype(np.float64)
    prudente = classification_score(probabilita, esiti, threshold=0.9)
    generoso = classification_score(probabilita, esiti, threshold=0.3)
    assert prudente.precision > generoso.precision
    assert generoso.recall > prudente.recall


def test_la_soglia_ottimale_batte_quella_convenzionale() -> None:
    """Su un evento con frequenza lontana da un mezzo, 0,5 non e' l'ottimo."""
    generatore = np.random.default_rng(5)
    vera = generatore.beta(2, 8, size=20000)
    esiti = (generatore.random(20000) < vera).astype(np.float64)
    soglia, f1_ottimo = best_f1_threshold(vera, esiti)
    assert f1_ottimo >= classification_score(vera, esiti, threshold=0.5).f1
    assert 0.0 < soglia < 1.0


def test_senza_eventi_osservati_il_richiamo_non_e_un_numero() -> None:
    """Meglio dichiarare l'indefinito che restituire zero e farlo sembrare un risultato."""
    punteggio = classification_score(np.full(10, 0.6), np.zeros(10))
    assert np.isnan(punteggio.recall)
    assert np.isnan(punteggio.f1)


# --------------------------------------------------------------------------- #
# Errore di calibrazione
# --------------------------------------------------------------------------- #


def test_una_previsione_calibrata_ha_errore_quasi_nullo() -> None:
    generatore = np.random.default_rng(7)
    vera = generatore.random(50000)
    esiti = (generatore.random(50000) < vera).astype(np.float64)
    assert calibration_error(vera, esiti) < 0.01


def test_una_previsione_distorta_ha_errore_grande() -> None:
    dichiarata, esiti = dati_distorti(n=50000)
    assert calibration_error(dichiarata, esiti) > 0.1


def test_senza_dati_l_errore_non_e_un_numero() -> None:
    assert np.isnan(calibration_error(np.array([]), np.array([])))
