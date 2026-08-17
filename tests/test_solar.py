"""Verifiche della geometria solare contro valori astronomici noti.

I confronti usano valori indipendenti dal codice (obliquita' dell'eclittica, date di
perielio e afelio, durata del giorno all'equatore): un test che si limitasse alla
coerenza interna passerebbe anche con le formule sbagliate.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from dwf.solar import (
    SOLAR_CHANNEL_NAMES,
    SOLAR_CONSTANT,
    cos_solar_zenith,
    day_length_hours,
    earth_sun_distance_factor,
    equation_of_time,
    solar_declination,
    solar_fields,
    toa_insolation,
)

# Giorni dell'anno degli eventi astronomici, in anno non bisestile.
EQUINOZIO_MARZO = 79
SOLSTIZIO_GIUGNO = 172
EQUINOZIO_SETTEMBRE = 266
SOLSTIZIO_DICEMBRE = 355

# Inclinazione dell'asse terrestre in gradi: e' il valore che la declinazione deve
# raggiungere ai solstizi.
OBLIQUITA = 23.44


class TestDeclinazione:
    def test_solstizio_giugno_raggiunge_obliquita(self) -> None:
        gradi = np.rad2deg(solar_declination(SOLSTIZIO_GIUGNO))
        assert gradi == pytest.approx(OBLIQUITA, abs=0.3)

    def test_solstizio_dicembre_raggiunge_obliquita_negativa(self) -> None:
        gradi = np.rad2deg(solar_declination(SOLSTIZIO_DICEMBRE))
        assert gradi == pytest.approx(-OBLIQUITA, abs=0.3)

    def test_equinozi_annullano_la_declinazione(self) -> None:
        for giorno in (EQUINOZIO_MARZO, EQUINOZIO_SETTEMBRE):
            gradi = np.rad2deg(solar_declination(giorno))
            assert abs(gradi) < 1.0

    def test_resta_entro_obliquita_tutto_lanno(self) -> None:
        gradi = np.rad2deg(solar_declination(np.arange(1, 366)))
        assert np.all(np.abs(gradi) <= OBLIQUITA + 0.3)

    def test_accetta_array(self) -> None:
        risultato = solar_declination(np.array([1.0, 100.0, 200.0]))
        assert risultato.shape == (3,)


class TestDistanzaDalSole:
    def test_massima_al_perielio_di_inizio_gennaio(self) -> None:
        fattore = earth_sun_distance_factor(np.arange(1, 366))
        assert int(np.argmax(fattore)) + 1 == pytest.approx(3, abs=4)

    def test_minima_allafelio_di_inizio_luglio(self) -> None:
        fattore = earth_sun_distance_factor(np.arange(1, 366))
        assert int(np.argmin(fattore)) + 1 == pytest.approx(185, abs=5)

    def test_modulazione_dell_irradianza_di_circa_sette_percento(self) -> None:
        fattore = earth_sun_distance_factor(np.arange(1, 366))
        assert float(fattore.max()) == pytest.approx(1.0345, abs=0.002)
        assert float(fattore.min()) == pytest.approx(0.9666, abs=0.002)

    def test_media_annuale_unitaria(self) -> None:
        fattore = earth_sun_distance_factor(np.arange(1, 366))
        assert float(fattore.mean()) == pytest.approx(1.0, abs=0.001)


class TestEquazioneDelTempo:
    def test_resta_entro_il_quarto_dora(self) -> None:
        minuti = equation_of_time(np.arange(1, 366))
        assert float(minuti.max()) == pytest.approx(16.4, abs=1.0)
        assert float(minuti.min()) == pytest.approx(-14.2, abs=1.0)

    def test_media_annuale_quasi_nulla(self) -> None:
        minuti = equation_of_time(np.arange(1, 366))
        assert abs(float(minuti.mean())) < 0.5


class TestDurataDelGiorno:
    def test_equatore_dodici_ore_in_ogni_stagione(self) -> None:
        for giorno in (1, 90, 180, 270, 365):
            ore = day_length_hours(np.array([0.0]), giorno)
            assert float(ore[0]) == pytest.approx(12.0, abs=0.01)

    def test_sole_di_mezzanotte_oltre_il_circolo_polare(self) -> None:
        ore = day_length_hours(np.array([70.0]), SOLSTIZIO_GIUGNO)
        assert float(ore[0]) == pytest.approx(24.0, abs=0.01)

    def test_notte_polare_al_solstizio_dinverno(self) -> None:
        ore = day_length_hours(np.array([70.0]), SOLSTIZIO_DICEMBRE)
        assert float(ore[0]) == pytest.approx(0.0, abs=0.01)

    def test_cresce_verso_nord_in_estate(self) -> None:
        ore = day_length_hours(np.array([10.0, 30.0, 50.0, 65.0]), SOLSTIZIO_GIUGNO)
        assert np.all(np.diff(ore) > 0)

    def test_cala_verso_nord_in_inverno(self) -> None:
        ore = day_length_hours(np.array([10.0, 30.0, 50.0, 65.0]), SOLSTIZIO_DICEMBRE)
        assert np.all(np.diff(ore) < 0)

    def test_somma_solstizi_vale_ventiquattro_ore(self) -> None:
        # Fuori dai circoli polari la durata del giorno d'estate e quella d'inverno
        # sono simmetriche rispetto alle dodici ore.
        latitudini = np.array([0.0, 20.0, 45.0, 60.0])
        estate = day_length_hours(latitudini, SOLSTIZIO_GIUGNO)
        inverno = day_length_hours(latitudini, SOLSTIZIO_DICEMBRE)
        assert np.allclose(estate + inverno, 24.0, atol=0.1)


class TestCosenoZenitale:
    def test_sole_allo_zenit_a_mezzogiorno_sullequatore_allequinozio(self) -> None:
        istante = datetime(2025, 3, 20, 12, 0, tzinfo=UTC)
        coseno = cos_solar_zenith(np.array([0.0]), np.array([0.0]), istante)
        assert float(coseno[0, 0]) > 0.99

    def test_notte_produce_zero(self) -> None:
        istante = datetime(2025, 3, 20, 0, 0, tzinfo=UTC)
        coseno = cos_solar_zenith(np.array([0.0]), np.array([0.0]), istante)
        assert float(coseno[0, 0]) == 0.0

    def test_mai_negativo_ne_oltre_uno(self) -> None:
        latitudini = np.linspace(10.0, 75.0, 20)
        longitudini = np.linspace(-40.0, 60.0, 30)
        for ora in (0, 6, 12, 18):
            istante = datetime(2025, 7, 15, ora, tzinfo=UTC)
            coseno = cos_solar_zenith(latitudini, longitudini, istante)
            assert float(coseno.min()) >= 0.0
            assert float(coseno.max()) <= 1.0

    def test_forma_della_griglia(self) -> None:
        coseno = cos_solar_zenith(
            np.linspace(75.0, 10.0, 261),
            np.linspace(-40.0, 60.0, 401),
            datetime(2025, 6, 1, 12, tzinfo=UTC),
        )
        assert coseno.shape == (261, 401)

    def test_est_illuminato_prima_dell_ovest_al_mattino(self) -> None:
        # Alle 06 UTC il sole e' gia' sorto sull'Europa orientale ma non sull'Atlantico.
        istante = datetime(2025, 6, 21, 6, 0, tzinfo=UTC)
        coseno = cos_solar_zenith(np.array([45.0]), np.array([-40.0, 60.0]), istante)
        assert float(coseno[0, 1]) > float(coseno[0, 0])

    def test_notte_polare_resta_buia_tutto_il_giorno(self) -> None:
        for ora in range(0, 24, 3):
            istante = datetime(2025, 12, 21, ora, tzinfo=UTC)
            coseno = cos_solar_zenith(np.array([75.0]), np.array([0.0]), istante)
            assert float(coseno[0, 0]) == 0.0


class TestInsolazione:
    def test_non_supera_la_costante_solare_corretta(self) -> None:
        istante = datetime(2025, 1, 3, 12, tzinfo=UTC)
        valori = toa_insolation(np.linspace(10.0, 75.0, 40), np.linspace(-40.0, 60.0, 40), istante)
        assert float(valori.max()) <= SOLAR_CONSTANT * 1.035

    def test_perielio_piu_intenso_dellafelio_a_parita_di_zenit(self) -> None:
        # All'equatore a mezzogiorno la differenza residua e' quasi solo la distanza.
        equatore = (np.array([0.0]), np.array([0.0]))
        gennaio = toa_insolation(*equatore, datetime(2025, 1, 3, 12, tzinfo=UTC))
        luglio = toa_insolation(*equatore, datetime(2025, 7, 4, 12, tzinfo=UTC))
        assert float(gennaio[0, 0]) > float(luglio[0, 0])


class TestCampiSolari:
    def test_restituisce_i_canali_dichiarati(self) -> None:
        campi = solar_fields(
            np.linspace(75.0, 10.0, 10),
            np.linspace(-40.0, 60.0, 12),
            datetime(2025, 6, 21, 12, tzinfo=UTC),
        )
        assert tuple(campi) == SOLAR_CHANNEL_NAMES

    def test_tutti_i_canali_hanno_la_forma_della_griglia(self) -> None:
        campi = solar_fields(
            np.linspace(75.0, 10.0, 10),
            np.linspace(-40.0, 60.0, 12),
            datetime(2025, 6, 21, 12, tzinfo=UTC),
        )
        for valori in campi.values():
            assert valori.shape == (10, 12)
            assert valori.dtype == np.float32

    def test_tutti_i_canali_stanno_in_zero_uno(self) -> None:
        # I canali solari non passano dalle statistiche di train, quindi la scala
        # dichiarata deve valere per costruzione in ogni stagione e a ogni ora.
        for mese in (1, 4, 7, 10):
            for ora in (0, 6, 12, 18):
                campi = solar_fields(
                    np.linspace(75.0, 10.0, 20),
                    np.linspace(-40.0, 60.0, 20),
                    datetime(2025, mese, 15, ora, tzinfo=UTC),
                )
                for nome, valori in campi.items():
                    assert float(valori.min()) >= 0.0, nome
                    assert float(valori.max()) <= 1.0, nome

    def test_durata_del_giorno_costante_lungo_il_parallelo(self) -> None:
        campi = solar_fields(
            np.array([45.0]),
            np.linspace(-40.0, 60.0, 15),
            datetime(2025, 6, 21, 12, tzinfo=UTC),
        )
        assert float(campi["day_length"].std()) == 0.0

    def test_il_coseno_zenitale_varia_lungo_il_parallelo(self) -> None:
        # Se fosse costante in longitudine il canale non porterebbe l'informazione
        # sull'ora locale, che e' la ragione per cui esiste.
        campi = solar_fields(
            np.array([45.0]),
            np.linspace(-40.0, 60.0, 15),
            datetime(2025, 6, 21, 12, tzinfo=UTC),
        )
        assert float(campi["cos_zenith"].std()) > 0.01
