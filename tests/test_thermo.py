"""Verifiche della termodinamica dell'aria umida contro valori tabulati noti.

I riferimenti (tensione di vapore a temperature fissate, umidita' specifica di
saturazione, temperatura potenziale equivalente) provengono da tabelle
meteorologiche standard e non dal codice, cosi' un errore nelle costanti non passa
inosservato.
"""

from __future__ import annotations

import numpy as np
import pytest

from dwf.thermo import (
    ES_AT_ZERO,
    THERMO_CHANNEL_NAMES,
    dewpoint_depression,
    equivalent_potential_temperature,
    latent_heat_content,
    latent_heat_of_vaporization,
    mixing_ratio,
    relative_humidity,
    saturation_vapour_pressure,
    specific_humidity,
    surface_pressure_from_msl,
    thermo_fields,
)

PRESSIONE_STANDARD = 101325.0
KELVIN = 273.15


class TestTensioneDiVaporeSaturo:
    def test_valore_a_zero_gradi(self) -> None:
        assert float(saturation_vapour_pressure(np.array([0.0]))[0]) == pytest.approx(
            ES_AT_ZERO, abs=0.1
        )

    def test_valore_tabulato_a_venti_gradi(self) -> None:
        # Tabelle standard: circa 2339 Pa sull'acqua liquida.
        assert float(saturation_vapour_pressure(np.array([20.0]))[0]) == pytest.approx(
            2339.0, abs=8.0
        )

    def test_valore_tabulato_a_meno_dieci_gradi(self) -> None:
        assert float(saturation_vapour_pressure(np.array([-10.0]))[0]) == pytest.approx(
            286.5, abs=3.0
        )

    def test_raddoppia_circa_ogni_dieci_gradi(self) -> None:
        # Conseguenza di Clausius-Clapeyron: e' la non linearita' che giustifica
        # fornire il campo alla rete invece di lasciarglielo dedurre.
        a_dieci = float(saturation_vapour_pressure(np.array([10.0]))[0])
        a_venti = float(saturation_vapour_pressure(np.array([20.0]))[0])
        assert 1.8 < a_venti / a_dieci < 2.1

    def test_cresce_sempre_con_la_temperatura(self) -> None:
        valori = saturation_vapour_pressure(np.linspace(-40.0, 45.0, 100))
        assert np.all(np.diff(valori) > 0)

    def test_resta_positiva_anche_a_temperature_polari(self) -> None:
        assert float(saturation_vapour_pressure(np.array([-60.0]))[0]) > 0.0


class TestUmiditaRelativa:
    def test_saturazione_quando_rugiada_uguale_temperatura(self) -> None:
        temperatura = np.array([-20.0, 0.0, 15.0, 30.0])
        umidita = relative_humidity(temperatura, temperatura)
        assert np.allclose(umidita, 1.0, atol=1e-4)

    def test_cala_al_crescere_del_deficit_di_rugiada(self) -> None:
        temperatura = np.full(5, 20.0)
        rugiada = np.array([20.0, 15.0, 10.0, 5.0, 0.0])
        umidita = relative_humidity(temperatura, rugiada)
        assert np.all(np.diff(umidita) < 0)

    def test_valore_tabulato_a_venti_gradi_con_rugiada_a_dieci(self) -> None:
        # 1228 Pa su 2337 Pa: circa 52 %.
        umidita = relative_humidity(np.array([20.0]), np.array([10.0]))
        assert float(umidita[0]) == pytest.approx(0.525, abs=0.01)

    def test_mai_oltre_uno_anche_con_lieve_sovrasaturazione(self) -> None:
        umidita = relative_humidity(np.array([10.0]), np.array([10.5]))
        assert float(umidita[0]) == 1.0

    def test_resta_in_zero_uno_su_dati_realistici(self) -> None:
        generatore = np.random.default_rng(0)
        temperatura = generatore.uniform(-30.0, 40.0, 500)
        rugiada = temperatura - generatore.uniform(0.0, 25.0, 500)
        umidita = relative_humidity(temperatura, rugiada)
        assert float(umidita.min()) >= 0.0
        assert float(umidita.max()) <= 1.0


class TestDeficitDiRugiada:
    def test_nullo_a_saturazione(self) -> None:
        assert float(dewpoint_depression(np.array([12.0]), np.array([12.0]))[0]) == 0.0

    def test_pari_alla_differenza(self) -> None:
        valore = dewpoint_depression(np.array([20.0]), np.array([4.0]))
        assert float(valore[0]) == pytest.approx(16.0, abs=1e-4)


class TestPressioneAllaSuperficie:
    def test_al_livello_del_mare_coincide_con_la_pressione_ridotta(self) -> None:
        pressione = surface_pressure_from_msl(
            np.array([PRESSIONE_STANDARD]), np.array([0.0]), np.array([15.0])
        )
        assert float(pressione[0]) == pytest.approx(PRESSIONE_STANDARD, rel=1e-6)

    def test_a_tremila_metri_si_avvicina_al_valore_standard(self) -> None:
        # Atmosfera standard a 3000 m: circa 701 hPa. L'approssimazione isoterma
        # usata qui resta entro qualche decina di hPa, sufficiente per una feature.
        geopotenziale = np.array([3000.0 * 9.80665])
        pressione = surface_pressure_from_msl(
            np.array([PRESSIONE_STANDARD]), geopotenziale, np.array([0.0])
        )
        assert 65000.0 < float(pressione[0]) < 72000.0

    def test_cala_monotonamente_con_la_quota(self) -> None:
        quote = np.array([0.0, 500.0, 1500.0, 3000.0]) * 9.80665
        pressione = surface_pressure_from_msl(
            np.full(4, PRESSIONE_STANDARD), quote, np.full(4, 5.0)
        )
        assert np.all(np.diff(pressione) < 0)

    def test_geopotenziale_negativo_non_aumenta_la_pressione(self) -> None:
        # Alcune celle marine hanno geopotenziale lievemente negativo: il clip evita
        # di inventare pressioni superiori a quella al livello del mare.
        pressione = surface_pressure_from_msl(
            np.array([PRESSIONE_STANDARD]), np.array([-500.0]), np.array([10.0])
        )
        assert float(pressione[0]) == pytest.approx(PRESSIONE_STANDARD, rel=1e-6)


class TestUmiditaSpecifica:
    def test_valore_tabulato_a_saturazione(self) -> None:
        # Aria satura a 20 gradi e 1000 hPa contiene circa 14,7 g/kg.
        umidita = specific_humidity(np.array([20.0]), np.array([100000.0]))
        assert float(umidita[0]) * 1000.0 == pytest.approx(14.7, abs=0.2)

    def test_cresce_con_la_rugiada(self) -> None:
        umidita = specific_humidity(
            np.array([-10.0, 0.0, 10.0, 20.0]), np.full(4, 100000.0)
        )
        assert np.all(np.diff(umidita) > 0)

    def test_cala_al_crescere_della_pressione(self) -> None:
        umidita = specific_humidity(
            np.full(3, 10.0), np.array([70000.0, 85000.0, 100000.0])
        )
        assert np.all(np.diff(umidita) < 0)

    def test_rapporto_di_mescolanza_appena_superiore_allumidita_specifica(self) -> None:
        rugiada = np.array([15.0])
        pressione = np.array([100000.0])
        assert float(mixing_ratio(rugiada, pressione)[0]) > float(
            specific_humidity(rugiada, pressione)[0]
        )


class TestCaloreLatente:
    def test_valore_a_zero_gradi(self) -> None:
        valore = latent_heat_of_vaporization(np.array([0.0]))
        assert float(valore[0]) == pytest.approx(2.501e6, rel=1e-6)

    def test_cala_al_crescere_della_temperatura(self) -> None:
        valori = latent_heat_of_vaporization(np.array([0.0, 20.0, 40.0]))
        assert np.all(np.diff(valori) < 0)

    def test_contenuto_energetico_di_aria_satura_a_venti_gradi(self) -> None:
        # 14,7 g/kg per circa 2,45 MJ/kg danno circa 36 kJ per chilogrammo d'aria.
        energia = latent_heat_content(
            np.array([20.0]), np.array([20.0]), np.array([100000.0])
        )
        assert float(energia[0]) == pytest.approx(3.6e4, rel=0.05)

    def test_aria_secca_contiene_molta_meno_energia_di_aria_umida(self) -> None:
        secca = latent_heat_content(
            np.array([20.0]), np.array([-20.0]), np.array([100000.0])
        )
        umida = latent_heat_content(
            np.array([20.0]), np.array([20.0]), np.array([100000.0])
        )
        assert float(umida[0]) > 10.0 * float(secca[0])


class TestTemperaturaPotenzialeEquivalente:
    def test_valore_tabulato_per_aria_satura_a_venti_gradi(self) -> None:
        # Tabelle di theta-e: circa 335 K per aria satura a 20 gradi e 1000 hPa.
        theta = equivalent_potential_temperature(
            np.array([20.0]), np.array([20.0]), np.array([100000.0])
        )
        assert float(theta[0]) == pytest.approx(335.0, abs=3.0)

    def test_tende_alla_temperatura_potenziale_per_aria_secchissima(self) -> None:
        theta = equivalent_potential_temperature(
            np.array([20.0]), np.array([-70.0]), np.array([100000.0])
        )
        assert float(theta[0]) == pytest.approx(20.0 + KELVIN, abs=1.0)

    def test_supera_sempre_la_temperatura_potenziale(self) -> None:
        # L'invariante corretto e' rispetto a theta, non alla temperatura assoluta:
        # sopra i 1000 hPa di riferimento la compressione porta theta *sotto* T, e
        # theta-e puo' seguirla pur avendo aggiunto tutto il calore latente.
        generatore = np.random.default_rng(1)
        temperatura = generatore.uniform(-30.0, 35.0, 300)
        rugiada = temperatura - generatore.uniform(0.0, 20.0, 300)
        pressione = generatore.uniform(70000.0, 104000.0, 300)
        theta_e = equivalent_potential_temperature(temperatura, rugiada, pressione)
        theta = (temperatura + KELVIN) * (100000.0 / pressione) ** 0.2854
        assert np.all(theta_e >= theta - 1e-3)

    def test_supera_la_temperatura_assoluta_sotto_i_mille_hpa(self) -> None:
        generatore = np.random.default_rng(3)
        temperatura = generatore.uniform(-30.0, 35.0, 300)
        rugiada = temperatura - generatore.uniform(0.0, 20.0, 300)
        pressione = generatore.uniform(70000.0, 100000.0, 300)
        theta_e = equivalent_potential_temperature(temperatura, rugiada, pressione)
        assert np.all(theta_e >= temperatura + KELVIN - 1e-3)

    def test_cresce_con_lumidita_a_parita_di_temperatura(self) -> None:
        temperatura = np.full(4, 20.0)
        rugiada = np.array([-10.0, 0.0, 10.0, 20.0])
        theta = equivalent_potential_temperature(
            temperatura, rugiada, np.full(4, 100000.0)
        )
        assert np.all(np.diff(theta) > 0)

    def test_resta_finita_su_dati_estremi(self) -> None:
        theta = equivalent_potential_temperature(
            np.array([-60.0, 45.0]), np.array([-65.0, 45.0]), np.array([50000.0, 105000.0])
        )
        assert np.all(np.isfinite(theta))


class TestCampiTermodinamici:
    def _campi(self) -> dict[str, np.ndarray]:
        generatore = np.random.default_rng(2)
        temperatura = generatore.uniform(-30.0, 35.0, (12, 15)).astype(np.float32)
        rugiada = temperatura - generatore.uniform(0.0, 18.0, (12, 15)).astype(np.float32)
        pressione = generatore.uniform(80000.0, 104000.0, (12, 15)).astype(np.float32)
        return thermo_fields(temperatura, rugiada, pressione)

    def test_restituisce_i_canali_dichiarati(self) -> None:
        assert tuple(self._campi()) == THERMO_CHANNEL_NAMES

    def test_conserva_forma_e_tipo(self) -> None:
        for valori in self._campi().values():
            assert valori.shape == (12, 15)
            assert valori.dtype == np.float32

    def test_tutti_i_valori_sono_finiti(self) -> None:
        for nome, valori in self._campi().items():
            assert np.all(np.isfinite(valori)), nome

    def test_scale_portano_i_canali_in_ordine_di_grandezza_unitario(self) -> None:
        # I canali termodinamici non passano dalle statistiche di train: le scale
        # dichiarate devono bastare a tenerli confrontabili con le altre feature.
        for nome, valori in self._campi().items():
            assert float(np.abs(valori).max()) < 10.0, nome
