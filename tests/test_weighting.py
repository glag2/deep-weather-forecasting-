"""Test dei pesi spaziali e del termine spettrale della perdita.

Ogni componente e' verificata da sola, contro una proprieta' che deve valere per
costruzione. Il test piu' importante e' l'ultimo: dimostra su dati sintetici che senza
il termine spettrale l'ottimo dell'errore quadratico e' un campo **smorzato**, e che
aggiungendolo l'ottimo si sposta verso l'ampiezza corretta. Senza quella prova il
termine sarebbe soltanto una complicazione in piu'.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dwf.models.losses import spectral_amplitude_loss
from dwf.weighting import (
    VIGO_LATITUDE,
    VIGO_LONGITUDE,
    WeightingError,
    focus_weight,
    latitude_area_weight,
    spatial_weight,
)

LATITUDINI = np.linspace(75.0, 10.0, 40).astype(np.float32)
LONGITUDINI = np.linspace(-40.0, 60.0, 60).astype(np.float32)


# --------------------------------------------------------------------------- #
# Peso di area
# --------------------------------------------------------------------------- #


class TestPesoDiArea:
    def test_decresce_verso_il_polo(self) -> None:
        peso = latitude_area_weight(LATITUDINI, LONGITUDINI.size)
        # Le latitudini sono decrescenti da 75 a 10: il peso deve crescere.
        assert np.all(np.diff(peso[:, 0]) > 0)

    def test_vale_il_coseno_della_latitudine(self) -> None:
        peso = latitude_area_weight(np.array([0.0, 60.0]), 3)
        assert peso[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert peso[1, 0] == pytest.approx(0.5, abs=1e-6)

    def test_e_costante_lungo_la_longitudine(self) -> None:
        peso = latitude_area_weight(LATITUDINI, LONGITUDINI.size)
        assert np.allclose(peso, peso[:, :1])

    def test_il_rapporto_fra_le_aree_e_quello_fisico(self) -> None:
        # A 70 gradi una cella copre cos(70) = 0,342 dell'area di una equatoriale.
        peso = latitude_area_weight(np.array([0.0, 70.0]), 1)
        assert float(peso[1, 0] / peso[0, 0]) == pytest.approx(0.342, abs=1e-3)

    def test_rifiuta_latitudini_impossibili(self) -> None:
        with pytest.raises(WeightingError):
            latitude_area_weight(np.array([91.0]), 2)

    def test_rifiuta_una_larghezza_nulla(self) -> None:
        with pytest.raises(WeightingError):
            latitude_area_weight(LATITUDINI, 0)


# --------------------------------------------------------------------------- #
# Fuoco su Vigo di Cadore
# --------------------------------------------------------------------------- #


class TestFuocoLocale:
    def test_il_massimo_cade_sul_punto_scelto(self) -> None:
        peso = focus_weight(LATITUDINI, LONGITUDINI)
        riga, colonna = np.unravel_index(int(peso.argmax()), peso.shape)
        assert abs(float(LATITUDINI[riga]) - VIGO_LATITUDE) < 2.0
        assert abs(float(LONGITUDINI[colonna]) - VIGO_LONGITUDE) < 2.0

    def test_al_centro_vale_uno_piu_il_guadagno(self) -> None:
        peso = focus_weight(
            np.array([VIGO_LATITUDE]), np.array([VIGO_LONGITUDE]), gain=0.5
        )
        assert float(peso[0, 0]) == pytest.approx(1.5, abs=1e-5)

    def test_lontano_dal_centro_torna_a_uno(self) -> None:
        peso = focus_weight(np.array([-60.0]), np.array([-170.0]), gain=0.5)
        assert float(peso[0, 0]) == pytest.approx(1.0, abs=1e-6)

    def test_non_toglie_mai_peso_al_resto_del_dominio(self) -> None:
        peso = focus_weight(LATITUDINI, LONGITUDINI, gain=0.5)
        assert float(peso.min()) >= 1.0

    def test_guadagno_nullo_lascia_tutto_a_uno(self) -> None:
        peso = focus_weight(LATITUDINI, LONGITUDINI, gain=0.0)
        assert np.allclose(peso, 1.0)

    def test_la_campana_e_isotropa_in_chilometri_non_in_gradi(self) -> None:
        # A 46,5 gradi un grado di longitudine vale circa 0,69 gradi di latitudine.
        # Uno scarto di un grado in latitudine deve quindi pesare piu' di uno scarto
        # di un grado in longitudine.
        in_latitudine = focus_weight(
            np.array([VIGO_LATITUDE + 1.0]), np.array([VIGO_LONGITUDE])
        )
        in_longitudine = focus_weight(
            np.array([VIGO_LATITUDE]), np.array([VIGO_LONGITUDE + 1.0])
        )
        assert float(in_longitudine[0, 0]) > float(in_latitudine[0, 0])

    def test_rifiuta_un_raggio_non_positivo(self) -> None:
        with pytest.raises(WeightingError):
            focus_weight(LATITUDINI, LONGITUDINI, radius_deg=0.0)

    def test_rifiuta_un_guadagno_negativo(self) -> None:
        with pytest.raises(WeightingError):
            focus_weight(LATITUDINI, LONGITUDINI, gain=-0.1)


# --------------------------------------------------------------------------- #
# Peso complessivo
# --------------------------------------------------------------------------- #


class TestPesoComplessivo:
    def test_ha_media_unitaria(self) -> None:
        peso = spatial_weight(LATITUDINI, LONGITUDINI)
        assert float(peso.mean()) == pytest.approx(1.0, abs=1e-5)

    def test_ha_media_unitaria_anche_senza_area(self) -> None:
        peso = spatial_weight(LATITUDINI, LONGITUDINI, use_area=False)
        assert float(peso.mean()) == pytest.approx(1.0, abs=1e-5)

    def test_la_media_unitaria_rende_confrontabili_le_configurazioni(self) -> None:
        # E' la proprieta' che permette di cambiare pesatura senza dover riaggiustare
        # i pesi relativi fra le teste: la scala della perdita non si muove.
        campo = np.random.default_rng(0).normal(size=(LATITUDINI.size, LONGITUDINI.size))
        errore = campo**2
        senza = errore.mean()
        con = (errore * spatial_weight(LATITUDINI, LONGITUDINI)).mean()
        assert con == pytest.approx(senza, rel=0.35)

    def test_disattivando_tutto_resta_uniforme(self) -> None:
        peso = spatial_weight(LATITUDINI, LONGITUDINI, use_area=False, focus_gain=0.0)
        assert np.allclose(peso, 1.0)

    def test_la_forma_segue_il_ritaglio(self) -> None:
        peso = spatial_weight(LATITUDINI[:10], LONGITUDINI[:7])
        assert peso.shape == (10, 7)

    def test_il_fuoco_alza_il_peso_sulla_cella_di_vigo(self) -> None:
        senza = spatial_weight(LATITUDINI, LONGITUDINI, focus_gain=0.0)
        con = spatial_weight(LATITUDINI, LONGITUDINI, focus_gain=1.0)
        riga = int(np.abs(LATITUDINI - VIGO_LATITUDE).argmin())
        colonna = int(np.abs(LONGITUDINI - VIGO_LONGITUDE).argmin())
        assert con[riga, colonna] > senza[riga, colonna]


# --------------------------------------------------------------------------- #
# Termine spettrale
# --------------------------------------------------------------------------- #


def campo_casuale(semente: int, forma: tuple[int, ...] = (2, 3, 32, 32)) -> torch.Tensor:
    generatore = torch.Generator().manual_seed(semente)
    return torch.randn(forma, generator=generatore)


class TestTermineSpettrale:
    def test_e_nullo_su_una_previsione_perfetta(self) -> None:
        campo = campo_casuale(1)
        assert float(spectral_amplitude_loss(campo, campo)) == pytest.approx(0.0, abs=1e-10)

    def test_e_positivo_su_una_previsione_smorzata(self) -> None:
        campo = campo_casuale(2)
        assert float(spectral_amplitude_loss(0.5 * campo, campo)) > 0.0

    def test_cresce_al_crescere_dello_smorzamento(self) -> None:
        campo = campo_casuale(3)
        valori = [
            float(spectral_amplitude_loss(fattore * campo, campo))
            for fattore in (0.9, 0.7, 0.5, 0.3)
        ]
        assert valori == sorted(valori)

    def test_ignora_la_posizione_e_guarda_solo_l_ampiezza(self) -> None:
        # Traslare il campo cambia la fase ma non i moduli della trasformata: il
        # termine deve restare nullo, altrimenti reintrodurrebbe la penalizzazione di
        # posizione che serve proprio a evitare.
        campo = campo_casuale(4)
        traslato = torch.roll(campo, shifts=(5, 7), dims=(-2, -1))
        assert float(spectral_amplitude_loss(traslato, campo)) == pytest.approx(0.0, abs=1e-8)

    def test_non_dipende_dalla_dimensione_del_ritaglio(self) -> None:
        # La normalizzazione serve a poter fissare il peso del termine una volta sola.
        piccolo = spectral_amplitude_loss(
            0.5 * campo_casuale(5, (1, 1, 16, 16)), campo_casuale(5, (1, 1, 16, 16))
        )
        grande = spectral_amplitude_loss(
            0.5 * campo_casuale(5, (1, 1, 64, 64)), campo_casuale(5, (1, 1, 64, 64))
        )
        assert float(grande) == pytest.approx(float(piccolo), rel=0.5)

    def test_rifiuta_forme_diverse(self) -> None:
        with pytest.raises(ValueError, match="Forme incompatibili"):
            spectral_amplitude_loss(campo_casuale(6), campo_casuale(6, (2, 3, 16, 16)))

    def test_e_derivabile(self) -> None:
        previsione = campo_casuale(7).requires_grad_(True)
        spectral_amplitude_loss(previsione, campo_casuale(8)).backward()
        assert previsione.grad is not None
        assert torch.isfinite(previsione.grad).all()

    def test_corregge_lo_smorzamento_ottimo_dell_errore_quadratico(self) -> None:
        """La prova che giustifica l'intero termine.

        Si costruisce una previsione correlata con la realta' ma non perfetta, come
        accade davvero, e si cerca il fattore di ampiezza migliore. Con il solo errore
        quadratico il minimo cade a un'ampiezza **inferiore** a uno: conviene sfumare.
        Aggiungendo il termine spettrale il minimo si sposta verso l'ampiezza corretta.
        """
        generatore = torch.Generator().manual_seed(11)
        vero = torch.randn(1, 1, 64, 64, generator=generatore)
        rumore = torch.randn(1, 1, 64, 64, generator=generatore)
        correlazione = 0.6
        grezza = correlazione * vero + np.sqrt(1 - correlazione**2) * rumore

        fattori = torch.linspace(0.1, 1.4, 53)
        solo_quadratico = [
            float(((fattore * grezza - vero) ** 2).mean()) for fattore in fattori
        ]
        con_spettrale = [
            float(
                ((fattore * grezza - vero) ** 2).mean()
                + 2.0 * spectral_amplitude_loss(fattore * grezza, vero)
            )
            for fattore in fattori
        ]

        migliore_quadratico = float(fattori[int(np.argmin(solo_quadratico))])
        migliore_con_spettrale = float(fattori[int(np.argmin(con_spettrale))])

        # L'errore quadratico da solo preferisce smorzare, come previsto dalla teoria:
        # l'ottimo cade attorno alla correlazione stessa.
        assert migliore_quadratico < 0.8
        assert migliore_quadratico == pytest.approx(correlazione, abs=0.15)
        # Il termine spettrale sposta l'ottimo verso l'ampiezza piena.
        assert migliore_con_spettrale > migliore_quadratico
