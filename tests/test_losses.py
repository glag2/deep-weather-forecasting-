"""Test delle funzioni di perdita.

Una perdita sbagliata non fa fallire nulla: il modello converge e prevede male. Questi
test verificano le proprieta' che la rendono corretta, non solo che restituisca un
numero: minimo nel punto giusto, penalizzazione dell'eccesso di sicurezza, e
mascheramento effettivo dei punti che non devono contribuire.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from dwf.config import Config
from dwf.models.heads import OutputLayout
from dwf.models.losses import (
    LOG_TWO_PI,
    MAX_LOG_VAR,
    MIN_LOG_VAR,
    CompositeLoss,
    fraction_loss,
    gaussian_nll,
    hurdle_amount_loss,
    hurdle_occurrence_loss,
    masked_mean,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config.load(CONFIG_PATH, project_root=tmp_path)


@pytest.fixture
def layout(config: Config) -> OutputLayout:
    return OutputLayout.from_targets(config.targets, config.windows.output_slots)


def batch_finto(config: Config, altezza: int = 4, larghezza: int = 4) -> dict:
    slot = config.windows.output_slots
    generatore = torch.Generator().manual_seed(0)
    occorrenza = (torch.rand(1, slot, altezza, larghezza, generator=generatore) > 0.6).float()
    return {
        "target_t2m": torch.randn(1, slot, altezza, larghezza, generator=generatore),
        "target_tp_occurrence": occorrenza,
        "target_tp_amount": torch.randn(1, slot, altezza, larghezza, generator=generatore),
        "mask_tp_amount": occorrenza,
        "target_sf_fraction": torch.rand(1, slot, altezza, larghezza, generator=generatore),
        "mask_sf_fraction": occorrenza,
    }


# --------------------------------------------------------------------------- #
# Media mascherata
# --------------------------------------------------------------------------- #


def test_la_maschera_esclude_i_punti_non_ammessi() -> None:
    valori = torch.tensor([1.0, 1000.0])
    assert float(masked_mean(valori, torch.tensor([1.0, 0.0]))) == 1.0


def test_senza_maschera_si_media_tutto() -> None:
    assert float(masked_mean(torch.tensor([1.0, 3.0]), None)) == 2.0


def test_una_maschera_vuota_non_divide_per_zero() -> None:
    """Capita davvero: una finestra estiva puo' non avere un solo punto di neve."""
    perdita = masked_mean(torch.tensor([5.0, 5.0]), torch.zeros(2))
    assert float(perdita) == 0.0
    assert torch.isfinite(perdita)


def test_la_maschera_vuota_lascia_passare_il_gradiente() -> None:
    valori = torch.tensor([5.0, 5.0], requires_grad=True)
    masked_mean(valori, torch.zeros(2)).backward()
    assert valori.grad is not None
    assert torch.isfinite(valori.grad).all()


# --------------------------------------------------------------------------- #
# NLL gaussiana
# --------------------------------------------------------------------------- #


def test_la_nll_gaussiana_vale_la_formula_nota() -> None:
    bersaglio = torch.zeros(100)
    valore = gaussian_nll(torch.zeros(100), torch.zeros(100), bersaglio)
    assert float(valore) == pytest.approx(0.5 * LOG_TWO_PI, abs=1e-6)


def test_la_nll_e_minima_con_la_media_esatta() -> None:
    bersaglio = torch.randn(500)
    esatta = gaussian_nll(bersaglio, torch.zeros(500), bersaglio)
    spostata = gaussian_nll(bersaglio + 0.5, torch.zeros(500), bersaglio)
    assert float(esatta) < float(spostata)


def test_la_nll_punisce_la_sicurezza_ingiustificata() -> None:
    """Una previsione sbagliata e sicura deve costare piu' di una sbagliata e incerta.
    E' la proprieta' che rende misurabile l'affidabilita'."""
    bersaglio = torch.zeros(200)
    sbagliata = torch.full((200,), 2.0)
    sicura = gaussian_nll(sbagliata, torch.full((200,), -2.0), bersaglio)
    incerta = gaussian_nll(sbagliata, torch.full((200,), 1.0), bersaglio)
    assert float(sicura) > float(incerta)


def test_la_log_varianza_e_limitata() -> None:
    """Senza limiti l'esponenziale va in overflow e la perdita diventa NaN."""
    bersaglio = torch.zeros(10)
    estrema = gaussian_nll(torch.zeros(10), torch.full((10,), -1000.0), bersaglio)
    assert torch.isfinite(estrema)
    attesa = 0.5 * (LOG_TWO_PI + MIN_LOG_VAR)
    assert float(estrema) == pytest.approx(attesa, abs=1e-5)


def test_il_limite_superiore_della_log_varianza_e_attivo() -> None:
    valore = gaussian_nll(torch.zeros(10), torch.full((10,), 1000.0), torch.zeros(10))
    assert float(valore) == pytest.approx(0.5 * (LOG_TWO_PI + MAX_LOG_VAR), abs=1e-5)


# --------------------------------------------------------------------------- #
# Teste di precipitazione
# --------------------------------------------------------------------------- #


def test_l_occorrenza_a_logit_zero_vale_log_due() -> None:
    valore = hurdle_occurrence_loss(torch.zeros(50), torch.ones(50))
    assert float(valore) == pytest.approx(math.log(2.0), abs=1e-6)


def test_l_occorrenza_premia_la_previsione_corretta() -> None:
    bersaglio = torch.ones(50)
    giusta = hurdle_occurrence_loss(torch.full((50,), 5.0), bersaglio)
    sbagliata = hurdle_occurrence_loss(torch.full((50,), -5.0), bersaglio)
    assert float(giusta) < float(sbagliata)


def test_la_quantita_si_addestra_solo_dove_piove() -> None:
    """Un errore enorme su un punto asciutto non deve entrare nella perdita."""
    previsione = torch.tensor([0.0, 100.0])
    bersaglio = torch.tensor([0.0, 0.0])
    maschera = torch.tensor([1.0, 0.0])
    assert float(hurdle_amount_loss(previsione, bersaglio, maschera)) == 0.0


def test_la_quantita_usa_una_perdita_robusta() -> None:
    """Huber cresce linearmente: un evento estremo non domina il gradiente."""
    maschera = torch.ones(1)
    piccolo = hurdle_amount_loss(torch.tensor([1.0]), torch.zeros(1), maschera)
    grande = hurdle_amount_loss(torch.tensor([100.0]), torch.zeros(1), maschera)
    assert float(grande) < float(piccolo) * 10000.0


# --------------------------------------------------------------------------- #
# Frazione di neve
# --------------------------------------------------------------------------- #


def test_la_frazione_e_minima_quando_la_probabilita_uguaglia_la_frazione() -> None:
    bersaglio = torch.full((200,), 0.3)
    maschera = torch.ones(200)
    logit_giusto = torch.full((200,), math.log(0.3 / 0.7))
    esatta = fraction_loss(logit_giusto, bersaglio, maschera)
    for spostamento in (-1.0, 1.0):
        assert float(esatta) < float(
            fraction_loss(logit_giusto + spostamento, bersaglio, maschera)
        )


def test_la_frazione_ignora_i_punti_senza_precipitazione() -> None:
    logit = torch.tensor([0.0, 50.0])
    bersaglio = torch.tensor([0.5, 0.0])
    maschera = torch.tensor([1.0, 0.0])
    assert torch.isfinite(fraction_loss(logit, bersaglio, maschera))


def test_una_frazione_fuori_intervallo_viene_limitata() -> None:
    """Il rumore di impacchettamento dei GRIB puo' produrre rapporti appena sopra 1."""
    valore = fraction_loss(torch.zeros(4), torch.full((4,), 1.04), torch.ones(4))
    assert torch.isfinite(valore)
    assert float(valore) == pytest.approx(math.log(2.0), abs=1e-6)


# --------------------------------------------------------------------------- #
# Perdita composita
# --------------------------------------------------------------------------- #


def test_la_composita_produce_una_componente_per_testa(
    config: Config, layout: OutputLayout
) -> None:
    criterio = CompositeLoss(layout, config.training.loss_weights)
    previsione = torch.zeros(1, layout.total_channels, 4, 4)
    esito = criterio(previsione, batch_finto(config))
    assert set(esito.components) == {
        "t2m_nll", "tp_occurrence", "tp_amount", "sf_fraction"
    }


def test_la_composita_e_derivabile(config: Config, layout: OutputLayout) -> None:
    criterio = CompositeLoss(layout, config.training.loss_weights)
    previsione = torch.zeros(1, layout.total_channels, 4, 4, requires_grad=True)
    criterio(previsione, batch_finto(config)).total.backward()
    assert previsione.grad is not None
    assert torch.isfinite(previsione.grad).all()
    assert float(previsione.grad.abs().sum()) > 0.0


def test_un_peso_nullo_annulla_il_contributo(
    config: Config, layout: OutputLayout
) -> None:
    class Pesi:
        gaussian = 0.0
        precip_occurrence = 1.0
        precip_amount = 0.0
        snow_fraction = 0.0

    criterio = CompositeLoss(layout, Pesi())
    previsione = torch.zeros(1, layout.total_channels, 4, 4)
    esito = criterio(previsione, batch_finto(config))
    assert float(esito.total) == pytest.approx(float(esito.components["tp_occurrence"]))


def test_un_bersaglio_mancante_e_segnalato(config: Config, layout: OutputLayout) -> None:
    criterio = CompositeLoss(layout, config.training.loss_weights)
    incompleto = batch_finto(config)
    del incompleto["target_t2m"]
    with pytest.raises(KeyError, match="target_t2m"):
        criterio(torch.zeros(1, layout.total_channels, 4, 4), incompleto)


def test_il_riepilogo_e_serializzabile(config: Config, layout: OutputLayout) -> None:
    criterio = CompositeLoss(layout, config.training.loss_weights)
    esito = criterio(torch.zeros(1, layout.total_channels, 4, 4), batch_finto(config))
    valori = esito.detached()
    assert "total" in valori
    assert all(isinstance(valore, float) for valore in valori.values())
