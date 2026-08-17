"""Test del registro delle varianti di blocco.

Il valore del confronto fra architetture sta tutto nel fatto che cambi **una cosa
sola**. Questi test verificano proprio quello: che ogni variante rispetti lo stesso
contratto, che la rete costruita attorno a ciascuna abbia gli stessi ingressi e le
stesse uscite, e che una variante inesistente fallisca subito invece che a meta'
addestramento.
"""

from __future__ import annotations

import pytest
import torch

from dwf.models import variants
from dwf.models.heads import OutputLayout
from dwf.models.network import DeepWeatherNet, NetworkSpec
from dwf.models.variants.fourier import SpectralConv2d
from dwf.models.variants.recurrent import ConvGRUCell, RecurrentBlock

CANALI_INGRESSO = 24
SCADENZE = 9


class TargetFinto:
    def __init__(self, nome: str, testa: str) -> None:
        self.name = nome
        self.head = testa
        self.threshold = 0.0001
        self.reference = "tp"


@pytest.fixture
def layout() -> OutputLayout:
    return OutputLayout.from_targets(
        [
            TargetFinto("t2m", "gaussian"),
            TargetFinto("tp", "hurdle"),
            TargetFinto("sf", "fraction_of"),
        ],
        SCADENZE,
    )


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #


class TestRegistro:
    def test_contiene_le_famiglie_previste_dalla_ricerca(self) -> None:
        assert set(variants.available()) == {
            "conv", "attention", "fourier", "recurrent", "hybrid"
        }

    def test_ogni_variante_ha_una_descrizione(self) -> None:
        for nome in variants.available():
            assert variants.describe(nome).strip()

    def test_una_variante_sconosciuta_fallisce_subito(self) -> None:
        with pytest.raises(variants.VariantError, match="sconosciuta"):
            variants.get("trasformatore_magico")

    def test_l_errore_elenca_le_alternative(self) -> None:
        with pytest.raises(variants.VariantError, match="conv"):
            variants.get("inesistente")

    def test_la_specifica_rifiuta_una_variante_inesistente(self) -> None:
        # Meglio fallire costruendo la specifica che dopo un'ora di addestramento.
        with pytest.raises(variants.VariantError):
            NetworkSpec(in_channels=CANALI_INGRESSO, variant="inesistente")


# --------------------------------------------------------------------------- #
# Contratto comune del blocco
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("nome", variants.available())
class TestContrattoDelBlocco:
    def test_conserva_la_risoluzione(self, nome: str) -> None:
        blocco = variants.get(nome)(16, 16, 0.0)
        assert blocco(torch.randn(2, 16, 24, 20)).shape == (2, 16, 24, 20)

    def test_cambia_il_numero_di_canali_come_richiesto(self, nome: str) -> None:
        blocco = variants.get(nome)(16, 32, 0.0)
        assert blocco(torch.randn(2, 16, 24, 20)).shape == (2, 32, 24, 20)

    def test_funziona_su_dimensioni_dispari(self, nome: str) -> None:
        # Il dominio reale e' 261 x 401, entrambi dispari: una variante che assumesse
        # dimensioni pari fallirebbe solo in produzione.
        blocco = variants.get(nome)(8, 8, 0.0)
        assert blocco(torch.randn(1, 8, 13, 17)).shape == (1, 8, 13, 17)

    def test_produce_valori_finiti(self, nome: str) -> None:
        blocco = variants.get(nome)(8, 8, 0.0)
        assert torch.isfinite(blocco(torch.randn(2, 8, 16, 16))).all()

    def test_propaga_il_gradiente(self, nome: str) -> None:
        blocco = variants.get(nome)(8, 8, 0.0)
        ingresso = torch.randn(1, 8, 16, 16, requires_grad=True)
        blocco(ingresso).square().mean().backward()
        assert ingresso.grad is not None
        assert torch.isfinite(ingresso.grad).all()
        assert float(ingresso.grad.abs().sum()) > 0.0

    def test_accetta_il_dropout(self, nome: str) -> None:
        blocco = variants.get(nome)(8, 8, 0.25)
        assert blocco(torch.randn(1, 8, 16, 16)).shape == (1, 8, 16, 16)


# --------------------------------------------------------------------------- #
# Rete completa: e' qui che il confronto diventa lecito
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("nome", variants.available())
class TestReteCompleta:
    def test_ingresso_e_uscita_sono_identici_fra_varianti(
        self, nome: str, layout: OutputLayout
    ) -> None:
        rete = DeepWeatherNet(NetworkSpec(in_channels=CANALI_INGRESSO, variant=nome), layout)
        with torch.no_grad():
            uscita = rete(torch.randn(1, CANALI_INGRESSO, 32, 32))
        assert uscita.shape == (1, layout.total_channels, 32, 32)

    def test_regge_una_griglia_non_multipla_delle_riduzioni(
        self, nome: str, layout: OutputLayout
    ) -> None:
        rete = DeepWeatherNet(NetworkSpec(in_channels=CANALI_INGRESSO, variant=nome), layout)
        with torch.no_grad():
            uscita = rete(torch.randn(1, CANALI_INGRESSO, 29, 37))
        assert uscita.shape == (1, layout.total_channels, 29, 37)

    def test_parte_da_una_previsione_neutra(self, nome: str, layout: OutputLayout) -> None:
        # I pesi finali sono azzerati: qualunque sia il blocco, la rete appena creata
        # deve produrre un campo costante, non rumore.
        rete = DeepWeatherNet(NetworkSpec(in_channels=CANALI_INGRESSO, variant=nome), layout)
        with torch.no_grad():
            uscita = rete(torch.randn(1, CANALI_INGRESSO, 32, 32))
        assert torch.allclose(uscita, uscita[..., :1, :1], atol=1e-6)


def test_le_varianti_hanno_costi_diversi(layout: OutputLayout) -> None:
    """Se due varianti avessero gli stessi parametri, il confronto sarebbe inutile."""
    conteggi = {
        nome: DeepWeatherNet(
            NetworkSpec(in_channels=CANALI_INGRESSO, variant=nome), layout
        ).n_parameters
        for nome in variants.available()
    }
    assert len(set(conteggi.values())) == len(conteggi)


# --------------------------------------------------------------------------- #
# Proprieta' specifiche delle singole varianti
# --------------------------------------------------------------------------- #


class TestOperatoreSpettrale:
    def test_i_parametri_non_dipendono_dalla_dimensione_del_dominio(self) -> None:
        """E' la proprieta' che permette di addestrare su ritagli e prevedere su tutto."""
        operatore = SpectralConv2d(8, 8, modes=4)
        prima = sum(p.numel() for p in operatore.parameters())
        with torch.no_grad():
            operatore(torch.randn(1, 8, 64, 64))
        assert sum(p.numel() for p in operatore.parameters()) == prima

    def test_tronca_i_modi_quando_la_griglia_e_piccola(self) -> None:
        # Con otto celle i modi disponibili sono quattro: chiederne dodici non deve
        # far fallire nulla, solo usarne meno.
        operatore = SpectralConv2d(4, 4, modes=12)
        assert operatore(torch.randn(1, 4, 8, 8)).shape == (1, 4, 8, 8)

    def test_l_uscita_e_reale(self) -> None:
        operatore = SpectralConv2d(4, 4, modes=3)
        assert not operatore(torch.randn(1, 4, 16, 16)).is_complex()

    def test_il_costo_cresce_col_quadrato_dei_modi(self) -> None:
        # Vincolo che ha imposto otto modi invece di sedici: con sedici la rete
        # arrivava a 228 milioni di parametri, in gran parte troncati a ogni passata.
        pochi = sum(p.numel() for p in SpectralConv2d(8, 8, modes=4).parameters())
        molti = sum(p.numel() for p in SpectralConv2d(8, 8, modes=8).parameters())
        assert molti == pytest.approx(4 * pochi, rel=0.01)

    def test_rifiuta_zero_modi(self) -> None:
        with pytest.raises(ValueError, match="almeno un modo"):
            SpectralConv2d(4, 4, modes=0)


class TestRicorrenza:
    def test_la_cella_conserva_forma_e_canali(self) -> None:
        cella = ConvGRUCell(6)
        stato = torch.randn(2, 6, 12, 12)
        assert cella(stato, torch.randn(2, 6, 12, 12)).shape == stato.shape

    def test_iterare_di_piu_non_aggiunge_parametri(self) -> None:
        """E' l'ipotesi che la variante ricorrente serve a testare."""
        poche = RecurrentBlock(8, 8, steps=1)
        molte = RecurrentBlock(8, 8, steps=6)
        assert sum(p.numel() for p in poche.parameters()) == sum(
            p.numel() for p in molte.parameters()
        )

    def test_il_numero_di_iterazioni_cambia_il_risultato(self) -> None:
        torch.manual_seed(0)
        blocco = RecurrentBlock(8, 8, steps=1)
        ingresso = torch.randn(1, 8, 16, 16)
        with torch.no_grad():
            una = blocco(ingresso).clone()
            blocco.steps = 4
            quattro = blocco(ingresso)
        assert not torch.allclose(una, quattro)

    def test_rifiuta_zero_iterazioni(self) -> None:
        with pytest.raises(ValueError, match="almeno un'iterazione"):
            RecurrentBlock(8, 8, steps=0)
