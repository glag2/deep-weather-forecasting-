"""Contratto dell'architettura rivale a nucleo globale.

La proprieta' che conta piu' di tutte e' l'indipendenza dalla risoluzione: ci si
addestra su ritagli 96x96 e si prevede sul dominio intero 261x401. Un peso legato al
numero di token renderebbe la rete inutilizzabile fuori dal ritaglio, e il difetto si
manifesterebbe solo al momento della previsione, cioe' dopo ore di addestramento.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dwf.config import Config
from dwf.data.features import InputLayout
from dwf.models.global_network import GlobalContextNet, GlobalNetworkSpec
from dwf.models.heads import OutputLayout
from dwf.models.network import DeepWeatherNet
from dwf.train import build_network

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config.load(CONFIG_PATH, project_root=tmp_path)


@pytest.fixture
def layout(config: Config) -> OutputLayout:
    return OutputLayout.from_targets(config.targets, config.windows.output_slots)


def _rete(layout: OutputLayout, **scarti: object) -> GlobalContextNet:
    parametri: dict[str, object] = {
        "in_channels": 6,
        "base_channels": 8,
        "patch": 4,
        "embed_channels": 16,
        "blocks": 1,
        "heads": 2,
    }
    parametri.update(scarti)
    return GlobalContextNet(GlobalNetworkSpec(**parametri), layout)


def test_l_uscita_ha_la_forma_del_layout(layout: OutputLayout) -> None:
    rete = _rete(layout)
    uscita = rete(torch.zeros(2, 6, 16, 16))
    assert uscita.shape == (2, layout.total_channels, 16, 16)


@pytest.mark.parametrize(("altezza", "larghezza"), [(16, 16), (17, 23), (13, 40)])
def test_una_dimensione_non_multipla_della_patch_viene_gestita(
    layout: OutputLayout, altezza: int, larghezza: int
) -> None:
    """Il dominio reale, 261 x 401, non e' multiplo di nessuna patch potenza di due."""
    rete = _rete(layout)
    uscita = rete(torch.zeros(1, 6, altezza, larghezza))
    assert uscita.shape[-2:] == (altezza, larghezza)


def test_gli_stessi_pesi_valgono_a_qualunque_risoluzione(layout: OutputLayout) -> None:
    """Addestrata su ritagli, applicata al dominio intero: nessun peso puo' dipendere
    dal numero di token."""
    rete = _rete(layout)
    rete.eval()
    with torch.no_grad():
        piccola = rete(torch.zeros(1, 6, 16, 16))
        grande = rete(torch.zeros(1, 6, 64, 96))
    assert piccola.shape[-2:] == (16, 16)
    assert grande.shape[-2:] == (64, 96)


def test_la_previsione_iniziale_e_neutra(layout: OutputLayout) -> None:
    """Se una delle due architetture partisse da rumore il confronto non sarebbe equo."""
    rete = _rete(layout)
    uscita = rete(torch.randn(1, 6, 16, 16))
    assert torch.all(uscita == 0.0)


def test_un_numero_di_canali_non_divisibile_per_le_teste_e_rifiutato(
    layout: OutputLayout,
) -> None:
    with pytest.raises(ValueError, match="divisibile"):
        _rete(layout, embed_channels=18, heads=4)


def test_una_patch_non_potenza_di_due_e_rifiutata(layout: OutputLayout) -> None:
    with pytest.raises(ValueError, match="potenza di due"):
        _rete(layout, patch=6)


def test_un_input_con_i_canali_sbagliati_e_rifiutato(layout: OutputLayout) -> None:
    rete = _rete(layout)
    with pytest.raises(ValueError, match="canali"):
        rete(torch.zeros(1, 5, 16, 16))


def test_la_configurazione_sceglie_l_architettura(config: Config) -> None:
    layout = OutputLayout.from_targets(config.targets, config.windows.output_slots)
    canali = InputLayout.from_config(config).n_channels

    a_u = build_network(config, layout, canali)
    variante = config.model.model_copy(update={"architecture": "global"})
    globale = build_network(config.model_copy(update={"model": variante}), layout, canali)

    assert isinstance(a_u, DeepWeatherNet)
    assert isinstance(globale, GlobalContextNet)


def test_la_configurazione_rifiuta_teste_incoerenti(config: Config) -> None:
    """Meglio fallire caricando la configurazione che dopo aver preparato le finestre."""
    with pytest.raises(ValueError, match="divisibile"):
        config.model.model_copy(
            update={"embed_channels": 100, "heads": 3}
        ).model_validate(
            {**config.model.model_dump(), "embed_channels": 100, "heads": 3}
        )
