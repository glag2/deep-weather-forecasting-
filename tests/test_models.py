"""Test della rete e della mappa dei canali di uscita.

Il punto critico verificato qui e' che la rete allenata su crop 96 x 96 sia
applicabile senza modifiche al dominio reale 261 x 401, che non e' divisibile per
2^depth, e che la corrispondenza canale -> (variabile, componente, lead time) resti
quella dichiarata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from dwf.config import Config
from dwf.models.blocks import crop_padding, group_count, pad_to_multiple
from dwf.models.heads import HEAD_COMPONENTS, OutputLayout
from dwf.models.network import DeepWeatherNet, NetworkSpec, build_network

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
REAL_GRID = (261, 401)


@dataclass(frozen=True)
class FakeTarget:
    name: str
    head: str


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def layout(config: Config) -> OutputLayout:
    return OutputLayout.from_targets(config.targets, config.windows.output_slots)


# --------------------------------------------------------------------------- #
# Layout dei canali
# --------------------------------------------------------------------------- #


def test_layout_conta_i_canali_come_la_configurazione(config: Config, layout: OutputLayout) -> None:
    assert layout.total_channels == config.n_output_channels == 45


def test_blocchi_sono_contigui_e_senza_sovrapposizioni(layout: OutputLayout) -> None:
    cursor = 0
    for block in layout.blocks:
        assert block.start == cursor
        cursor = block.stop
    assert cursor == layout.total_channels


def test_ogni_blocco_copre_tutti_i_lead_time(layout: OutputLayout) -> None:
    assert all(block.n_channels == layout.output_slots for block in layout.blocks)


def test_ordine_dei_blocchi_segue_i_target_dichiarati(layout: OutputLayout) -> None:
    assert [block.key for block in layout.blocks] == [
        ("t2m", "mean"),
        ("t2m", "log_var"),
        ("tp", "occurrence_logit"),
        ("tp", "amount"),
        ("sf", "fraction_logit"),
    ]


def test_componenti_corrispondono_al_tipo_di_testa(layout: OutputLayout) -> None:
    assert layout.components_of("t2m") == HEAD_COMPONENTS["gaussian"]
    assert layout.components_of("tp") == HEAD_COMPONENTS["hurdle"]
    assert layout.components_of("sf") == HEAD_COMPONENTS["fraction_of"]


def test_select_estrae_il_blocco_giusto(layout: OutputLayout) -> None:
    prediction = torch.zeros(2, layout.total_channels, 8, 8)
    # Marca il blocco della log-varianza per verificare che venga estratto proprio quello.
    block = layout.block("t2m", "log_var")
    prediction[:, block.start : block.stop] = 7.0
    selected = layout.select(prediction, "t2m", "log_var")
    assert selected.shape == (2, layout.output_slots, 8, 8)
    assert torch.all(selected == 7.0)
    assert torch.all(layout.select(prediction, "t2m", "mean") == 0.0)


def test_split_copre_tutte_le_coppie(layout: OutputLayout) -> None:
    prediction = torch.randn(1, layout.total_channels, 4, 4)
    parts = layout.split(prediction)
    assert set(parts) == {block.key for block in layout.blocks}
    assert all(tensor.shape[1] == layout.output_slots for tensor in parts.values())


def test_componente_inesistente_solleva(layout: OutputLayout) -> None:
    with pytest.raises(KeyError, match="assente"):
        layout.block("t2m", "occurrence_logit")


def test_variabile_inesistente_solleva(layout: OutputLayout) -> None:
    with pytest.raises(KeyError, match="non presente nel layout"):
        layout.head_of("msl")


def test_select_rifiuta_numero_di_canali_errato(layout: OutputLayout) -> None:
    with pytest.raises(ValueError, match="il layout ne richiede"):
        layout.select(torch.zeros(1, 3, 4, 4), "t2m", "mean")


def test_select_rifiuta_forma_non_quadridimensionale(layout: OutputLayout) -> None:
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        layout.select(torch.zeros(layout.total_channels, 4, 4), "t2m", "mean")


def test_testa_non_supportata_e_rifiutata() -> None:
    with pytest.raises(ValueError, match="Testa non supportata"):
        OutputLayout.from_targets([FakeTarget("t2m", "poisson")], 9)


def test_layout_senza_target_e_rifiutato() -> None:
    with pytest.raises(ValueError, match="almeno un target"):
        OutputLayout.from_targets([], 9)


def test_layout_con_orizzonte_nullo_e_rifiutato() -> None:
    with pytest.raises(ValueError, match="output_slots deve essere positivo"):
        OutputLayout.from_targets([FakeTarget("t2m", "gaussian")], 0)


def test_describe_e_tabellare(layout: OutputLayout) -> None:
    rows = layout.describe()
    assert len(rows) == len(layout.blocks)
    assert set(rows[0]) == {"variable", "head", "component", "start", "stop", "n_channels"}


# --------------------------------------------------------------------------- #
# Padding a multiplo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("height,width", [(261, 401), (96, 96), (100, 100), (7, 9)])
def test_padding_porta_a_multiplo_e_il_crop_ripristina(height: int, width: int) -> None:
    features = torch.randn(1, 2, height, width)
    padded, padding = pad_to_multiple(features, 8)
    assert padded.shape[-2] % 8 == 0
    assert padded.shape[-1] % 8 == 0
    restored = crop_padding(padded, padding)
    assert restored.shape == features.shape
    torch.testing.assert_close(restored, features)


def test_padding_non_tocca_dimensioni_gia_multiple() -> None:
    features = torch.randn(1, 2, 96, 128)
    padded, padding = pad_to_multiple(features, 8)
    assert padding == (0, 0)
    assert padded is features


def test_padding_su_dominio_reale_estende_al_minimo() -> None:
    """261 -> 264 e 401 -> 408: tre e sette righe/colonne aggiunte."""
    features = torch.randn(1, 1, *REAL_GRID)
    _, padding = pad_to_multiple(features, 8)
    assert padding == (3, 7)


@pytest.mark.parametrize("channels,expected", [(48, 8), (45, 5), (7, 7), (1, 1), (96, 8)])
def test_group_count_divide_esattamente(channels: int, expected: int) -> None:
    assert group_count(channels) == expected
    assert channels % group_count(channels) == 0


def test_group_count_rifiuta_canali_non_positivi() -> None:
    with pytest.raises(ValueError, match="positivo"):
        group_count(0)


# --------------------------------------------------------------------------- #
# Rete
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def small_network(layout: OutputLayout) -> DeepWeatherNet:
    """Rete ridotta: i test verificano forme e gradienti, non la capacita'."""
    return DeepWeatherNet(
        NetworkSpec(in_channels=12, base_channels=8, depth=2, blocks_per_level=1), layout
    )


def test_forward_su_crop_conserva_la_risoluzione(
    small_network: DeepWeatherNet, layout: OutputLayout
) -> None:
    output = small_network(torch.randn(2, 12, 96, 96))
    assert output.shape == (2, layout.total_channels, 96, 96)


def test_forward_su_dominio_reale_non_divisibile(
    small_network: DeepWeatherNet, layout: OutputLayout
) -> None:
    """La rete allenata su crop deve applicarsi a 261 x 401 senza modifiche."""
    with torch.no_grad():
        output = small_network(torch.randn(1, 12, *REAL_GRID))
    assert output.shape == (1, layout.total_channels, *REAL_GRID)


@pytest.mark.parametrize("size", [(24, 24), (32, 40), (56, 72)])
def test_forward_su_dimensioni_arbitrarie(
    small_network: DeepWeatherNet, layout: OutputLayout, size: tuple[int, int]
) -> None:
    with torch.no_grad():
        output = small_network(torch.randn(1, 12, *size))
    assert output.shape == (1, layout.total_channels, *size)


def test_forward_rifiuta_numero_di_canali_errato(small_network: DeepWeatherNet) -> None:
    with pytest.raises(ValueError, match="la rete ne attende"):
        small_network(torch.randn(1, 5, 32, 32))


def test_forward_rifiuta_input_senza_batch(small_network: DeepWeatherNet) -> None:
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        small_network(torch.randn(12, 32, 32))


def test_uscita_iniziale_e_neutra(small_network: DeepWeatherNet) -> None:
    """Con pesi finali azzerati la rete parte da una previsione costante, non da rumore."""
    with torch.no_grad():
        output = small_network(torch.randn(1, 12, 32, 32))
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)


def test_gradienti_raggiungono_tutti_i_parametri(small_network: DeepWeatherNet) -> None:
    output = small_network(torch.randn(1, 12, 32, 32))
    output.square().mean().backward()
    senza_gradiente = [
        name
        for name, parameter in small_network.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert senza_gradiente == []


def test_rete_e_deterministica_in_valutazione(small_network: DeepWeatherNet) -> None:
    small_network.eval()
    features = torch.randn(1, 12, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(small_network(features), small_network(features))


def test_build_network_usa_la_configurazione(config: Config, layout: OutputLayout) -> None:
    network = build_network(in_channels=20, layout=layout, model_config=config.model)
    assert network.spec.base_channels == config.model.base_channels
    assert network.spec.depth == config.model.depth
    assert network.spec.size_multiple == 2**config.model.depth


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"in_channels": 0}, "in_channels"),
        ({"base_channels": 0}, "base_channels"),
        ({"depth": 6}, "depth"),
        ({"blocks_per_level": 0}, "blocks_per_level"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_spec_rifiuta_iperparametri_invalidi(kwargs: dict, message: str) -> None:
    base = {"in_channels": 12, "base_channels": 8, "depth": 2, "blocks_per_level": 1}
    with pytest.raises(ValueError, match=message):
        NetworkSpec(**{**base, **kwargs})


def test_larghezza_dei_livelli_e_limitata() -> None:
    spec = NetworkSpec(in_channels=4, base_channels=256, depth=3, max_channels=384)
    assert spec.channels_at(0) == 256
    assert spec.channels_at(3) == 384  # limite raggiunto, non 2048
