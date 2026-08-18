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
from torch import nn

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

    # Entrambe le architetture sono chieste per nome: dedurne una dal default renderebbe
    # il test dipendente da quale sia il default, che infatti e' cambiato.
    def con_architettura(nome: str) -> Config:
        return config.model_copy(
            update={"model": config.model.model_copy(update={"architecture": nome})}
        )

    a_u = build_network(con_architettura("unet"), layout, canali)
    globale = build_network(con_architettura("global"), layout, canali)

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


def test_un_addestramento_non_cancella_il_checkpoint_di_un_altra_architettura(
    tmp_path: Path,
) -> None:
    """Due corse concorrenti finiscono nella stessa cartella: la seconda cancellava la
    prima appena migliorava, in silenzio. E' costato due checkpoint da ore di calcolo."""
    import json

    from dwf.train import TrainingError, check_destination_free

    check_destination_free(tmp_path, "unet")

    (tmp_path / "metadata.json").write_text(
        json.dumps({"architecture": "global"}), encoding="utf-8"
    )
    with pytest.raises(TrainingError, match="sarebbe cancellato"):
        check_destination_free(tmp_path, "unet")

    check_destination_free(tmp_path, "global")


def test_un_checkpoint_senza_architettura_e_letto_come_convoluzionale(
    tmp_path: Path,
) -> None:
    """I checkpoint anteriori alla scelta dell'architettura non hanno la chiave: allora
    ne esisteva una sola, quindi l'assenza la identifica."""
    import json

    from dwf.train import TrainingError, check_destination_free

    (tmp_path / "metadata.json").write_text(json.dumps({"epoch": 3}), encoding="utf-8")
    check_destination_free(tmp_path, "unet")
    with pytest.raises(TrainingError, match="'unet'"):
        check_destination_free(tmp_path, "global")


class TestContestoCompresso:
    """Il ramo HCA: attenzione densa su token molto grossi."""

    def test_il_ramo_parte_come_un_non_ramo(self, layout: OutputLayout) -> None:
        """Iniezione a zero: l'uscita iniziale coincide con quella della rete senza ramo.

        Serve perche' un eventuale guadagno sia del meccanismo e non del diverso punto di
        partenza dei pesi.
        """
        torch.manual_seed(0)
        senza = _rete(layout)
        torch.manual_seed(0)
        con = _rete(layout, hca_pool=2)
        ingresso = torch.randn(2, 6, 16, 16)

        with torch.no_grad():
            # I due modelli condividono i pesi dei blocchi fini perche' il seme e' lo
            # stesso e il ramo compresso e' costruito dopo.
            con.load_state_dict(senza.state_dict(), strict=False)
            assert torch.allclose(senza(ingresso), con(ingresso), atol=1e-6)

    def test_il_ramo_cambia_l_uscita_quando_ha_pesi(self, layout: OutputLayout) -> None:
        """Se restasse ininfluente anche con pesi non nulli non starebbe facendo nulla."""
        torch.manual_seed(0)
        rete = _rete(layout, hca_pool=2)
        token = torch.randn(2, 16, 8, 8)

        # Il confronto e' sui token e non sull'uscita perche' `output_conv` parte a zero:
        # sull'uscita qualunque modifica interna sarebbe invisibile.
        with torch.no_grad():
            invariati = rete._contesto_compresso(token)
            nn.init.normal_(rete.hca_merge.weight, std=0.1)
            modificati = rete._contesto_compresso(token)

        assert torch.allclose(invariati, token, atol=1e-6)
        assert not torch.allclose(modificati, token, atol=1e-4)

    def test_il_ramo_aggiunge_pochi_parametri(self, layout: OutputLayout) -> None:
        """Il senso del ramo e' contesto quasi globale a costo trascurabile."""
        senza = _rete(layout, blocks=4)
        con = _rete(layout, blocks=4, hca_pool=4)

        aggiunti = con.n_parameters - senza.n_parameters
        assert 0 < aggiunti < senza.n_parameters / 2

    def test_una_griglia_di_token_piu_piccola_della_riduzione_non_fa_fallire(
        self, layout: OutputLayout
    ) -> None:
        """Il ritaglio d'addestramento e' piccolo e il dominio intero no: passino entrambi."""
        rete = _rete(layout, patch=4, hca_pool=16)

        uscita = rete(torch.zeros(1, 6, 16, 16))
        assert uscita.shape == (1, layout.total_channels, 16, 16)

    def test_una_riduzione_di_uno_e_rifiutata(self, layout: OutputLayout) -> None:
        with pytest.raises(ValueError, match="hca_pool"):
            _rete(layout, hca_pool=1)

    def test_gli_stessi_pesi_valgono_a_qualunque_risoluzione_anche_col_ramo(
        self, layout: OutputLayout
    ) -> None:
        """Il ramo non deve introdurre dipendenze dal numero di token."""
        rete = _rete(layout, hca_pool=2)
        rete.eval()

        with torch.no_grad():
            assert rete(torch.zeros(1, 6, 16, 16)).shape[-2:] == (16, 16)
            assert rete(torch.zeros(1, 6, 40, 24)).shape[-2:] == (40, 24)


class TestAttentionSink:
    def test_il_sink_aggiunge_un_solo_token_per_blocco(self, layout: OutputLayout) -> None:
        senza = _rete(layout, blocks=2)
        con = _rete(layout, blocks=2, attention_sink=True)

        assert con.n_parameters - senza.n_parameters == 2 * 16

    def test_un_sink_nullo_lascia_l_attenzione_invariata(self, layout: OutputLayout) -> None:
        """Inizializzato a zero il sink non e' neutro: assorbe peso di attenzione.

        Il valore del token e' nullo, ma il suo logit non lo e', quindi il denominatore
        del softmax cresce e l'uscita e' attenuata. E' esattamente l'effetto voluto, e va
        verificato che avvenga, non che non avvenga.
        """
        torch.manual_seed(0)
        senza = _rete(layout, blocks=1)
        torch.manual_seed(0)
        con = _rete(layout, blocks=1, attention_sink=True)
        con.load_state_dict(senza.state_dict(), strict=False)
        ingresso = torch.randn(1, 6, 16, 16)

        with torch.no_grad():
            uscita_senza = senza.blocks[0](senza.to_tokens(senza.stem(ingresso)))
            uscita_con = con.blocks[0](con.to_tokens(con.stem(ingresso)))

        assert not torch.allclose(uscita_senza, uscita_con, atol=1e-6)

    def test_il_sink_e_appreso(self, layout: OutputLayout) -> None:
        """Se non ricevesse gradiente sarebbe una costante inutile.

        La convoluzione d'uscita e' inizializzata a zero perche' la previsione iniziale
        sia neutra, e questo azzera il gradiente di *tutta* la rete al primo passo: per
        misurare il sink va quindi resa non nulla.
        """
        rete = _rete(layout, blocks=1, attention_sink=True)
        nn.init.normal_(rete.output_conv.weight, std=0.1)
        rete(torch.randn(1, 6, 16, 16)).pow(2).mean().backward()

        assert rete.blocks[0].sink.grad is not None
        assert float(rete.blocks[0].sink.grad.abs().sum()) > 0.0


def test_channels_last_non_cambia_il_risultato(layout: OutputLayout) -> None:
    """La disposizione in memoria e' un'ottimizzazione, non un cambio di modello.

    Vale l'8% di tempo per passo misurato su questa CPU: se cambiasse anche i numeri
    non sarebbe un'ottimizzazione, sarebbe un'altra rete.
    """
    torch.manual_seed(0)
    rete = _rete(layout)
    ingresso = torch.randn(2, 6, 16, 16)

    with torch.no_grad():
        atteso = rete(ingresso)
        ottenuto = rete.to(memory_format=torch.channels_last)(
            ingresso.contiguous(memory_format=torch.channels_last)
        )

    assert torch.allclose(atteso, ottenuto, atol=1e-5)


def test_la_configurazione_accende_channels_last_per_default(config: Config) -> None:
    """Misurato piu' veloce su entrambe le architetture, quindi acceso di default."""
    assert config.training.channels_last is True


def test_l_andamento_costante_non_crea_uno_scheduler() -> None:
    import torch

    from dwf.train import build_scheduler

    peso = torch.nn.Parameter(torch.zeros(1))
    ottimizzatore = torch.optim.AdamW([peso], lr=1.0)
    assert build_scheduler(ottimizzatore, "constant", 0.05, 100) is None


def test_l_andamento_a_coseno_sale_e_scende() -> None:
    """Il passo deve partire piccolo, arrivare a quello pieno alla fine del riscaldamento
    e ridursi verso il termine, senza mai annullarsi."""
    import torch

    from dwf.train import build_scheduler

    peso = torch.nn.Parameter(torch.zeros(1))
    ottimizzatore = torch.optim.AdamW([peso], lr=1.0)
    scheduler = build_scheduler(ottimizzatore, "cosine", 0.1, 100)
    assert scheduler is not None

    letture = []
    for _ in range(100):
        letture.append(ottimizzatore.param_groups[0]["lr"])
        ottimizzatore.step()
        scheduler.step()

    assert letture[0] < letture[9]
    assert letture[9] == pytest.approx(1.0)
    assert letture[-1] < letture[50] < letture[9]
    assert letture[-1] > 0.0


def test_un_andamento_sconosciuto_e_rifiutato() -> None:
    import torch

    from dwf.train import TrainingError, build_scheduler

    peso = torch.nn.Parameter(torch.zeros(1))
    ottimizzatore = torch.optim.AdamW([peso], lr=1.0)
    with pytest.raises(TrainingError, match="non riconosciuto"):
        build_scheduler(ottimizzatore, "lineare", 0.05, 100)
