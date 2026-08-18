"""Test di CMuon.

La proprieta' che conta non e' che l'ottimizzatore "funzioni": e' che l'aggiornamento sia
davvero ortogonalizzato (valori singolari vicini a 1) e che le matrici fuse siano trattate
a blocchi separati, perche' quello e' l'unico motivo per cui questo modulo esiste al posto
di `torch.optim.AdamW`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from dwf.config import Config
from dwf.optim import (
    CMuon,
    HybridOptimizer,
    OptimError,
    build_optimizer,
    ortogonalizza,
    split_parameters,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class TestOrtogonalizzazione:
    def test_i_valori_singolari_finiscono_in_una_banda_stretta(self) -> None:
        """L'iterazione non converge a 1, e non e' un difetto.

        I coefficienti quintici sono scelti per essere veloci, non esatti: misurato su
        questa implementazione il punto fisso sta fra 0,68 e 1,14. Basta, perche' cio' che
        serve e' che nessuna direzione avanzi centinaia di volte piu' di un'altra.
        """
        torch.manual_seed(0)
        valori = torch.linalg.svdvals(ortogonalizza(torch.randn(64, 32)))

        assert valori.min() > 0.6
        assert valori.max() < 1.4

    def test_una_matrice_mal_condizionata_viene_riequilibrata(self) -> None:
        """Il vantaggio di Muon: le direzioni debolissime avanzano come le forti.

        Misura secondaria e importante: con i cinque passi del paper un gradiente
        condizionato 1e4 resta condizionato 37, mentre con dieci scende a 1,7. Cinque
        passi non bastano sui casi estremi; il default resta cinque per fedelta' al paper
        e perche' i gradienti reali non sono cosi' degeneri, ma `ns_steps` e' un parametro.
        """
        torch.manual_seed(0)
        u, _, v = torch.linalg.svd(torch.randn(32, 32))
        scala = torch.logspace(0, -4, 32)
        malcondizionata = u @ torch.diag(scala) @ v

        assert scala.max() / scala.min() > 1e3
        con_cinque = torch.linalg.svdvals(ortogonalizza(malcondizionata, 5))
        con_dieci = torch.linalg.svdvals(ortogonalizza(malcondizionata, 10))

        assert con_cinque.max() / con_cinque.min() < 100
        assert con_dieci.max() / con_dieci.min() < 3

    def test_le_matrici_alte_e_larghe_sono_trattate_allo_stesso_modo(self) -> None:
        """La trasposizione interna e' un dettaglio di calcolo, non un cambio di semantica."""
        torch.manual_seed(0)
        matrice = torch.randn(48, 16)

        alta = ortogonalizza(matrice)
        larga = ortogonalizza(matrice.T.contiguous())

        assert torch.allclose(alta, larga.T, atol=1e-5)

    def test_un_tensore_non_matriciale_e_rifiutato(self) -> None:
        with pytest.raises(OptimError, match="matrice"):
            ortogonalizza(torch.zeros(4, 4, 4))


class TestPasso:
    def test_l_aggiornamento_riduce_una_perdita_quadratica(self) -> None:
        bersaglio = torch.eye(8)
        peso = nn.Parameter(torch.zeros(8, 8))
        ottimizzatore = CMuon([peso], lr=0.05)

        perdite = []
        for _ in range(40):
            perdita = ((peso - bersaglio) ** 2).mean()
            ottimizzatore.zero_grad()
            perdita.backward()
            ottimizzatore.step()
            perdite.append(float(perdita.detach()))

        assert perdite[-1] < perdite[0] / 10

    def test_un_gradiente_assente_non_muove_il_peso(self) -> None:
        peso = nn.Parameter(torch.ones(4, 4))
        CMuon([peso], lr=0.1).step()

        assert torch.equal(peso, torch.ones(4, 4))

    def test_il_weight_decay_e_disaccoppiato_dal_gradiente(self) -> None:
        """Senza gradiente il decadimento agisce comunque: e' la definizione di disaccoppiato."""
        peso = nn.Parameter(torch.ones(4, 4))
        peso.grad = torch.zeros(4, 4)
        CMuon([peso], lr=0.1, weight_decay=0.5).step()

        assert float(peso[0, 0]) == pytest.approx(0.95)

    def test_i_blocchi_di_una_matrice_fusa_sono_ortogonalizzati_separatamente(self) -> None:
        """Il difetto che CMuon corregge: un solo precondizionatore per tre funzioni.

        I due aggiornamenti devono differire, altrimenti spezzare la matrice non sta
        facendo nulla e il modulo sarebbe Muon con un nome diverso.
        """
        torch.manual_seed(0)
        gradiente = torch.randn(24, 8)

        insieme = CMuon._direzione_ortogonale(gradiente, 1, 5)
        spezzata = CMuon._direzione_ortogonale(gradiente, 3, 5)

        assert not torch.allclose(insieme, spezzata, atol=1e-3)
        # Ogni blocco, da solo, e' ortogonale.
        for blocco in spezzata.chunk(3, dim=0):
            valori = torch.linalg.svdvals(blocco / blocco.norm() * (8 ** 0.5))
            assert valori.min() > 0.3

    def test_una_prima_dimensione_non_divisibile_e_rifiutata(self) -> None:
        with pytest.raises(OptimError, match="divisibile"):
            CMuon([{"params": [nn.Parameter(torch.zeros(7, 4))], "chunks": 3}])

    def test_un_parametro_a_una_dimensione_e_rifiutato(self) -> None:
        """Le norme e i bias non hanno sottospazi da equalizzare: vanno ad AdamW."""
        with pytest.raises(OptimError, match="due dimensioni"):
            CMuon([nn.Parameter(torch.zeros(8))])

    def test_una_convoluzione_e_vista_come_matrice_di_filtri(self) -> None:
        peso = nn.Parameter(torch.zeros(16, 8, 3, 3))
        peso.grad = torch.randn(16, 8, 3, 3)
        CMuon([peso], lr=0.1).step()

        assert peso.shape == (16, 8, 3, 3)
        assert float(peso.abs().sum()) > 0.0


class TestRipartizione:
    def _rete(self) -> nn.Module:
        rete = nn.Module()
        rete.stem = nn.Conv2d(3, 8, 3)
        rete.attention = nn.MultiheadAttention(8, 2, batch_first=True)
        rete.dense = nn.Linear(8, 8)
        rete.norm = nn.LayerNorm(8)
        rete.output_conv = nn.Conv2d(8, 2, 1)
        return rete

    def test_il_qkv_fuso_finisce_nel_gruppo_a_tre_blocchi(self) -> None:
        gruppi, _ = split_parameters(self._rete())
        per_chunks = {gruppo["chunks"]: gruppo["params"] for gruppo in gruppi}

        assert 3 in per_chunks
        assert [tuple(p.shape) for p in per_chunks[3]] == [(24, 8)]

    def test_norme_bias_e_i_due_estremi_restano_ad_adamw(self) -> None:
        rete = self._rete()
        gruppi, resto = split_parameters(rete)
        ortogonalizzati = {id(p) for gruppo in gruppi for p in gruppo["params"]}

        assert id(rete.stem.weight) not in ortogonalizzati
        assert id(rete.output_conv.weight) not in ortogonalizzati
        assert id(rete.norm.weight) in {id(p) for p in resto}
        assert id(rete.dense.weight) in ortogonalizzati

    def test_ogni_parametro_finisce_in_un_solo_posto(self) -> None:
        rete = self._rete()
        gruppi, resto = split_parameters(rete)
        contati = [id(p) for gruppo in gruppi for p in gruppo["params"]] + [
            id(p) for p in resto
        ]

        assert sorted(contati) == sorted(id(p) for p in rete.parameters())
        assert len(contati) == len(set(contati))


class TestCostruzione:
    def test_la_configurazione_predefinita_usa_adamw(self, tmp_path: Path) -> None:
        """Finche' CMuon non ha vinto una misura, il default non cambia."""
        config = Config.load(CONFIG_PATH, project_root=tmp_path)
        assert config.training.optimizer == "adamw"

    def test_cmuon_produce_un_ottimizzatore_ibrido(self) -> None:
        rete = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        ottimizzatore = build_optimizer(rete, kind="cmuon", lr=1e-3, weight_decay=0.0)

        assert isinstance(ottimizzatore, HybridOptimizer)
        assert isinstance(ottimizzatore, torch.optim.Optimizer)

    def test_uno_scheduler_agisce_su_entrambi_i_rami(self) -> None:
        """Se lo scheduler vedesse un solo ramo, meta' della rete ignorerebbe il coseno."""
        rete = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        ottimizzatore = build_optimizer(rete, kind="cmuon", lr=1.0, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.ConstantLR(ottimizzatore, factor=0.5, total_iters=5)
        scheduler.step()

        assert len(ottimizzatore.param_groups) == 2
        for gruppo in ottimizzatore.param_groups:
            assert gruppo["lr"] == pytest.approx(0.5)

    def test_un_ottimizzatore_sconosciuto_e_rifiutato(self) -> None:
        with pytest.raises(OptimError, match="sconosciuto"):
            build_optimizer(nn.Linear(4, 4), kind="lion", lr=1e-3, weight_decay=0.0)

    def test_una_rete_senza_matrici_non_finge_di_usare_cmuon(self) -> None:
        with pytest.raises(OptimError, match="Nessuna matrice"):
            build_optimizer(nn.LayerNorm(8), kind="cmuon", lr=1e-3, weight_decay=0.0)

    def test_l_ibrido_aggiorna_sia_le_matrici_sia_le_norme(self) -> None:
        torch.manual_seed(0)
        rete = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        prima = [p.detach().clone() for p in rete.parameters()]
        ottimizzatore = build_optimizer(rete, kind="cmuon", lr=0.1, weight_decay=0.0)

        uscita = rete(torch.randn(4, 8)).pow(2).mean()
        ottimizzatore.zero_grad()
        uscita.backward()
        ottimizzatore.step()

        mossi = [not torch.equal(a, b) for a, b in zip(prima, rete.parameters(), strict=True)]
        assert all(mossi), "qualche parametro non e' stato aggiornato da nessuno dei due rami"

    def test_lo_stato_si_salva_e_si_ricarica(self) -> None:
        rete = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        ottimizzatore = build_optimizer(rete, kind="cmuon", lr=0.1, weight_decay=0.0)
        rete(torch.randn(4, 8)).pow(2).mean().backward()
        ottimizzatore.step()

        stato = ottimizzatore.state_dict()
        gemello = build_optimizer(rete, kind="cmuon", lr=0.1, weight_decay=0.0)
        gemello.load_state_dict(stato)

        assert set(stato) == {"muon", "adamw"}
        assert gemello.state_dict()["muon"]["state"].keys() == stato["muon"]["state"].keys()
