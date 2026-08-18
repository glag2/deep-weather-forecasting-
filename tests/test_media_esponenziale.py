"""Cosa deve garantire la media esponenziale dei pesi.

Non basta che i numeri si muovano: la media e' utile solo se cade *dentro* la nuvola
delle posizioni recenti, se non sostituisce mai i pesi veri senza rimetterli a posto, e se
non puo' danneggiare l'addestramento quando fa peggio. I test qui sotto misurano queste
tre cose, non la sola esecuzione senza errori.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dwf.optim import MediaEsponenziale, OptimError


def _rete() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))


class TestCoefficiente:
    def test_un_coefficiente_fuori_intervallo_e_rifiutato(self) -> None:
        for valore in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(OptimError, match=r"\(0, 1\)"):
                MediaEsponenziale(_rete(), valore)

    def test_la_rampa_evita_che_la_media_resti_all_inizializzazione(self) -> None:
        """Con 0,999 fisso, dopo pochi passi la media sarebbe ancora il peso iniziale.

        La rampa e' la differenza fra un meccanismo utile e uno che sembra rotto: al primo
        passo il coefficiente vale 0,18, non 0,999.
        """
        media = MediaEsponenziale(_rete(), 0.999)

        assert media.coefficiente() == pytest.approx(1.0 / 10.0)
        media.passi = 1
        assert media.coefficiente() == pytest.approx(2.0 / 11.0)
        # A regime prevale il valore chiesto in configurazione, non la rampa.
        media.passi = 100_000
        assert media.coefficiente() == pytest.approx(0.999)


class TestMedia:
    def test_la_media_sta_fra_le_posizioni_viste(self) -> None:
        rete = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            rete.weight.fill_(0.0)
        media = MediaEsponenziale(rete, 0.5)

        with torch.no_grad():
            rete.weight.fill_(10.0)
        media.update(rete)
        primo = float(media.state_dict()["weight"])

        with torch.no_grad():
            rete.weight.fill_(20.0)
        media.update(rete)
        secondo = float(media.state_dict()["weight"])

        # Deve inseguire, non saltare: sempre dentro l'intervallo delle posizioni viste e
        # sempre nella direzione del movimento.
        assert 0.0 < primo < 10.0
        assert primo < secondo < 20.0

    def test_su_pesi_costanti_la_media_converge_a_quel_valore(self) -> None:
        rete = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            rete.weight.fill_(3.0)
        media = MediaEsponenziale(rete, 0.9)

        for _ in range(200):
            media.update(rete)

        assert float(media.state_dict()["weight"]) == pytest.approx(3.0, abs=1e-4)

    def test_la_media_e_una_copia_distinta_dai_pesi(self) -> None:
        """Se fosse un riferimento seguirebbe i pesi invece di mediarli, e il meccanismo
        sarebbe inerte senza dare alcun segno."""
        rete = nn.Linear(2, 2)
        media = MediaEsponenziale(rete, 0.9)
        prima = media.state_dict()["weight"].clone()

        with torch.no_grad():
            rete.weight.add_(100.0)

        assert torch.equal(media.ombra["weight"], prima)


class TestApplicazione:
    def test_i_pesi_veri_tornano_al_loro_posto(self) -> None:
        rete = _rete()
        media = MediaEsponenziale(rete, 0.9)
        with torch.no_grad():
            for parametro in rete.parameters():
                parametro.add_(1.0)
        originali = {nome: valore.clone() for nome, valore in rete.state_dict().items()}

        with media.applicata(rete):
            # Dentro il contesto la rete deve avere i pesi medi, cioe' quelli vecchi.
            assert not torch.equal(rete.state_dict()["0.weight"], originali["0.weight"])

        for nome, valore in rete.state_dict().items():
            assert torch.equal(valore, originali[nome]), nome

    def test_i_pesi_estratti_dentro_il_contesto_sopravvivono_all_uscita(self) -> None:
        """Difetto reale, trovato misurando: il checkpoint dichiarava i pesi medi e
        conteneva quelli grezzi.

        Su CPU `.cpu()` non copia e `.numpy()` restituisce una *vista* sulla memoria del
        tensore. Le matrici estratte dentro il contesto puntavano quindi ai buffer della
        rete, e il ripristino dei pesi veri le sovrascriveva prima che venissero scritte
        su disco. Il salvataggio deve clonare.
        """
        rete = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            rete.weight.fill_(0.0)
        media = MediaEsponenziale(rete, 0.99)
        with torch.no_grad():
            rete.weight.fill_(1.0)
        media.update(rete)
        atteso = float(media.ombra["weight"])

        with media.applicata(rete):
            estratti = {
                nome: valore.detach().clone().cpu().numpy()
                for nome, valore in rete.state_dict().items()
            }

        salvato = float(estratti["weight"].reshape(-1)[0])
        assert salvato == pytest.approx(atteso)
        # E deve essere davvero la media, non il peso vero, altrimenti il test passerebbe
        # anche con il difetto.
        assert salvato != pytest.approx(1.0)

    def test_un_errore_nella_valutazione_non_lascia_i_pesi_medi_installati(self) -> None:
        """Se la validazione solleva, l'addestramento riprenderebbe da una posizione che
        non ha mai raggiunto. Il ripristino deve avvenire comunque."""
        rete = _rete()
        media = MediaEsponenziale(rete, 0.9)
        with torch.no_grad():
            for parametro in rete.parameters():
                parametro.add_(1.0)
        originali = {nome: valore.clone() for nome, valore in rete.state_dict().items()}

        with pytest.raises(RuntimeError, match="validazione finta"), media.applicata(rete):
            raise RuntimeError("validazione finta")

        for nome, valore in rete.state_dict().items():
            assert torch.equal(valore, originali[nome]), nome
