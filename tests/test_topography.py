"""I descrittori topografici devono descrivere il terreno, non un artefatto di calcolo.

Ogni prova confronta il descrittore con un terreno di forma nota, perche' un errore di
segno o di asse non farebbe fallire nulla: produrrebbe soltanto un modello che impara
associazioni sbagliate.
"""

from __future__ import annotations

import numpy as np
import pytest

from dwf.data.topography import (
    GRAVITA,
    TopografiaError,
    _finestre,
    descrittori,
    nomi_descrittori,
    quota_in_metri,
    valida_raggi,
)


def _rampa(altezza: int = 12, larghezza: int = 16, passo: float = 10.0) -> np.ndarray:
    """Geopotenziale di un piano inclinato verso est di `passo` metri per punto."""
    metri = np.arange(larghezza, dtype=np.float32) * passo
    return np.broadcast_to(metri, (altezza, larghezza)).astype(np.float32) * GRAVITA


class TestConversione:
    def test_geopotenziale_diventa_metri(self) -> None:
        un_metro = np.full((3, 3), GRAVITA, dtype=np.float32)
        assert quota_in_metri(un_metro)[0, 0] == pytest.approx(1.0)

    def test_un_campo_con_l_asse_del_tempo_e_un_errore(self) -> None:
        with pytest.raises(TopografiaError, match="lat, lon"):
            quota_in_metri(np.zeros((2, 3, 3), dtype=np.float32))


class TestRaggi:
    def test_raggi_validi_passano(self) -> None:
        valida_raggi((1, 3, 9))

    @pytest.mark.parametrize("raggi", [(0,), (-1,), (3, 3), (9, 3)])
    def test_raggi_non_validi_sollevano(self, raggi: tuple[int, ...]) -> None:
        with pytest.raises(TopografiaError):
            valida_raggi(raggi)

    def test_i_nomi_seguono_l_ordine_dei_raggi(self) -> None:
        assert nomi_descrittori((2, 4)) == (
            "topo_slope_ns",
            "topo_slope_we",
            "topo_std_r2",
            "topo_tpi_r2",
            "topo_relief_r2",
            "topo_std_r4",
            "topo_tpi_r4",
            "topo_relief_r4",
        )

    def test_un_raggio_piu_grande_della_griglia_solleva(self) -> None:
        with pytest.raises(TopografiaError, match="griglia"):
            descrittori(np.zeros((6, 6), dtype=np.float32), (6,))


class TestDescrittori:
    def test_terreno_piatto_da_descrittori_tutti_nulli(self) -> None:
        campi = descrittori(np.full((10, 10), 500.0 * GRAVITA, dtype=np.float32), (2,))
        for nome, valore in campi.items():
            assert np.all(valore == 0.0), nome

    def test_su_una_rampa_verso_est_pende_solo_l_asse_est_ovest(self) -> None:
        campi = descrittori(_rampa(), (2,))
        # Standardizzata una pendenza costante resta nulla: cio' che conta e' che la
        # componente nord-sud sia identicamente piatta e la est-ovest non lo sia prima
        # della standardizzazione, verificata qui sui valori grezzi.
        quota = quota_in_metri(_rampa())
        pendenza_ns, pendenza_we = np.gradient(quota)
        assert np.all(pendenza_ns == 0.0)
        assert np.allclose(pendenza_we[:, 1:-1], 10.0)
        assert np.all(campi["topo_slope_ns"] == 0.0)

    def test_una_cima_isolata_ha_indice_di_posizione_massimo_al_centro(self) -> None:
        quota = np.zeros((21, 21), dtype=np.float32)
        quota[10, 10] = 1000.0
        campi = descrittori(quota * GRAVITA, (3,))
        tpi = campi["topo_tpi_r3"]
        assert np.argmax(tpi) == 10 * 21 + 10

    def test_la_rugosita_cresce_col_dislivello(self) -> None:
        liscio = np.zeros((20, 20), dtype=np.float32)
        liscio[:, ::2] = 10.0
        aspro = np.zeros((20, 20), dtype=np.float32)
        aspro[:, ::2] = 400.0
        # Il confronto va fatto sui valori grezzi: la standardizzazione, dividendo per la
        # dispersione del dominio, cancella per costruzione un fattore di scala globale.
        std_liscio = _finestre(quota_in_metri(liscio * GRAVITA), 2).std(axis=(-2, -1))
        std_aspro = _finestre(quota_in_metri(aspro * GRAVITA), 2).std(axis=(-2, -1))
        assert std_aspro.mean() > std_liscio.mean() * 30

    def test_il_rilievo_e_la_differenza_fra_massimo_e_minimo_del_vicinato(self) -> None:
        quota = np.zeros((15, 15), dtype=np.float32)
        quota[7, 7] = 300.0
        rilievo = _finestre(quota_in_metri(quota * GRAVITA), 2).max(
            axis=(-2, -1)
        ) - _finestre(quota_in_metri(quota * GRAVITA), 2).min(axis=(-2, -1))
        # Il rilievo vale il dislivello dentro il vicinato della cima, zero lontano.
        assert rilievo[7, 7] == pytest.approx(300.0, rel=1e-4)
        assert rilievo[0, 0] == pytest.approx(0.0)

    def test_ogni_descrittore_e_standardizzato(self) -> None:
        casuale = np.random.default_rng(0).normal(500.0, 200.0, size=(30, 40))
        campi = descrittori((casuale * GRAVITA).astype(np.float32), (1, 3))
        assert set(campi) == set(nomi_descrittori((1, 3)))
        for nome, valore in campi.items():
            assert valore.dtype == np.float32
            assert abs(float(valore.mean())) < 1e-4, nome
            assert float(valore.std()) == pytest.approx(1.0, rel=1e-3), nome

    def test_il_calcolo_e_ripetibile(self) -> None:
        quota = (np.random.default_rng(1).normal(0.0, 100.0, size=(18, 22)) * GRAVITA).astype(
            np.float32
        )
        primo = descrittori(quota, (2,))
        secondo = descrittori(quota, (2,))
        for nome in primo:
            assert np.array_equal(primo[nome], secondo[nome])
