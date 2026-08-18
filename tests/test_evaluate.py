"""L'incertezza dichiarata dal modello viene giudicata, non solo prodotta.

Il progetto promette una previsione che dice quanto e' sicura. Prima di queste metriche la
testa gaussiana poteva annunciare qualunque varianza senza che un solo numero se ne
accorgesse: la NLL la penalizzava in addestramento, ma nessuna valutazione la leggeva.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dwf.config import Config
from dwf.data.features import NormStats
from dwf.evaluate import Prediction, incertezza_dichiarata, metrics_table

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config.load(CONFIG_PATH, project_root=tmp_path)


def _errori(sigma_vero: float, quanti: int = 40_000) -> np.ndarray:
    return np.random.default_rng(0).normal(0.0, sigma_vero, quanti).astype(np.float32)


class TestIncertezzaDichiarata:
    def test_un_modello_ben_tarato_ha_rapporto_uno_e_copertura_novanta(self) -> None:
        errore = _errori(0.5)
        sigma = np.full_like(errore, 0.5)

        esito = incertezza_dichiarata(sigma, errore, scala=1.0)

        assert esito["spread_skill_ratio"] == pytest.approx(1.0, abs=0.02)
        assert esito["coverage_90"] == pytest.approx(0.90, abs=0.02)

    def test_un_modello_troppo_sicuro_si_riconosce(self) -> None:
        """Barre d'errore troppo strette: e' il difetto che si vuole poter vedere."""
        errore = _errori(0.5)
        sigma = np.full_like(errore, 0.5 / 3.0)

        esito = incertezza_dichiarata(sigma, errore, scala=1.0)

        assert esito["spread_skill_ratio"] < 0.4
        assert esito["coverage_90"] < 0.5

    def test_un_modello_troppo_prudente_si_riconosce(self) -> None:
        errore = _errori(0.5)
        sigma = np.full_like(errore, 1.5)

        esito = incertezza_dichiarata(sigma, errore, scala=1.0)

        assert esito["spread_skill_ratio"] > 2.5
        assert esito["coverage_90"] > 0.99

    def test_la_dispersione_e_riportata_in_gradi(self) -> None:
        """Il numero leggibile e' in gradi: `scala` e' la deviazione standard di train."""
        errore = _errori(0.5)
        sigma = np.full_like(errore, 0.5)

        esito = incertezza_dichiarata(sigma, errore, scala=8.0)

        assert esito["spread_celsius"] == pytest.approx(4.0, abs=0.05)

    def test_un_errore_identicamente_nullo_non_produce_una_divisione_per_zero(self) -> None:
        esito = incertezza_dichiarata(np.ones(10), np.zeros(10), scala=1.0)

        assert np.isnan(esito["spread_skill_ratio"])
        assert esito["coverage_90"] == 1.0


def _statistiche() -> NormStats:
    """Deviazione standard di 8 gradi: l'ordine di grandezza vero della temperatura."""
    return NormStats(
        mean={"t2m": 280.0},
        std={"t2m": 8.0},
        transform={"t2m": "none"},
        scale={"t2m": 1.0},
        computed_on_split="train",
    )


class TestTabellaDelleMetriche:
    def _previsione(self, con_sigma: bool) -> Prediction:
        generatore = np.random.default_rng(1)
        quanti = 2_000
        vero = generatore.normal(0.0, 1.0, quanti).astype(np.float32)
        return Prediction(
            t2m_mean=(vero + generatore.normal(0.0, 0.3, quanti)).astype(np.float32),
            t2m_target=vero,
            tp_probability=generatore.uniform(0.0, 1.0, quanti).astype(np.float32),
            tp_occurrence=(generatore.uniform(0.0, 1.0, quanti) < 0.2).astype(np.float32),
            month=np.full(quanti, 6, dtype=np.int16),
            lead=np.repeat(np.arange(4, dtype=np.int16), quanti // 4),
            t2m_sigma=np.full(quanti, 0.3, dtype=np.float32) if con_sigma else None,
        )

    def test_le_metriche_di_incertezza_compaiono_per_scadenza(self, config: Config) -> None:
        tabella = metrics_table(
            self._previsione(True),
            _statistiche(),
            model="dwf", split="test", fold=0,
        )
        nomi = set(tabella["metric"].unique().to_list())

        assert {"spread_celsius", "spread_skill_ratio", "coverage_90"} <= nomi
        per_scadenza = tabella.filter(
            (tabella["metric"] == "coverage_90") & (tabella["lead_slot"] >= 0)
        )
        assert per_scadenza.height == 4

    def test_senza_sigma_le_metriche_non_vengono_inventate(self, config: Config) -> None:
        """La persistenza non dichiara un'incertezza: non deve comparire una riga finta."""
        tabella = metrics_table(
            self._previsione(False),
            _statistiche(),
            model="persistence", split="test", fold=0,
        )
        nomi = set(tabella["metric"].unique().to_list())

        assert "coverage_90" not in nomi
