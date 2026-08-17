"""Test degli andamenti climatici.

Il rischio di questa parte non e' sbagliare un'aritmetica: e' produrre una figura
convincente che i dati non sostengono. I test guardano quindi soprattutto i freni:
che una pendenza su due punti non venga dichiarata significativa, che un anno
incompleto non entri in una media annuale, e che il giudizio sulla serie dica la
verita' sulla lunghezza del periodo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from dwf.climate import (
    ANNI_MINIMI_PER_TENDENZA,
    Copertura,
    adatta_tendenza,
    ciclo_stagionale,
    confronto_interannuale,
    copertura,
    giudizio_sulla_serie,
    medie_annuali,
    medie_mensili,
    mesi_confrontabili,
)
from dwf.config import Config
from dwf.tables import SLOTS, cast_to_schema, write_table

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config.load(CONFIG_PATH, project_root=tmp_path)


def scrivi_catalogo(config: Config, righe: list[tuple[int, int, int]]) -> None:
    """Catalogo sintetico: ogni voce e' (anno, mese, quanti slot)."""
    from datetime import UTC, datetime

    dati: list[dict] = []
    indice = 0
    for anno, mese, quanti in righe:
        for _ in range(quanti):
            dati.append(
                {
                    "slot_index": indice,
                    "valid_time": datetime(anno, mese, 1, 6, tzinfo=UTC),
                    "year": anno,
                    "month": mese,
                    "day": 1,
                    "hour": 6,
                    "slot_of_day": 0,
                    "day_of_year": 1,
                    "source_month": f"{anno}-{mese:02d}",
                    "split": "train",
                    "usable": True,
                }
            )
            indice += 1
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    write_table(cast_to_schema(pl.DataFrame(dati), SLOTS), SLOTS, config.tables_dir)


# --------------------------------------------------------------------------- #
# Adattamento della tendenza
# --------------------------------------------------------------------------- #


def test_la_pendenza_di_una_retta_esatta_viene_recuperata() -> None:
    anni = np.arange(2000, 2020)
    valori = 3.0 + 0.05 * (anni - 2000)
    tendenza = adatta_tendenza(anni, valori)
    assert tendenza.pendenza == pytest.approx(0.05, abs=1e-9)
    assert tendenza.incertezza == pytest.approx(0.0, abs=1e-9)
    assert tendenza.n_anni == 20


def test_due_punti_non_producono_una_tendenza_credibile() -> None:
    """Per due punti passa esattamente una retta: l'errore non e' zero, e' indefinito.

    Dichiararlo nullo renderebbe significativa qualunque coppia di annate, che e'
    proprio l'errore che questa pagina deve evitare.
    """
    tendenza = adatta_tendenza(np.array([2024, 2025]), np.array([10.0, 12.0]))
    assert tendenza.pendenza == pytest.approx(2.0)
    assert np.isnan(tendenza.incertezza)
    assert not tendenza.significativa


def test_una_pendenza_netta_ma_su_pochi_anni_non_e_significativa() -> None:
    anni = np.arange(2020, 2025)
    valori = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
    tendenza = adatta_tendenza(anni, valori)
    assert tendenza.pendenza == pytest.approx(0.5)
    assert tendenza.n_anni < ANNI_MINIMI_PER_TENDENZA
    assert not tendenza.significativa


def test_una_pendenza_su_abbastanza_anni_puo_essere_significativa() -> None:
    generatore = np.random.default_rng(0)
    anni = np.arange(2000, 2020)
    valori = 0.04 * (anni - 2000) + generatore.normal(0, 0.05, anni.size)
    tendenza = adatta_tendenza(anni, valori)
    assert tendenza.n_anni >= ANNI_MINIMI_PER_TENDENZA
    assert tendenza.significativa


def test_una_serie_piatta_non_e_significativa() -> None:
    generatore = np.random.default_rng(1)
    anni = np.arange(2000, 2020)
    valori = 10.0 + generatore.normal(0, 0.5, anni.size)
    assert not adatta_tendenza(anni, valori).significativa


# --------------------------------------------------------------------------- #
# Giudizio sulla serie
# --------------------------------------------------------------------------- #


def test_il_giudizio_avverte_quando_manca_tutto() -> None:
    assert "Nessun dato" in giudizio_sulla_serie(None)


def test_il_giudizio_nega_la_tendenza_con_meno_di_due_anni() -> None:
    breve = Copertura(
        anni=(2024, 2025),
        anni_completi=(2025,),
        mesi_per_anno={2024: 1, 2025: 12},
        primo="2024-01-01",
        ultimo="2025-12-31",
        slot=1188,
    )
    testo = giudizio_sulla_serie(breve)
    assert "descriverebbe il caso" in testo


def test_il_giudizio_parla_di_differenze_fra_annate_con_pochi_anni() -> None:
    due = Copertura(
        anni=(2024, 2025),
        anni_completi=(2024, 2025),
        mesi_per_anno={2024: 12, 2025: 12},
        primo="2024-01-01",
        ultimo="2025-12-31",
        slot=2190,
    )
    assert "differenze fra" in giudizio_sulla_serie(due)


def test_il_giudizio_ammette_le_tendenze_solo_con_abbastanza_anni() -> None:
    lunga = Copertura(
        anni=tuple(range(2000, 2015)),
        anni_completi=tuple(range(2000, 2015)),
        mesi_per_anno=dict.fromkeys(range(2000, 2015), 12),
        primo="2000-01-01",
        ultimo="2014-12-31",
        slot=99999,
    )
    assert "calcolabili" in giudizio_sulla_serie(lunga)


# --------------------------------------------------------------------------- #
# Aggregazioni sul catalogo
# --------------------------------------------------------------------------- #


def test_la_copertura_distingue_gli_anni_completi(config: Config) -> None:
    scrivi_catalogo(
        config,
        [(2024, mese, 3) for mese in range(1, 13)] + [(2025, 1, 3), (2025, 2, 3)],
    )
    esito = copertura(config)
    assert esito is not None
    assert esito.anni == (2024, 2025)
    assert esito.anni_completi == (2024,)
    assert esito.mesi_per_anno == {2024: 12, 2025: 2}


def test_senza_catalogo_la_copertura_non_esplode(config: Config) -> None:
    assert copertura(config) is None
    assert medie_mensili(config) is None
    assert ciclo_stagionale(config) is None
    assert confronto_interannuale(config) is None
    assert medie_annuali(config) is None
    assert mesi_confrontabili(config) == []


def test_i_mesi_confrontabili_ignorano_quelli_troppo_scarsi(config: Config) -> None:
    """Un mese con pochi slot non rappresenta il mese: confrontarlo sarebbe rumore."""
    scrivi_catalogo(config, [(2024, 1, 90), (2025, 1, 5), (2024, 2, 90), (2025, 2, 90)])
    confrontabili = dict(mesi_confrontabili(config))
    assert 1 not in confrontabili
    assert confrontabili[2] == [2024, 2025]
