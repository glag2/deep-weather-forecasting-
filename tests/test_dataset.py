"""Test del dataset e della costruzione dei bersagli.

Il lettore reale legge da Zarr; qui viene sostituito da uno finto in memoria che
espone la stessa interfaccia, cosi' i test girano senza dati scaricati e restano
veloci. Le proprieta' verificate sono quelle che, se sbagliate, non farebbero fallire
nulla: allineamento fra finestra di input e bersagli, maschere che escludono davvero,
e frazione di neve confinata in [0, 1] nonostante il rumore dei GRIB.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from dwf.config import Config
from dwf.data.dataset import (
    KEY_FEATURES,
    KEY_SLOT,
    DatasetError,
    WeatherWindowDataset,
    WindowBatchSampler,
    build_targets,
    target_specs,
)
from dwf.data.features import InputLayout, NormStats

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config.load(CONFIG_PATH, project_root=tmp_path)


@pytest.fixture
def layout(config: Config) -> InputLayout:
    return InputLayout.from_config(config)


def stats_neutre(layout: InputLayout) -> NormStats:
    nomi = list(layout.normalized_variables)
    return NormStats(
        mean=dict.fromkeys(nomi, 0.0),
        std=dict.fromkeys(nomi, 1.0),
        transform=dict.fromkeys(nomi, "identity"),
        scale=dict.fromkeys(nomi, 1.0),
        computed_on_split="train",
    )


class LettoreFinto:
    """Lettore in memoria con la stessa interfaccia di `ZarrWindowReader`."""

    def __init__(
        self,
        layout: InputLayout,
        n_slots: int = 60,
        altezza: int = 12,
        larghezza: int = 14,
    ) -> None:
        self._altezza = altezza
        self._larghezza = larghezza
        self.letture = 0
        self.richieste: list[tuple[int, int]] = []
        generatore = np.random.default_rng(0)
        # Gradiente spaziale nullo nell'origine: i test che leggono [0, 0] restano
        # leggibili, ma ritagli in posizioni diverse contengono valori diversi.
        gradiente = (
            np.arange(altezza, dtype=np.float32).reshape(-1, 1) * 0.01
            + np.arange(larghezza, dtype=np.float32).reshape(1, -1) * 0.001
        )
        self._dati = {}
        for indice, nome in enumerate(layout.dynamic_variables):
            base = np.arange(n_slots, dtype=np.float32).reshape(-1, 1, 1) + indice
            self._dati[nome] = (base + gradiente).astype(np.float32)
        # Precipitazione e neve con struttura realistica: molti zeri, code positive.
        pioggia = generatore.exponential(0.0005, size=(n_slots, altezza, larghezza))
        pioggia[pioggia < 0.0002] = 0.0
        self._dati["tp"] = pioggia.astype(np.float32)
        self._dati["sf"] = (pioggia * generatore.uniform(0.0, 1.05, pioggia.shape)).astype(
            np.float32
        )
        self.static = {
            nome: np.zeros((altezza, larghezza), dtype=np.float32)
            for nome in layout.static_variables
        }

    @property
    def shape(self) -> tuple[int, int]:
        return self._altezza, self._larghezza

    @property
    def latitudes(self) -> np.ndarray:
        return np.linspace(75.0, 10.0, self._altezza).astype(np.float32)

    def valid_time(self, slot_index: int) -> datetime:
        return datetime(2024, 1, 1, 6, tzinfo=UTC) + timedelta(hours=6 * int(slot_index))

    def read_window(self, start: int, length: int) -> dict[str, np.ndarray]:
        self.letture += 1
        self.richieste.append((start, length))
        return {nome: valori[start : start + length] for nome, valori in self._dati.items()}


def dataset_finto(
    config: Config, layout: InputLayout, **kwargs
) -> WeatherWindowDataset:
    lettore = LettoreFinto(layout)
    parametri = {"crop_size": None, "crops_per_window": 1, "seed": 0}
    parametri.update(kwargs)
    return WeatherWindowDataset(
        config, layout, stats_neutre(layout), [0, 1, 2], lettore, **parametri
    )


# --------------------------------------------------------------------------- #
# Bersagli
# --------------------------------------------------------------------------- #


def test_i_bersagli_coprono_tutte_le_teste(config: Config, layout: InputLayout) -> None:
    finestra = {
        "t2m": np.zeros((3, 2, 2), dtype=np.float32),
        "tp": np.zeros((3, 2, 2), dtype=np.float32),
        "sf": np.zeros((3, 2, 2), dtype=np.float32),
    }
    bersagli = build_targets(target_specs(config), finestra, stats_neutre(layout))
    assert set(bersagli) == {
        "target_t2m",
        "target_tp_occurrence",
        "target_tp_amount",
        "mask_tp_amount",
        "target_sf_fraction",
        "mask_sf_fraction",
    }


def test_l_occorrenza_segue_la_soglia(config: Config, layout: InputLayout) -> None:
    soglia = next(spec.threshold for spec in target_specs(config) if spec.name == "tp")
    assert soglia is not None
    pioggia = np.array([[[0.0, soglia * 0.5, soglia * 2]]], dtype=np.float32)
    finestra = {
        "t2m": np.zeros_like(pioggia),
        "tp": pioggia,
        "sf": np.zeros_like(pioggia),
    }
    bersagli = build_targets(target_specs(config), finestra, stats_neutre(layout))
    assert bersagli["target_tp_occurrence"].flatten().tolist() == [0.0, 0.0, 1.0]


def test_la_maschera_della_quantita_coincide_con_l_occorrenza(
    config: Config, layout: InputLayout
) -> None:
    """Addestrare la quantita' dove non piove spingerebbe il modello verso lo zero."""
    pioggia = np.array([[[0.0, 0.01]]], dtype=np.float32)
    finestra = {"t2m": np.zeros_like(pioggia), "tp": pioggia, "sf": np.zeros_like(pioggia)}
    bersagli = build_targets(target_specs(config), finestra, stats_neutre(layout))
    assert torch.equal(bersagli["mask_tp_amount"], bersagli["target_tp_occurrence"])


def test_la_frazione_di_neve_resta_in_zero_uno(
    config: Config, layout: InputLayout
) -> None:
    """I GRIB impacchettano `tp` e `sf` su griglie diverse e il rapporto puo' superare 1."""
    pioggia = np.array([[[0.001, 0.001]]], dtype=np.float32)
    neve = np.array([[[0.00104, 0.0005]]], dtype=np.float32)  # il primo eccede tp
    finestra = {"t2m": np.zeros_like(pioggia), "tp": pioggia, "sf": neve}
    bersagli = build_targets(target_specs(config), finestra, stats_neutre(layout))
    frazione = bersagli["target_sf_fraction"]
    assert float(frazione.max()) <= 1.0
    assert float(frazione.min()) >= 0.0
    assert float(frazione.flatten()[0]) == 1.0


def test_la_frazione_e_definita_solo_sopra_la_soglia(
    config: Config, layout: InputLayout
) -> None:
    """Sotto la soglia il rapporto sarebbe dominato dal rumore di quantizzazione."""
    pioggia = np.array([[[1e-8, 0.01]]], dtype=np.float32)
    neve = np.array([[[1e-8, 0.005]]], dtype=np.float32)
    finestra = {"t2m": np.zeros_like(pioggia), "tp": pioggia, "sf": neve}
    bersagli = build_targets(target_specs(config), finestra, stats_neutre(layout))
    assert bersagli["mask_sf_fraction"].flatten().tolist() == [0.0, 1.0]
    # Dove la maschera e' nulla il valore non deve essere NaN o infinito.
    assert torch.isfinite(bersagli["target_sf_fraction"]).all()


def test_una_variabile_target_mancante_e_segnalata(
    config: Config, layout: InputLayout
) -> None:
    with pytest.raises(DatasetError, match="Manca la variabile target"):
        build_targets(
            target_specs(config), {"t2m": np.zeros((1, 1, 1))}, stats_neutre(layout)
        )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def test_il_campione_ha_le_forme_dichiarate(config: Config, layout: InputLayout) -> None:
    dataset = dataset_finto(config, layout)
    campione = dataset[0]
    altezza, larghezza = dataset.reader.shape
    assert campione[KEY_FEATURES].shape == (layout.n_channels, altezza, larghezza)
    assert campione["target_t2m"].shape == (config.windows.output_slots, altezza, larghezza)


def test_il_campione_e_finito(config: Config, layout: InputLayout) -> None:
    campione = dataset_finto(config, layout)[0]
    for nome, valore in campione.items():
        if nome != KEY_SLOT:
            assert torch.isfinite(valore).all(), nome


def test_input_e_bersagli_non_si_sovrappongono(
    config: Config, layout: InputLayout
) -> None:
    """Se i bersagli entrassero nell'input il modello leggerebbe la risposta."""
    dataset = dataset_finto(config, layout)
    campione = dataset[0]
    prima = layout.dynamic_variables[0]
    # I dati finti valgono `slot + indice_variabile`, quindi il canale piu' recente
    # deve valere l'ultimo slot di input, non il primo bersaglio.
    recente = float(campione[KEY_FEATURES][layout.index_of(f"{prima}_t-0")][0, 0])
    assert recente == pytest.approx(config.windows.input_slots - 1)


def test_il_ritaglio_riduce_le_dimensioni(config: Config, layout: InputLayout) -> None:
    dataset = dataset_finto(config, layout, crop_size=8)
    campione = dataset[0]
    assert campione[KEY_FEATURES].shape[-2:] == (8, 8)
    assert campione["target_t2m"].shape[-2:] == (8, 8)


def test_un_ritaglio_piu_grande_del_dominio_e_rifiutato(
    config: Config, layout: InputLayout
) -> None:
    with pytest.raises(DatasetError, match="eccede il dominio"):
        dataset_finto(config, layout, crop_size=64)


def test_senza_finestre_ammesse_il_dataset_non_si_costruisce(
    config: Config, layout: InputLayout
) -> None:
    with pytest.raises(DatasetError, match="Nessuna finestra ammessa"):
        WeatherWindowDataset(
            config, layout, stats_neutre(layout), [], LettoreFinto(layout)
        )


def test_piu_ritagli_chiedono_la_stessa_finestra(
    config: Config, layout: InputLayout
) -> None:
    """E' cio' che rende efficace la cache del lettore, e quindi ammortizza la
    lettura, che la misura indica come costo dominante."""
    dataset = dataset_finto(config, layout, crop_size=8, crops_per_window=4)
    for indice in range(4):
        dataset[indice]
    assert len(set(dataset.reader.richieste)) == 1


def test_finestre_diverse_chiedono_letture_diverse(
    config: Config, layout: InputLayout
) -> None:
    dataset = dataset_finto(config, layout, crop_size=8, crops_per_window=2)
    dataset[0]
    dataset[2]
    assert len(set(dataset.reader.richieste)) == 2


def test_i_ritagli_della_stessa_finestra_sono_in_posizioni_diverse(
    config: Config, layout: InputLayout
) -> None:
    """Ritagli identici renderebbero il batch quattro copie dello stesso campione."""
    dataset = dataset_finto(config, layout, crop_size=8, crops_per_window=4)
    primi = [float(dataset[indice][KEY_FEATURES][0].mean()) for indice in range(4)]
    assert len(set(primi)) > 1


def test_la_lunghezza_tiene_conto_dei_ritagli(
    config: Config, layout: InputLayout
) -> None:
    dataset = dataset_finto(config, layout, crop_size=8, crops_per_window=3)
    assert len(dataset) == 3 * 3


def test_un_indice_fuori_intervallo_solleva(config: Config, layout: InputLayout) -> None:
    dataset = dataset_finto(config, layout)
    with pytest.raises(IndexError):
        dataset[len(dataset) + 5]


# --------------------------------------------------------------------------- #
# Campionatore
# --------------------------------------------------------------------------- #


def test_ogni_batch_resta_dentro_una_finestra() -> None:
    campionatore = WindowBatchSampler(n_windows=5, crops_per_window=4, batch_size=4)
    for lotto in campionatore:
        finestre = {indice // 4 for indice in lotto}
        assert len(finestre) == 1


def test_il_campionatore_copre_tutte_le_finestre() -> None:
    campionatore = WindowBatchSampler(n_windows=5, crops_per_window=2, batch_size=2)
    visti = {indice // 2 for lotto in campionatore for indice in lotto}
    assert visti == set(range(5))


def test_il_limite_di_batch_e_rispettato() -> None:
    campionatore = WindowBatchSampler(
        n_windows=100, crops_per_window=4, batch_size=4, max_batches=7
    )
    assert len(list(campionatore)) == 7
    assert len(campionatore) == 7


def test_senza_mescolamento_l_ordine_e_deterministico() -> None:
    primo = list(WindowBatchSampler(n_windows=4, crops_per_window=2, batch_size=2, shuffle=False))
    secondo = list(WindowBatchSampler(n_windows=4, crops_per_window=2, batch_size=2, shuffle=False))
    assert primo == secondo


def test_semi_diversi_danno_ordini_diversi() -> None:
    primo = list(WindowBatchSampler(n_windows=30, crops_per_window=1, batch_size=1, seed=1))
    secondo = list(WindowBatchSampler(n_windows=30, crops_per_window=1, batch_size=1, seed=2))
    assert primo != secondo
