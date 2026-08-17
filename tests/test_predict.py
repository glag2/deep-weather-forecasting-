"""Test della previsione e della conversione in unita' fisiche.

Il rischio qui non e' che la rete sbagli, ma che l'uscita venga interpretata male: una
probabilita' scambiata per una quantita', millimetri lasciati in metri, la neve trattata
come indipendente dalla pioggia. Sono errori che non fanno fallire nulla e producono
previsioni plausibili ma sbagliate, quindi i test guardano le relazioni fra le grandezze,
non i valori.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from dwf.config import Config
from dwf.data.features import InputLayout, NormStats
from dwf.models.heads import OutputLayout
from dwf.predict import (
    PredictionError,
    forecast_to_table,
    latest_usable_start,
    predict_window,
    summarize,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config.load(CONFIG_PATH, project_root=tmp_path)


@pytest.fixture
def layout(config: Config) -> InputLayout:
    return InputLayout.from_config(config)


@pytest.fixture
def output_layout(config: Config) -> OutputLayout:
    return OutputLayout.from_targets(config.targets, config.windows.output_slots)


def stats_note(layout: InputLayout) -> NormStats:
    """Statistiche con media e scala non banali: se una denormalizzazione manca, si vede."""
    nomi = list(layout.normalized_variables)
    return NormStats(
        mean=dict.fromkeys(nomi, 5.0),
        std=dict.fromkeys(nomi, 2.0),
        transform=dict.fromkeys(nomi, "identity"),
        scale=dict.fromkeys(nomi, 1.0),
        computed_on_split="train",
    )


class LettoreFinto:
    """Lettore in memoria con l'interfaccia di `ZarrWindowReader`.

    Gli istanti seguono le ore realmente configurate (06, 12, 18), quindi gli slot
    **non** sono equispaziati: fra le 18 e le 06 del giorno dopo passano 12 ore, non 6.
    Un finto a passo uniforme nasconderebbe gli errori di calcolo delle scadenze.
    """

    def __init__(
        self,
        layout: InputLayout,
        altezza: int = 8,
        larghezza: int = 10,
        slot_hours: tuple[int, ...] = (6, 12, 18),
    ) -> None:
        self._altezza, self._larghezza = altezza, larghezza
        self._slot_hours = tuple(slot_hours)
        generatore = np.random.default_rng(0)
        self._dati = {
            nome: generatore.normal(size=(60, altezza, larghezza)).astype(np.float32)
            for nome in layout.dynamic_variables
        }
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

    @property
    def longitudes(self) -> np.ndarray:
        return np.linspace(-40.0, 60.0, self._larghezza).astype(np.float32)

    def valid_time(self, slot_index: int) -> datetime:
        per_giorno = len(self._slot_hours)
        giorno, resto = divmod(int(slot_index), per_giorno)
        return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(
            days=giorno, hours=self._slot_hours[resto]
        )

    def read_window(self, start: int, length: int) -> dict[str, np.ndarray]:
        return {nome: valori[start : start + length] for nome, valori in self._dati.items()}


class ReteCostante(torch.nn.Module):
    """Rete che restituisce valori scelti: cosi' l'atteso e' calcolabile a mano.

    Con una rete vera l'unico controllo possibile sarebbe "il numero e' finito"; qui
    invece si puo' verificare che le formule di conversione siano quelle dichiarate.
    """

    def __init__(self, layout: OutputLayout, altezza: int, larghezza: int, valore: float):
        super().__init__()
        self._layout = layout
        self._forma = (1, layout.total_channels, altezza, larghezza)
        self._valore = valore

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        return torch.full(self._forma, self._valore)


def prevedi(config, layout, output_layout, valore=0.0, altezza=8, larghezza=10):
    lettore = LettoreFinto(layout, altezza, larghezza, tuple(config.time.slot_hours))
    rete = ReteCostante(output_layout, altezza, larghezza, valore)
    return predict_window(
        config, rete, layout, output_layout, stats_note(layout), lettore, 0
    )


# --------------------------------------------------------------------------- #
# Scelta della finestra
# --------------------------------------------------------------------------- #


def test_si_sceglie_la_finestra_piu_recente(config: Config) -> None:
    utilizzabili = np.ones(50, dtype=bool)
    assert latest_usable_start(config, utilizzabili) == 50 - config.windows.input_slots


def test_una_finestra_interrotta_viene_scartata(config: Config) -> None:
    utilizzabili = np.ones(50, dtype=bool)
    # Un buco vicino alla fine costringe a retrocedere: la finestra deve essere intera.
    utilizzabili[45] = False
    inizio = latest_usable_start(config, utilizzabili)
    assert inizio + config.windows.input_slots <= 45


def test_i_target_mancanti_non_impediscono_la_previsione(config: Config) -> None:
    """Nella previsione operativa i giorni previsti non esistono ancora, per definizione."""
    utilizzabili = np.zeros(50, dtype=bool)
    utilizzabili[:21] = True
    assert latest_usable_start(config, utilizzabili) == 0


def test_senza_finestre_complete_si_solleva(config: Config) -> None:
    utilizzabili = np.zeros(50, dtype=bool)
    utilizzabili[:5] = True
    with pytest.raises(PredictionError, match="consecutivi"):
        latest_usable_start(config, utilizzabili)


# --------------------------------------------------------------------------- #
# Forma e coerenza della previsione
# --------------------------------------------------------------------------- #


def test_la_previsione_copre_tutte_le_scadenze(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout)
    assert previsione.n_lead == config.windows.output_slots
    for campo in (
        previsione.t2m_mean,
        previsione.precip_probability,
        previsione.snow_probability,
    ):
        assert campo.shape == (config.windows.output_slots, 8, 10)


def test_gli_istanti_previsti_seguono_l_inizializzazione(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout)
    assert previsione.valid_times[0] > previsione.init_time
    assert list(previsione.valid_times) == sorted(previsione.valid_times)
    # Solo le ore configurate, e nessuna ripetizione.
    assert {istante.hour for istante in previsione.valid_times} <= set(config.time.slot_hours)
    assert len(set(previsione.valid_times)) == config.windows.output_slots


def test_gli_istanti_rispettano_la_spaziatura_irregolare(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    """Fra le 18 e le 06 passano 12 ore: un passo fisso di 6 sarebbe sbagliato."""
    previsione = prevedi(config, layout, output_layout)
    passi = {
        int((dopo - prima).total_seconds() // 3600)
        for prima, dopo in zip(
            previsione.valid_times, previsione.valid_times[1:], strict=False
        )
    }
    assert passi == {6, 12}


def test_le_coordinate_vengono_dal_lettore(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    """Se venissero dalla configurazione, un dominio ritagliato darebbe assi sbagliati."""
    previsione = prevedi(config, layout, output_layout, altezza=8, larghezza=10)
    assert previsione.latitudes.size == 8
    assert previsione.longitudes.size == 10


def test_tutto_e_finito(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout, valore=3.0)
    for campo in (
        previsione.t2m_mean,
        previsione.t2m_std,
        previsione.precip_probability,
        previsione.precip_amount,
        previsione.snow_probability,
    ):
        assert np.isfinite(campo).all()


# --------------------------------------------------------------------------- #
# Conversione in unita' fisiche
# --------------------------------------------------------------------------- #


def test_la_temperatura_e_denormalizzata(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout, valore=0.0)
    # media 5.0 e deviazione 2.0: un'uscita normalizzata nulla vale esattamente 5.0.
    assert np.allclose(previsione.t2m_mean, 5.0)


def test_l_incertezza_e_riportata_in_kelvin(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout, valore=0.0)
    # log_var nullo significa deviazione normalizzata 1, quindi 1 x std = 2 K.
    assert np.allclose(previsione.t2m_std, 2.0)


def test_l_incertezza_e_positiva(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    for valore in (-8.0, 0.0, 8.0):
        previsione = prevedi(config, layout, output_layout, valore=valore)
        assert (previsione.t2m_std > 0).all()


def test_le_probabilita_restano_in_zero_uno(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    for valore in (-20.0, 0.0, 20.0):
        previsione = prevedi(config, layout, output_layout, valore=valore)
        for campo in (previsione.precip_probability, previsione.snow_probability):
            assert campo.min() >= 0.0
            assert campo.max() <= 1.0


def test_a_logit_nullo_la_probabilita_e_un_mezzo(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout, valore=0.0)
    assert np.allclose(previsione.precip_probability, 0.5)


def test_la_neve_non_supera_mai_la_pioggia(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    """Non puo' nevicare piu' spesso di quanto precipiti: la neve e' una quota della pioggia."""
    for valore in (-5.0, -1.0, 0.0, 1.0, 5.0):
        previsione = prevedi(config, layout, output_layout, valore=valore)
        assert (previsione.snow_probability <= previsione.precip_probability + 1e-6).all()


def test_la_neve_e_il_prodotto_dichiarato(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout, valore=0.0)
    # P(neve) = P(precipita) x quota di neve = 0.5 x 0.5
    assert np.allclose(previsione.snow_probability, 0.25)


def test_la_quantita_non_e_mai_negativa(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    # Con statistiche di media 5 e uscita molto negativa, la quantita' denormalizzata
    # sarebbe negativa: deve essere tagliata a zero, non propagata.
    previsione = prevedi(config, layout, output_layout, valore=-10.0)
    assert (previsione.precip_amount >= 0.0).all()


def test_la_quantita_e_pesata_dalla_probabilita(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    """Il valore atteso, non la quantita' condizionata: altrimenti piove sempre."""
    previsione = prevedi(config, layout, output_layout, valore=0.0)
    # quantita' denormalizzata 5 m, per probabilita' 0.5, per 1000 mm/m.
    assert np.allclose(previsione.precip_amount, 5.0 * 0.5 * 1000.0)


# --------------------------------------------------------------------------- #
# Tabelle
# --------------------------------------------------------------------------- #


def test_la_tabella_ha_una_riga_per_punto_e_scadenza(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout)
    tabella = forecast_to_table(previsione)
    assert tabella.height == config.windows.output_slots * 8 * 10


def test_il_sottocampionamento_riduce_le_righe(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout)
    intera = forecast_to_table(previsione, stride=1)
    ridotta = forecast_to_table(previsione, stride=2)
    assert ridotta.height == config.windows.output_slots * 4 * 5
    assert ridotta.height < intera.height


def test_le_scadenze_in_ore_sono_coerenti(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout)
    tabella = forecast_to_table(previsione)
    ore = (
        tabella.unique(subset=["lead_slot"])
        .sort("lead_slot")
        .get_column("lead_hours")
        .to_list()
    )
    atteso = [
        int((istante - previsione.init_time).total_seconds() // 3600)
        for istante in previsione.valid_times
    ]
    assert ore == atteso
    # Le scadenze crescono e coprono i tre giorni richiesti.
    assert ore == sorted(ore)
    assert max(ore) == 72


def test_la_tabella_copre_l_intero_dominio(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout)
    tabella = forecast_to_table(previsione)
    assert tabella.get_column("latitude").n_unique() == 8
    assert tabella.get_column("longitude").n_unique() == 10


def test_il_riepilogo_ha_una_riga_per_scadenza(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    previsione = prevedi(config, layout, output_layout)
    riepilogo = summarize(previsione)
    assert riepilogo.height == config.windows.output_slots
    assert riepilogo.get_column("lead_slot").to_list() == list(
        range(config.windows.output_slots)
    )


def test_il_riepilogo_riporta_i_gradi_celsius(
    config: Config, layout: InputLayout, output_layout: OutputLayout
) -> None:
    # La conversione avviene una sola volta, in lettura: qui la previsione e' gia' in
    # gradi Celsius e sottrarre ancora sarebbe una doppia conversione.
    previsione = prevedi(config, layout, output_layout, valore=0.0)
    riepilogo = summarize(previsione)
    assert riepilogo.get_column("t2m_mean_celsius").to_numpy() == pytest.approx(5.0)
