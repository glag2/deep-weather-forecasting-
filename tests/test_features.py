"""Test del layout dei canali di input e della normalizzazione.

L'ordine dei canali e' un contratto silenzioso: scambiarne due non fa fallire nulla,
produce solo un modello che impara associazioni sbagliate. Questi test lo bloccano.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from dwf.config import Config
from dwf.data.features import (
    GROUP_STATE,
    GROUP_STATIC,
    GROUP_TENDENCY,
    GROUP_TIME,
    GROUP_TOPOGRAPHY,
    GROUP_WIND,
    WIND_SPEED,
    FeatureError,
    InputLayout,
    NormStats,
    SlotReader,
    apply_transform,
    build_input_tensor,
    compute_norm_stats,
    invert_transform,
    latitude_channels,
    store_offset_of,
    to_working_units,
    transform_of,
    wind_speed,
)
from dwf.tables import CHANNELS, validate_schema

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config.load(CONFIG_PATH, project_root=tmp_path)


@pytest.fixture
def layout(config: Config) -> InputLayout:
    return InputLayout.from_config(config)


def finestra_finta(layout: InputLayout, altezza: int = 4, larghezza: int = 5) -> dict:
    """Finestra deterministica: ogni variabile cresce di 1 a ogni slot."""
    finestra = {}
    for indice, nome in enumerate(layout.dynamic_variables):
        base = np.arange(layout.input_slots, dtype=np.float32).reshape(-1, 1, 1)
        finestra[nome] = np.broadcast_to(
            base + indice, (layout.input_slots, altezza, larghezza)
        ).copy()
    return finestra


def campi_statici(layout: InputLayout, altezza: int = 4, larghezza: int = 5) -> dict:
    return {
        nome: np.zeros((altezza, larghezza), dtype=np.float32)
        for nome in layout.static_variables
    }


def stats_neutre(layout: InputLayout) -> NormStats:
    """Statistiche che non alterano i valori, per isolare la logica di stacking."""
    nomi = list(layout.normalized_variables)
    return NormStats(
        mean=dict.fromkeys(nomi, 0.0),
        std=dict.fromkeys(nomi, 1.0),
        transform=dict.fromkeys(nomi, "identity"),
        scale=dict.fromkeys(nomi, 1.0),
        computed_on_split="train",
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def test_il_conteggio_dei_canali_e_coerente_coi_gruppi(layout: InputLayout) -> None:
    atteso = (
        len(layout.dynamic_variables) * layout.input_slots
        + len(layout.dynamic_variables) * len(layout.tendency_lags)
        + layout.input_slots
        + len(layout.static_variables)
        + 2
        + 4
    )
    assert layout.n_channels == atteso


def test_la_configurazione_di_default_conserva_i_245_canali(layout: InputLayout) -> None:
    """Il checkpoint a piena scala e' addestrato su questo layout esatto."""
    assert layout.n_channels == 245


def test_i_livelli_di_pressione_si_aggiungono_in_coda(config: Config) -> None:
    """Abilitandoli cambia il numero di canali, ma non l'indice di quelli esistenti."""
    dati = config.model_dump()
    dati["variables"]["pressure"] = [
        {"variable": "geopotential", "level": 500},
        {"variable": "temperature", "level": 850},
        {"variable": "temperature", "level": 500},
        {"variable": "specific_humidity", "level": 700},
    ]
    esteso = InputLayout.from_config(Config.model_validate(dati))
    base = InputLayout.from_config(config)
    # 4 variabili in piu': uno stato per slot di input e una tendenza per ritardo.
    per_variabile = base.input_slots + len(base.tendency_lags)
    assert esteso.n_channels == base.n_channels + 4 * per_variabile
    assert esteso.dynamic_variables[: len(base.dynamic_variables)] == base.dynamic_variables
    aggiunti = {canale.source_variable for canale in esteso.channels} - {
        canale.source_variable for canale in base.channels
    }
    assert aggiunti == {"z500", "t850", "t500", "q700"}


def test_gli_indici_sono_consecutivi_e_unici(layout: InputLayout) -> None:
    assert [canale.index for canale in layout.channels] == list(range(layout.n_channels))


def test_i_nomi_dei_canali_sono_unici(layout: InputLayout) -> None:
    nomi = [canale.name for canale in layout.channels]
    assert len(nomi) == len(set(nomi))


def test_lo_stato_e_in_ordine_cronologico(layout: InputLayout) -> None:
    """Il primo canale di ogni variabile e' il piu' vecchio, l'ultimo il piu' recente."""
    stato = [c for c in layout.channels if c.group == GROUP_STATE]
    prima_variabile = [c for c in stato if c.source_variable == layout.dynamic_variables[0]]
    assert prima_variabile[0].lag == layout.input_slots - 1
    assert prima_variabile[-1].lag == 0


def test_la_tabella_dei_canali_rispetta_lo_schema(layout: InputLayout) -> None:
    validate_schema(layout.to_table(), CHANNELS)


def test_una_tendenza_piu_lunga_della_finestra_e_rifiutata(config: Config) -> None:
    """La configurazione la blocca gia' in validazione; il layout resta come guardia
    per chi costruisse un `Config` senza passare da li'."""
    dati = config.model_dump()
    dati["features"]["tendency_lags"] = [config.windows.input_slots]
    with pytest.raises(ValidationError, match="non minore di input_slots"):
        Config.model_validate(dati)


def test_il_vento_richiede_le_componenti(config: Config) -> None:
    dati = config.model_dump()
    dati["variables"]["instantaneous"] = ["2m_temperature"]
    dati["targets"] = [{"name": "t2m", "head": "gaussian"}]
    with pytest.raises(FeatureError, match="include_wind_speed"):
        InputLayout.from_config(Config.model_validate(dati))


class TestDescrittoriTopografici:
    """Sono spenti per default: accendendoli devono comparire in coda e valere qualcosa."""

    @staticmethod
    def _acceso(config: Config, raggi: list[int]) -> Config:
        dati = config.model_dump()
        dati["features"]["topographic_radii"] = raggi
        return Config.model_validate(dati)

    def test_spenti_per_default(self, layout: InputLayout) -> None:
        assert layout.topographic_radii == ()
        assert layout.indices_of_group(GROUP_TOPOGRAPHY) == []

    def test_si_aggiungono_in_coda_senza_spostare_gli_altri(self, config: Config) -> None:
        base = InputLayout.from_config(config)
        esteso = InputLayout.from_config(self._acceso(config, [1, 3]))
        # Due pendenze piu' tre descrittori per ciascun raggio.
        assert esteso.n_channels == base.n_channels + 2 + 3 * 2
        # I descrittori stanno accanto agli altri campi invarianti, non in coda: cio' che
        # deve restare invariato e' l'ordine relativo di tutti i canali preesistenti.
        senza_topografia = [
            canale.name for canale in esteso.channels if canale.group != GROUP_TOPOGRAPHY
        ]
        assert senza_topografia == [canale.name for canale in base.channels]

    def test_senza_la_quota_fra_le_statiche_e_un_errore(self, config: Config) -> None:
        dati = self._acceso(config, [2]).model_dump()
        dati["variables"]["static"] = ["land_sea_mask"]
        with pytest.raises(FeatureError, match="descrittori topografici"):
            InputLayout.from_config(Config.model_validate(dati))

    def test_i_canali_finiscono_nel_tensore(self, config: Config) -> None:
        esteso = InputLayout.from_config(self._acceso(config, [2]))
        statici = campi_statici(esteso, altezza=9, larghezza=9)
        # Una cima isolata: senza descrittori nessun canale la distingue dal piano.
        statici["z"] = np.zeros((9, 9), dtype=np.float32)
        statici["z"][4, 4] = 1000.0 * 9.80665
        tensore = build_input_tensor(
            esteso,
            finestra_finta(esteso, altezza=9, larghezza=9),
            stats_neutre(esteso),
            static_fields=statici,
            latitudes=np.linspace(75.0, 10.0, 9),
            reference_time=datetime(2025, 3, 1, 12),
            slot_hours=config.time.slot_hours,
        )
        indici = esteso.indices_of_group(GROUP_TOPOGRAPHY)
        assert len(indici) == 5
        posizione = esteso.index_of("topo_tpi_r2")
        assert int(np.argmax(tensore[posizione])) == 4 * 9 + 4
        assert np.isfinite(tensore).all()

    def test_senza_il_campo_di_quota_il_tensore_non_si_costruisce(self, config: Config) -> None:
        esteso = InputLayout.from_config(self._acceso(config, [2]))
        with pytest.raises(FeatureError, match="statico"):
            build_input_tensor(
                esteso,
                finestra_finta(esteso),
                stats_neutre(esteso),
                static_fields={},
                latitudes=np.linspace(75.0, 10.0, 4),
                reference_time=datetime(2025, 3, 1, 12),
                slot_hours=config.time.slot_hours,
            )


def test_indice_di_un_canale_inesistente(layout: InputLayout) -> None:
    with pytest.raises(KeyError, match="Canale assente"):
        layout.index_of("non_esiste")


# --------------------------------------------------------------------------- #
# Trasformazioni
# --------------------------------------------------------------------------- #


def test_la_scala_rende_efficace_il_log1p() -> None:
    """In metri `log1p` e' quasi l'identita': e' la scala a comprimere la coda."""
    pioggia_metri = np.array([0.0001, 0.07], dtype=np.float32)
    senza_scala = apply_transform(pioggia_metri, "log1p", 1.0)
    con_scala = apply_transform(pioggia_metri, "log1p", 1000.0)

    # Senza scala la trasformazione lascia i valori quasi immutati: non comprime nulla.
    assert np.allclose(senza_scala, pioggia_metri, rtol=0.04)
    # Con la scala, 0.1 mm e 70 mm finiscono in un intervallo utilizzabile.
    assert con_scala[0] == pytest.approx(np.log1p(0.1), rel=1e-5)
    assert con_scala[1] > 4.0


def test_la_trasformazione_e_invertibile() -> None:
    valori = np.array([0.0, 0.0001, 0.01, 0.07], dtype=np.float32)
    tornati = invert_transform(apply_transform(valori, "log1p", 1000.0), "log1p", 1000.0)
    assert np.allclose(tornati, valori, atol=1e-9)


def test_il_log1p_tollera_negativi_di_arrotondamento() -> None:
    """Le cumulate ERA5 possono avere negativi minuscoli da rumore numerico."""
    trasformati = apply_transform(np.array([-1e-9], dtype=np.float32), "log1p", 1000.0)
    assert np.isfinite(trasformati).all()
    assert trasformati[0] == 0.0


def test_trasformazione_sconosciuta_e_rifiutata() -> None:
    with pytest.raises(FeatureError, match="Trasformazione sconosciuta"):
        apply_transform(np.zeros(1), "radice")


def test_la_precipitazione_dichiara_la_scala_in_millimetri() -> None:
    assert transform_of("tp") == ("log1p", 1000.0)
    assert transform_of("t2m") == ("identity", 1.0)
    assert transform_of(WIND_SPEED) == ("identity", 1.0)


# --------------------------------------------------------------------------- #
# Normalizzazione
# --------------------------------------------------------------------------- #


def test_la_normalizzazione_e_invertibile() -> None:
    stats = NormStats(
        mean={"t2m": 280.0}, std={"t2m": 10.0}, transform={"t2m": "identity"},
        scale={"t2m": 1.0}, computed_on_split="train",
    )
    valori = np.array([270.0, 290.0], dtype=np.float32)
    assert np.allclose(stats.denormalize("t2m", stats.normalize("t2m", valori)), valori)


def test_normalizzare_una_variabile_ignota_fallisce_esplicitamente() -> None:
    stats = NormStats({}, {}, {}, {}, "train")
    with pytest.raises(FeatureError, match="Mancano le statistiche"):
        stats.normalize("t2m", np.zeros(1))


def test_le_statistiche_usano_solo_gli_slot_indicati() -> None:
    """E' questa restrizione a impedire che il futuro entri nella normalizzazione."""
    # `msl` non ha conversione di unita': cosi' il test misura la selezione degli slot
    # e non resta legato alla scala di una variabile particolare.
    valori = np.zeros((10, 2, 2), dtype=np.float32)
    valori[5:] = 100.0
    stats = compute_norm_stats(SlotReader({"msl": valori}), ["msl"], list(range(5)))
    assert stats.mean["msl"] == 0.0


def test_le_statistiche_su_tutti_gli_slot_differiscono() -> None:
    valori = np.zeros((10, 2, 2), dtype=np.float32)
    valori[5:] = 100.0
    stats = compute_norm_stats(SlotReader({"msl": valori}), ["msl"], list(range(10)))
    assert stats.mean["msl"] == pytest.approx(50.0)


def test_una_variabile_costante_non_divide_per_zero() -> None:
    valori = np.full((4, 2, 2), 7.0, dtype=np.float32)
    stats = compute_norm_stats(SlotReader({"t2m": valori}), ["t2m"], list(range(4)))
    assert stats.std["t2m"] > 0.0
    assert np.isfinite(stats.normalize("t2m", valori)).all()


def test_senza_slot_non_si_calcolano_statistiche() -> None:
    with pytest.raises(FeatureError, match="Servono slot di train"):
        compute_norm_stats(SlotReader({"t2m": np.zeros((2, 2, 2))}), ["t2m"], [])


def test_un_campo_statico_viene_replicato_non_indicizzato() -> None:
    """Il difetto restava nascosto finche' gli indici stavano sotto le righe della griglia.

    Con pochi istanti utilizzabili nessuno slot superava la latitudine e il campo statico
    tornava sbagliato ma senza errore; con lo store completo gli indici la superano.
    """
    statico = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    lettore = SlotReader({"t2m": np.zeros((900, 6, 4), dtype=np.float32), "lsm": statico})

    letti = lettore.read_slots([0, 300, 899], ["lsm"])

    assert letti["lsm"].shape == (3, 6, 4)
    for istante in range(3):
        assert np.array_equal(letti["lsm"][istante], statico)


def test_le_statistiche_di_un_campo_statico_non_dipendono_dagli_istanti() -> None:
    statico = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    lettore = SlotReader({"t2m": np.zeros((900, 6, 4), dtype=np.float32), "lsm": statico})

    poche = compute_norm_stats(lettore, ["lsm"], [0, 1, 2])
    lontane = compute_norm_stats(lettore, ["lsm"], [700, 800, 899])

    assert lontane.mean["lsm"] == pytest.approx(poche.mean["lsm"])
    assert lontane.mean["lsm"] == pytest.approx(float(statico.mean()))


def test_il_numero_di_istanti_ignora_i_campi_statici() -> None:
    """Ordinato prima, un campo statico farebbe dichiarare 6 istanti invece di 900."""
    lettore = SlotReader(
        {
            "lsm": np.zeros((6, 4), dtype=np.float32),
            "t2m": np.zeros((900, 6, 4), dtype=np.float32),
        }
    )
    assert lettore.n_slots == 900


def test_le_statistiche_fanno_andata_e_ritorno_su_tabella() -> None:
    valori = np.random.default_rng(0).normal(size=(4, 3, 3)).astype(np.float32)
    stats = compute_norm_stats(SlotReader({"tp": valori}), ["tp"], list(range(4)))
    tornate = NormStats.from_table(stats.to_table())
    assert tornate.mean["tp"] == pytest.approx(stats.mean["tp"])
    assert tornate.scale["tp"] == pytest.approx(stats.scale["tp"])
    assert tornate.transform["tp"] == stats.transform["tp"]


# --------------------------------------------------------------------------- #
# Canali derivati
# --------------------------------------------------------------------------- #


def test_la_velocita_del_vento_e_il_modulo() -> None:
    assert wind_speed(np.array([3.0]), np.array([4.0]))[0] == pytest.approx(5.0)


def test_i_canali_di_latitudine_sono_costanti_per_riga() -> None:
    canali = latitude_channels(np.array([60.0, 30.0]), 3)
    assert canali.shape == (2, 2, 3)
    assert np.allclose(canali[0, 0], 60.0 / 90.0)
    assert canali[1, 0, 0] == pytest.approx(np.cos(np.deg2rad(60.0)), abs=1e-6)


# --------------------------------------------------------------------------- #
# Costruzione del tensore
# --------------------------------------------------------------------------- #


def test_il_tensore_ha_la_forma_del_layout(layout: InputLayout, config: Config) -> None:
    finestra = finestra_finta(layout)
    tensore = build_input_tensor(
        layout, finestra, stats_neutre(layout),
        static_fields=campi_statici(layout),
        latitudes=np.linspace(75.0, 10.0, 4),
        reference_time=datetime(2024, 1, 15, 12, tzinfo=UTC),
        slot_hours=config.time.slot_hours,
    )
    assert tensore.shape == (layout.n_channels, 4, 5)
    assert np.isfinite(tensore).all()


def test_il_canale_di_stato_prende_lo_slot_giusto(
    layout: InputLayout, config: Config
) -> None:
    """Il canale con lag 0 deve contenere l'ultimo slot, non il primo."""
    finestra = finestra_finta(layout)
    tensore = build_input_tensor(
        layout, finestra, stats_neutre(layout),
        static_fields=campi_statici(layout),
        latitudes=np.linspace(75.0, 10.0, 4),
        reference_time=datetime(2024, 1, 15, 12, tzinfo=UTC),
        slot_hours=config.time.slot_hours,
    )
    prima = layout.dynamic_variables[0]
    recente = tensore[layout.index_of(f"{prima}_t-0")]
    vecchio = tensore[layout.index_of(f"{prima}_t-{layout.input_slots - 1}")]
    assert recente[0, 0] == pytest.approx(layout.input_slots - 1)
    assert vecchio[0, 0] == pytest.approx(0.0)


def test_la_tendenza_e_la_differenza_attesa(layout: InputLayout, config: Config) -> None:
    finestra = finestra_finta(layout)
    tensore = build_input_tensor(
        layout, finestra, stats_neutre(layout),
        static_fields=campi_statici(layout),
        latitudes=np.linspace(75.0, 10.0, 4),
        reference_time=datetime(2024, 1, 15, 12, tzinfo=UTC),
        slot_hours=config.time.slot_hours,
    )
    prima = layout.dynamic_variables[0]
    lag = layout.tendency_lags[0]
    # I valori crescono di 1 per slot, quindi la differenza su `lag` slot vale `lag`.
    assert tensore[layout.index_of(f"{prima}_delta{lag}")][0, 0] == pytest.approx(lag)


def test_una_finestra_di_lunghezza_sbagliata_e_rifiutata(
    layout: InputLayout, config: Config
) -> None:
    finestra = finestra_finta(layout)
    corta = {nome: valori[:-1] for nome, valori in finestra.items()}
    with pytest.raises(FeatureError, match="slot, il layout ne richiede"):
        build_input_tensor(layout, corta, stats_neutre(layout))


def test_i_campi_statici_mancanti_sono_segnalati(
    layout: InputLayout, config: Config
) -> None:
    with pytest.raises(FeatureError, match="campo statico"):
        build_input_tensor(
            layout, finestra_finta(layout), stats_neutre(layout),
            static_fields={}, latitudes=np.linspace(75.0, 10.0, 4),
            reference_time=datetime(2024, 1, 15, 12, tzinfo=UTC),
            slot_hours=config.time.slot_hours,
        )


def test_i_canali_temporali_sono_costanti_nello_spazio(
    layout: InputLayout, config: Config
) -> None:
    tensore = build_input_tensor(
        layout, finestra_finta(layout), stats_neutre(layout),
        static_fields=campi_statici(layout),
        latitudes=np.linspace(75.0, 10.0, 4),
        reference_time=datetime(2024, 6, 21, 12, tzinfo=UTC),
        slot_hours=config.time.slot_hours,
    )
    for canale in layout.channels:
        if canale.group == GROUP_TIME:
            valori = tensore[canale.index]
            assert np.allclose(valori, valori[0, 0])


def test_i_gruppi_presenti_sono_quelli_attesi(layout: InputLayout) -> None:
    gruppi = {canale.group for canale in layout.channels}
    assert gruppi == {
        GROUP_STATE, GROUP_TENDENCY, GROUP_WIND, GROUP_STATIC, "latitude", GROUP_TIME
    }

# --------------------------------------------------------------------------- #
# Conversione di unita' in lettura
# --------------------------------------------------------------------------- #

ZERO_CELSIUS_IN_KELVIN = 273.15


class TestConversioneInCelsius:
    """La conversione avviene una volta sola, subito dopo la lettura.

    E' quel punto unico a garantire che normalizzazione, target, metriche e previsioni
    parlino tutti la stessa unita', e che `normalize` e `denormalize` restino l'una
    l'inversa dell'altra.
    """

    def test_le_temperature_dichiarano_lo_scarto(self) -> None:
        assert store_offset_of("t2m") == pytest.approx(-ZERO_CELSIUS_IN_KELVIN)
        assert store_offset_of("d2m") == pytest.approx(-ZERO_CELSIUS_IN_KELVIN)

    def test_le_altre_variabili_non_hanno_scarto(self) -> None:
        for nome in ("msl", "tp", "sf", "u10", "v10", "tcc", "sd", "lsm", "z"):
            assert store_offset_of(nome) == 0.0, nome

    def test_la_velocita_del_vento_derivata_non_ha_scarto(self) -> None:
        assert store_offset_of(WIND_SPEED) == 0.0

    def test_una_variabile_sconosciuta_non_viene_traslata(self) -> None:
        assert store_offset_of("variabile_inventata") == 0.0

    def test_il_punto_di_congelamento_diventa_zero(self) -> None:
        convertiti = to_working_units("t2m", np.array([ZERO_CELSIUS_IN_KELVIN]))
        assert float(convertiti[0]) == pytest.approx(0.0, abs=1e-3)

    def test_una_temperatura_tipica_diventa_leggibile(self) -> None:
        convertiti = to_working_units("t2m", np.array([293.15]))
        assert float(convertiti[0]) == pytest.approx(20.0, abs=1e-3)

    def test_la_pressione_resta_invariata(self) -> None:
        valori = np.array([101325.0])
        assert to_working_units("msl", valori)[0] == pytest.approx(101325.0)

    def test_la_conversione_conserva_forma_e_tipo(self) -> None:
        valori = np.full((3, 4, 5), 280.0, dtype=np.float32)
        convertiti = to_working_units("t2m", valori)
        assert convertiti.shape == valori.shape
        assert convertiti.dtype == np.float32

    def test_il_lettore_applica_la_conversione(self) -> None:
        valori = np.full((4, 2, 2), 283.15, dtype=np.float32)
        letti = SlotReader({"t2m": valori}).read_slots([0, 1], ["t2m"])
        assert np.allclose(letti["t2m"], 10.0, atol=1e-3)

    def test_il_lettore_non_converte_le_altre_variabili(self) -> None:
        valori = np.full((4, 2, 2), 101325.0, dtype=np.float32)
        letti = SlotReader({"msl": valori}).read_slots([0], ["msl"])
        assert np.allclose(letti["msl"], 101325.0)

    def test_le_statistiche_risultano_in_gradi_leggibili(self) -> None:
        # Media di 283,15 K: in Celsius deve valere 10, non 283.
        valori = np.full((6, 2, 2), 283.15, dtype=np.float32)
        stats = compute_norm_stats(SlotReader({"t2m": valori}), ["t2m"], list(range(6)))
        assert stats.mean["t2m"] == pytest.approx(10.0, abs=1e-2)

    def test_normalizzazione_e_denormalizzazione_restano_inverse(self) -> None:
        # E' la proprieta' che si romperebbe convertendo piu' a valle invece che in
        # lettura: le due funzioni lavorerebbero in unita' diverse.
        stats = NormStats(
            {"t2m": 10.0}, {"t2m": 5.0}, {"t2m": "identity"}, {"t2m": 1.0}, "train"
        )
        gradi = np.array([-15.0, 0.0, 12.5, 33.0], dtype=np.float32)
        assert np.allclose(stats.denormalize("t2m", stats.normalize("t2m", gradi)), gradi)

    def test_traslare_la_media_lascia_invariata_la_normalizzazione(self) -> None:
        # Giustifica la correzione applicata al checkpoint gia' addestrato: spostare
        # dato e media della stessa quantita' non cambia il valore normalizzato.
        kelvin = np.array([270.0, 280.0, 290.0], dtype=np.float32)
        in_kelvin = NormStats(
            {"t2m": 280.0}, {"t2m": 8.0}, {"t2m": "identity"}, {"t2m": 1.0}, "train"
        )
        in_celsius = NormStats(
            {"t2m": 280.0 - ZERO_CELSIUS_IN_KELVIN}, {"t2m": 8.0},
            {"t2m": "identity"}, {"t2m": 1.0}, "train",
        )
        assert np.allclose(
            in_kelvin.normalize("t2m", kelvin),
            in_celsius.normalize("t2m", to_working_units("t2m", kelvin)),
            atol=1e-5,
        )
