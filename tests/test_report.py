"""Test del report PDF.

Il rischio di un report non e' l'eccezione, e' la figura sbagliata ma credibile: una
mappa capovolta, una scala di colore ricalcolata per pannello che nasconde l'evoluzione,
la cella di Vigo di Cadore presa altrove, un campo di forma diversa disegnato comunque.
I test guardano quindi le decisioni che precedono il disegno, e sul PDF verificano solo
che venga prodotto e con quante pagine.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from dwf.config import Config
from dwf.predict import Forecast
from dwf.report import (
    VIGO_COLUMN,
    VIGO_ROW,
    ReportError,
    check_grid,
    colour_scale,
    domain_mean_series,
    elevation_bias_kelvin,
    latent_heat_fields,
    local_series,
    map_frame,
    report_path,
    write_report,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

N_LEAD = 9
# Pagine attese: riepilogo, quattro mappe meteorologiche, calore latente, andamenti,
# focus locale.
N_PAGINE = 8


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load(CONFIG_PATH, project_root=Path.cwd())


def previsione_finta(
    latitudini: np.ndarray, longitudini: np.ndarray, *, semina: int = 0
) -> Forecast:
    """Previsione sintetica con un gradiente nord-sud: il sud e' piu' caldo.

    Il gradiente non e' decorativo: e' l'unico modo di accorgersi che una mappa e'
    capovolta, perche' un campo casuale sembra corretto in entrambi i versi.
    """
    generatore = np.random.default_rng(semina)
    n_lat, n_lon = latitudini.size, longitudini.size
    caldo_al_sud = np.linspace(-25.0, 30.0, n_lat)[::-1].reshape(1, n_lat, 1)
    temperatura = (
        np.broadcast_to(caldo_al_sud, (N_LEAD, n_lat, n_lon))
        + generatore.normal(scale=1.0, size=(N_LEAD, n_lat, n_lon))
    ).astype(np.float32)
    probabilita = generatore.uniform(size=(N_LEAD, n_lat, n_lon)).astype(np.float32)
    frazione = generatore.uniform(size=(N_LEAD, n_lat, n_lon)).astype(np.float32)

    inizio = datetime(2026, 1, 15, 18, tzinfo=UTC)
    istanti = tuple(inizio + timedelta(hours=6 * (passo + 1)) for passo in range(N_LEAD))
    return Forecast(
        init_time=inizio,
        valid_times=istanti,
        latitudes=latitudini.astype(np.float32),
        longitudes=longitudini.astype(np.float32),
        t2m_mean=temperatura,
        t2m_std=np.full((N_LEAD, n_lat, n_lon), 1.5, dtype=np.float32),
        precip_probability=probabilita,
        precip_amount=(probabilita * 4.0).astype(np.float32),
        snow_probability=(probabilita * frazione).astype(np.float32),
    )


@pytest.fixture
def previsione_piccola() -> Forecast:
    return previsione_finta(np.linspace(75.0, 10.0, 12), np.linspace(-40.0, 60.0, 16))


@pytest.fixture
def previsione_intera(config: Config) -> Forecast:
    """Previsione sugli assi reali del dominio: serve per indirizzare Vigo di Cadore."""
    return previsione_finta(config.region.latitudes, config.region.longitudes)


def numero_pagine(percorso: Path) -> int:
    """Pagine di un PDF contate sugli oggetti pagina, senza dipendenze aggiuntive."""
    contenuto = percorso.read_bytes()
    return contenuto.count(b"/Type /Page\n") + contenuto.count(b"/Type /Page ")


# --------------------------------------------------------------------------- #
# Riquadro geografico
# --------------------------------------------------------------------------- #


def test_le_latitudini_decrescenti_mettono_il_nord_in_alto(previsione_piccola: Forecast) -> None:
    """ERA5 ordina le latitudini dal nord al sud: con origine in basso la mappa si capovolge."""
    riquadro = map_frame(previsione_piccola.latitudes, previsione_piccola.longitudes)
    assert riquadro.origin == "upper"


def test_le_latitudini_crescenti_mettono_il_nord_in_basso() -> None:
    riquadro = map_frame(np.linspace(10.0, 75.0, 12), np.linspace(-40.0, 60.0, 16))
    assert riquadro.origin == "lower"


def test_l_estensione_arriva_ai_bordi_delle_celle() -> None:
    """Con i centri di cella meta' cella resterebbe fuori dalla mappa a ogni lato."""
    riquadro = map_frame(np.array([75.0, 74.75, 74.5]), np.array([-40.0, -39.75]))
    ovest, est, sud, nord = riquadro.extent
    assert nord == pytest.approx(75.125)
    assert sud == pytest.approx(74.375)
    assert ovest == pytest.approx(-40.125)
    assert est == pytest.approx(-39.625)


def test_l_estensione_copre_il_dominio_configurato(previsione_intera: Forecast) -> None:
    ovest, est, sud, nord = map_frame(
        previsione_intera.latitudes, previsione_intera.longitudes
    ).extent
    assert (nord, sud) == pytest.approx((75.125, 9.875), abs=1e-3)
    assert (ovest, est) == pytest.approx((-40.125, 60.125), abs=1e-3)


def test_un_asse_con_un_solo_punto_e_un_errore() -> None:
    with pytest.raises(ReportError, match="due punti"):
        map_frame(np.array([75.0]), np.linspace(-40.0, 60.0, 16))


# --------------------------------------------------------------------------- #
# Scala di colore
# --------------------------------------------------------------------------- #


def test_la_scala_esclude_le_code() -> None:
    """Un solo valore estremo appiattirebbe tutto il resto in una sola tinta."""
    campo = np.concatenate([np.linspace(0.0, 10.0, 999), np.array([1e6])])
    minimo, massimo = colour_scale(campo, quantile=0.02)
    assert massimo < 100.0
    assert minimo >= 0.0


def test_la_scala_simmetrica_e_centrata_sullo_zero() -> None:
    minimo, massimo = colour_scale(np.linspace(-3.0, 8.0, 100), symmetric=True)
    assert minimo == pytest.approx(-massimo)


def test_un_campo_costante_ha_comunque_un_intervallo() -> None:
    minimo, massimo = colour_scale(np.full(50, 4.0))
    assert massimo > minimo


def test_un_campo_senza_valori_finiti_e_un_errore() -> None:
    with pytest.raises(ReportError, match="finiti"):
        colour_scale(np.full(10, np.nan))


def test_un_quantile_fuori_intervallo_e_un_errore() -> None:
    with pytest.raises(ReportError, match="quantile"):
        colour_scale(np.linspace(0.0, 1.0, 10), quantile=0.7)


# --------------------------------------------------------------------------- #
# Coerenza dei campi
# --------------------------------------------------------------------------- #


def test_una_forma_incoerente_viene_segnalata(previsione_piccola: Forecast) -> None:
    """Un campo trasposto produrrebbe mappe plausibili e sbagliate."""
    guasta = Forecast(
        **{
            **{
                campo: getattr(previsione_piccola, campo)
                for campo in (
                    "init_time",
                    "valid_times",
                    "latitudes",
                    "longitudes",
                    "t2m_mean",
                    "t2m_std",
                    "precip_probability",
                    "precip_amount",
                )
            },
            "snow_probability": previsione_piccola.snow_probability.transpose(0, 2, 1),
        }
    )
    with pytest.raises(ReportError, match="snow_probability"):
        check_grid(guasta)


def test_una_previsione_coerente_dichiara_le_sue_dimensioni(
    previsione_piccola: Forecast,
) -> None:
    assert check_grid(previsione_piccola) == (N_LEAD, 12, 16)


# --------------------------------------------------------------------------- #
# Focus locale
# --------------------------------------------------------------------------- #


def test_la_cella_di_vigo_cade_sulle_coordinate_dichiarate(
    previsione_intera: Forecast,
) -> None:
    """46.50 N / 12.50 E: se gli indici scivolassero, il focus riguarderebbe un'altra valle."""
    assert float(previsione_intera.latitudes[VIGO_ROW]) == pytest.approx(46.5, abs=1e-4)
    assert float(previsione_intera.longitudes[VIGO_COLUMN]) == pytest.approx(12.5, abs=1e-4)


def test_la_serie_locale_legge_proprio_quella_cella(previsione_intera: Forecast) -> None:
    serie = local_series(previsione_intera)
    assert serie.height == N_LEAD
    assert serie.get_column("t2m_mean_celsius").to_numpy() == pytest.approx(
        previsione_intera.t2m_mean[:, VIGO_ROW, VIGO_COLUMN], abs=1e-5
    )
    assert serie.get_column("precip_amount_mm").to_numpy() == pytest.approx(
        previsione_intera.precip_amount[:, VIGO_ROW, VIGO_COLUMN], abs=1e-5
    )


def test_una_cella_fuori_dalla_griglia_e_un_errore(previsione_piccola: Forecast) -> None:
    with pytest.raises(ReportError, match="fuori dalla griglia"):
        local_series(previsione_piccola, row=VIGO_ROW, column=VIGO_COLUMN)


def test_lo_scarto_di_quota_vale_circa_tre_kelvin() -> None:
    """512 m di dislivello per 6.5 K/km: la previsione e' piu' fredda del paese."""
    assert elevation_bias_kelvin() == pytest.approx(3.328, abs=1e-3)


# --------------------------------------------------------------------------- #
# Calore latente
# --------------------------------------------------------------------------- #


def test_senza_rugiada_l_ipotesi_di_saturazione_e_dichiarata(
    previsione_piccola: Forecast,
) -> None:
    campo, ipotesi = latent_heat_fields(previsione_piccola)
    assert "satura" in ipotesi
    assert campo.shape == previsione_piccola.t2m_mean.shape
    assert np.isfinite(campo).all()
    assert (campo > 0.0).all()


def test_l_aria_secca_contiene_meno_calore_latente(previsione_piccola: Forecast) -> None:
    satura, _ = latent_heat_fields(previsione_piccola)
    secca, ipotesi = latent_heat_fields(
        previsione_piccola,
        dewpoint_celsius=previsione_piccola.t2m_mean - 10.0,
    )
    assert "rugiada" in ipotesi
    assert (secca < satura).all()


def test_un_solo_istante_di_rugiada_viene_replicato(previsione_piccola: Forecast) -> None:
    """Il campo osservato e' bidimensionale: persisterlo sulle scadenze e' l'uso previsto."""
    campo, _ = latent_heat_fields(
        previsione_piccola,
        dewpoint_celsius=previsione_piccola.t2m_mean[0] - 5.0,
        pressure_pa=np.full(previsione_piccola.t2m_mean.shape[1:], 95000.0),
    )
    assert campo.shape == previsione_piccola.t2m_mean.shape


def test_una_rugiada_di_forma_sbagliata_e_un_errore(previsione_piccola: Forecast) -> None:
    with pytest.raises(ReportError, match="dewpoint_celsius"):
        latent_heat_fields(previsione_piccola, dewpoint_celsius=np.zeros((3, 4)))


# --------------------------------------------------------------------------- #
# Medie sul dominio
# --------------------------------------------------------------------------- #


def test_le_medie_sul_dominio_hanno_una_riga_per_scadenza(
    previsione_piccola: Forecast,
) -> None:
    tabella = domain_mean_series(previsione_piccola, ["t2m_mean", "snow_probability"])
    assert tabella.height == N_LEAD
    assert tabella.get_column("t2m_mean").to_numpy() == pytest.approx(
        previsione_piccola.t2m_mean.mean(axis=(1, 2)), abs=1e-4
    )


def test_un_campo_inesistente_e_un_errore(previsione_piccola: Forecast) -> None:
    with pytest.raises(ReportError, match="sconosciuto"):
        domain_mean_series(previsione_piccola, ["umidita"])


# --------------------------------------------------------------------------- #
# Documento
# --------------------------------------------------------------------------- #


def test_il_report_viene_scritto_con_tutte_le_pagine(
    tmp_path: Path, previsione_intera: Forecast
) -> None:
    """Il report predefinito contiene le mappe **e** il testo tecnico: e' un file solo."""
    percorso = write_report(
        previsione_intera,
        tmp_path / "report.pdf",
        fold=0,
        n_parameters=1_234_567,
    )
    assert percorso.exists()
    assert percorso.stat().st_size > 20_000
    assert numero_pagine(percorso) > N_PAGINE


def test_il_testo_tecnico_puo_essere_escluso(
    tmp_path: Path, previsione_intera: Forecast
) -> None:
    """Chi vuole solo le mappe deve poterle avere, senza modificare il codice."""
    percorso = write_report(
        previsione_intera, tmp_path / "solo_mappe.pdf", documentation=False
    )
    assert numero_pagine(percorso) == N_PAGINE


def test_il_report_accetta_una_cella_di_focus_diversa(
    tmp_path: Path, previsione_piccola: Forecast
) -> None:
    """Su un dominio ridotto la cella di Vigo non esiste: il focus deve essere spostabile."""
    percorso = write_report(
        previsione_piccola,
        tmp_path / "annidato" / "report.pdf",
        focus_row=2,
        focus_column=3,
        focus_place="cella di prova",
        documentation=False,
    )
    assert percorso.stat().st_size > 10_000
    assert numero_pagine(percorso) == N_PAGINE


def test_il_report_non_nasce_se_i_campi_sono_incoerenti(
    tmp_path: Path, previsione_piccola: Forecast
) -> None:
    """La verifica precede la scrittura: nessun PDF a meta' da interpretare."""
    percorso = tmp_path / "report.pdf"
    with pytest.raises(ReportError, match="t2m_std"):
        write_report(
            Forecast(
                init_time=previsione_piccola.init_time,
                valid_times=previsione_piccola.valid_times,
                latitudes=previsione_piccola.latitudes,
                longitudes=previsione_piccola.longitudes,
                t2m_mean=previsione_piccola.t2m_mean,
                t2m_std=previsione_piccola.t2m_std[:-1],
                precip_probability=previsione_piccola.precip_probability,
                precip_amount=previsione_piccola.precip_amount,
                snow_probability=previsione_piccola.snow_probability,
            ),
            percorso,
        )
    assert not percorso.exists()


def test_il_nome_del_file_riporta_l_inizializzazione(previsione_piccola: Forecast) -> None:
    percorso = report_path(Path("qualsiasi"), previsione_piccola)
    assert percorso.name == "forecast_report_20260115_18.pdf"
