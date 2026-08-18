"""Test dell'ingestione GRIB -> Zarr.

Le funzioni che manipolano la struttura temporale sono collaudate su dataset sintetici
costruiti in memoria, che riproducono la forma reale dei file ERA5 verificata sui GRIB
scaricati: istantanee con asse `time` piatto, cumulate con `(time, step)` e
`valid_time` bidimensionale.

I test marcati come integrazione girano solo se i GRIB reali sono presenti, cosi' la
suite resta eseguibile su una macchina senza dati.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from dwf.config import Config
from dwf.data.dataset import holdout_mask
from dwf.data.ingest import (
    ACCUMULATED_DIMS,
    IngestError,
    as_naive_utc,
    available_months,
    build_catalogue,
    build_folds_table,
    check_grid,
    check_pressure_level,
    compute_stats,
    dynamic_short_names,
    flatten_accumulated,
    initialize_store,
    open_grib,
    primo_slot_addestrabile,
    slot_positions,
)
from dwf.tables import (
    FOLDS,
    SLOT_STATS,
    SLOTS,
    cast_to_schema,
    validate_schema,
    write_table,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config.load(CONFIG_PATH, project_root=tmp_path)


def accumulated_dataset(n_time: int = 2, n_step: int = 3) -> xr.Dataset:
    """Riproduce la struttura ERA5 delle cumulate: corse di previsione con step.

    Le corse partono a 18 e 06 UTC e ogni step vale un'ora, quindi gli istanti validi
    di corse consecutive si susseguono senza sovrapporsi, come nei file reali.
    """
    basi = pd.to_datetime(
        [datetime(2024, 1, 1, 18) if i % 2 == 0 else datetime(2024, 1, 2, 6) for i in range(n_time)]
    )
    passi = pd.to_timedelta(np.arange(1, n_step + 1), unit="h")
    valid = np.array([[base + passo for passo in passi] for base in basi], dtype="datetime64[ns]")

    valori = np.arange(n_time * n_step * 2 * 2, dtype=np.float32).reshape(n_time, n_step, 2, 2)
    return xr.Dataset(
        {"tp": (("time", "step", "latitude", "longitude"), valori)},
        coords={
            "time": basi,
            "step": passi,
            "latitude": [45.0, 44.75],
            "longitude": [10.0, 10.25],
            "valid_time": (("time", "step"), valid),
        },
    )


# --------------------------------------------------------------------------- #
# Conversione dei fusi
# --------------------------------------------------------------------------- #


def test_istanti_utc_diventano_datetime64_senza_fuso() -> None:
    tempi = [datetime(2024, 1, 1, 6, tzinfo=UTC), datetime(2024, 1, 1, 12, tzinfo=UTC)]
    convertiti = as_naive_utc(tempi)
    assert convertiti.dtype == np.dtype("datetime64[ns]")
    assert str(convertiti[0]) == "2024-01-01T06:00:00.000000000"


def test_la_conversione_conserva_l_ordine() -> None:
    tempi = [datetime(2024, 1, day, 6, tzinfo=UTC) for day in (1, 2, 3)]
    convertiti = as_naive_utc(tempi)
    assert list(convertiti) == sorted(convertiti)


# --------------------------------------------------------------------------- #
# Appiattimento delle cumulate
# --------------------------------------------------------------------------- #


def test_appiattimento_produce_un_solo_asse_temporale() -> None:
    piatto = flatten_accumulated(accumulated_dataset(n_time=2, n_step=3))
    assert "step" not in piatto.dims
    assert piatto.sizes["valid_time"] == 6


def test_appiattimento_impone_l_ordine_delle_dimensioni() -> None:
    """Senza transpose `stack` lascia il tempo per ultimo e ogni indice temporale
    finirebbe per selezionare la latitudine."""
    piatto = flatten_accumulated(accumulated_dataset())
    assert piatto["tp"].dims == ACCUMULATED_DIMS


def test_appiattimento_ordina_gli_istanti() -> None:
    piatto = flatten_accumulated(accumulated_dataset(n_time=2, n_step=3))
    tempi = pd.to_datetime(piatto.valid_time.values)
    assert list(tempi) == sorted(tempi)


def test_appiattimento_conserva_i_valori() -> None:
    """Il rimescolamento degli assi non deve alterare il contenuto."""
    grezzo = accumulated_dataset(n_time=2, n_step=3)
    piatto = flatten_accumulated(grezzo)
    atteso = grezzo["tp"].isel(time=0, step=0).values
    istante = pd.to_datetime(grezzo.valid_time.values[0, 0])
    trovato = piatto["tp"].sel(valid_time=istante).values
    assert np.array_equal(trovato, atteso)


def test_appiattimento_rimuove_gli_istanti_duplicati() -> None:
    grezzo = accumulated_dataset(n_time=2, n_step=2)
    # Si forza la seconda corsa a coincidere con la prima.
    duplicati = grezzo.valid_time.values.copy()
    duplicati[1] = duplicati[0]
    grezzo = grezzo.assign_coords(valid_time=(("time", "step"), duplicati))
    piatto = flatten_accumulated(grezzo)
    tempi = pd.to_datetime(piatto.valid_time.values)
    assert len(tempi) == tempi.nunique()


# --------------------------------------------------------------------------- #
# Controllo della griglia
# --------------------------------------------------------------------------- #


def griglia(n_lat: int, n_lon: int, *, decrescente: bool = True) -> xr.Dataset:
    lat = np.linspace(75.0, 10.0, n_lat)
    if not decrescente:
        lat = lat[::-1]
    return xr.Dataset(
        {"t2m": (("latitude", "longitude"), np.zeros((n_lat, n_lon), dtype=np.float32))},
        coords={"latitude": lat, "longitude": np.linspace(-40.0, 60.0, n_lon)},
    )


def test_griglia_corretta_passa(config: Config) -> None:
    check_grid(griglia(config.region.n_lat, config.region.n_lon), config)


def test_griglia_di_dimensione_sbagliata_e_rifiutata(config: Config) -> None:
    with pytest.raises(IngestError, match="Griglia inattesa"):
        check_grid(griglia(100, 200), config)


def test_latitudine_crescente_e_rifiutata(config: Config) -> None:
    """ERA5 fornisce la latitudine da nord a sud e il resto della pipeline lo assume."""
    dataset = griglia(config.region.n_lat, config.region.n_lon, decrescente=False)
    with pytest.raises(IngestError, match="Latitudine crescente"):
        check_grid(dataset, config)


# --------------------------------------------------------------------------- #
# Posizioni e mesi
# --------------------------------------------------------------------------- #


def test_le_posizioni_di_un_mese_sono_contigue(config: Config) -> None:
    posizioni, tempi = slot_positions(config, 2024, 1)
    assert len(posizioni) == 31 * config.time.slots_per_day
    assert list(posizioni) == list(range(posizioni[0], posizioni[-1] + 1))
    assert all(momento.month == 1 for momento in tempi)


def test_l_ultimo_mese_e_parziale(config: Config) -> None:
    posizioni, _ = slot_positions(config, 2026, 8)
    assert len(posizioni) == 11 * config.time.slots_per_day


def test_mese_fuori_periodo_e_rifiutato(config: Config) -> None:
    with pytest.raises(IngestError, match="non appartiene al periodo"):
        slot_positions(config, 2030, 5)


def test_mesi_disponibili_richiede_entrambe_le_famiglie(config: Config) -> None:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    (config.raw_dir / "instantaneous_2024-01.grib").write_bytes(b"x")
    assert available_months(config) == []  # manca il file delle cumulate
    (config.raw_dir / "accumulated_2024-01.grib").write_bytes(b"x")
    assert available_months(config) == [(2024, 1)]


def test_file_vuoto_non_conta_come_disponibile(config: Config) -> None:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    (config.raw_dir / "instantaneous_2024-01.grib").write_bytes(b"")
    (config.raw_dir / "accumulated_2024-01.grib").write_bytes(b"")
    assert available_months(config) == []


def test_grib_assente_produce_errore_leggibile(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="assente"):
        open_grib(tmp_path / "manca.grib")


def test_le_variabili_dinamiche_seguono_l_ordine_configurato(config: Config) -> None:
    nomi = dynamic_short_names(config)
    assert nomi[0] == "t2m"
    assert set(nomi) >= {"t2m", "tp", "sf", "sd"}
    assert len(nomi) == len(set(nomi))


# --------------------------------------------------------------------------- #
# Livelli di pressione
# --------------------------------------------------------------------------- #

PRESSIONE = [
    {"variable": "geopotential", "level": 500},
    {"variable": "temperature", "level": 850},
    {"variable": "temperature", "level": 500},
    {"variable": "specific_humidity", "level": 700},
]


@pytest.fixture
def config_pressione(tmp_path: Path) -> Config:
    payload = Config.load(CONFIG_PATH, project_root=tmp_path).model_dump()
    payload["variables"]["pressure"] = PRESSIONE
    return Config.model_validate({**payload, "project_root": tmp_path})


def livello(valore: float | list[float] | None, *, nome: str = "isobaricInhPa") -> xr.Dataset:
    """Dataset minimo con la coordinata del livello, per i soli controlli di coerenza."""
    dataset = xr.Dataset({"t": (("latitude",), np.zeros(2, dtype=np.float32))})
    if valore is None:
        return dataset
    return dataset.assign_coords({nome: valore})


def test_lo_store_riceve_una_variabile_per_livello(config_pressione: Config) -> None:
    nomi = dynamic_short_names(config_pressione)
    assert nomi[-4:] == ["z500", "t850", "t500", "q700"]
    assert len(nomi) == len(set(nomi))


def test_il_livello_atteso_passa_il_controllo() -> None:
    check_pressure_level(livello(500.0), 500, "pressure500_2024-01.grib")
    check_pressure_level(livello(850.0, nome="level"), 850, "pressure850_2024-01.grib")


def test_livello_diverso_da_quello_atteso_e_rifiutato() -> None:
    """Un file scambiato darebbe valori di un altro livello sotto l'etichetta giusta."""
    with pytest.raises(IngestError, match="atteso il solo livello 500"):
        check_pressure_level(livello(850.0), 500, "pressure500_2024-01.grib")


def test_file_con_piu_livelli_e_rifiutato() -> None:
    with pytest.raises(IngestError, match="atteso il solo livello"):
        check_pressure_level(livello([500.0, 850.0]), 500, "pressure500_2024-01.grib")


def test_file_senza_coordinata_di_livello_e_rifiutato() -> None:
    with pytest.raises(IngestError, match="manca la coordinata del livello"):
        check_pressure_level(livello(None), 500, "pressure500_2024-01.grib")


def test_mesi_disponibili_richiedono_anche_i_file_in_quota(
    config_pressione: Config,
) -> None:
    raw = config_pressione.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    for nome in ("instantaneous_2024-01.grib", "accumulated_2024-01.grib"):
        (raw / nome).write_bytes(b"x")
    assert available_months(config_pressione) == []
    for nome in (
        "pressure500_2024-01.grib",
        "pressure700_2024-01.grib",
        "pressure850_2024-01.grib",
    ):
        (raw / nome).write_bytes(b"x")
    assert available_months(config_pressione) == [(2024, 1)]


# --------------------------------------------------------------------------- #
# Statistiche e catalogo
# --------------------------------------------------------------------------- #


def test_le_statistiche_contano_i_valori_non_finiti() -> None:
    campo = np.zeros((2, 3, 3), dtype=np.float32)
    campo[0, 0, 0] = np.nan
    stats = compute_stats({"t2m": campo}, np.array([5, 6]))
    primo = stats.filter(pl.col("slot_index") == 5)
    assert primo.get_column("n_nan").item() == 1
    assert primo.get_column("n_valid").item() == 8
    secondo = stats.filter(pl.col("slot_index") == 6)
    assert secondo.get_column("n_nan").item() == 0


def test_il_catalogo_rispetta_lo_schema(config: Config) -> None:
    catalogo = build_catalogue(config, ingested={(2024, 1)})
    validate_schema(catalogo, SLOTS)
    assert catalogo.height == config.time.n_slots


def test_solo_i_mesi_ingeriti_sono_utilizzabili(config: Config) -> None:
    catalogo = build_catalogue(config, ingested={(2024, 1)})
    utilizzabili = catalogo.filter((pl.col("month") == 1) & (pl.col("year") == 2024))
    assert bool(utilizzabili.get_column("usable").all())
    altri = catalogo.filter(pl.col("month") == 5)
    assert not bool(altri.get_column("usable").any())


def test_gli_slot_con_nan_diventano_inutilizzabili(config: Config) -> None:
    """Un NaN in input si propagherebbe silenziosamente a tutta la finestra."""
    stats = cast_to_schema(
        pl.DataFrame(
            {
                "slot_index": [0, 1],
                "variable": ["t2m", "t2m"],
                "n_nan": [3, 0],
                "n_valid": [10, 13],
                "minimum": [0.0, 0.0],
                "maximum": [1.0, 1.0],
                "mean": [0.5, 0.5],
            }
        ),
        SLOT_STATS,
    )
    catalogo = build_catalogue(config, ingested={(2024, 1)}, stats=stats)
    assert catalogo.get_column("usable")[0] is False
    assert catalogo.get_column("usable")[1] is True


# --------------------------------------------------------------------------- #
# Tabella dei fold
# --------------------------------------------------------------------------- #


def test_la_tabella_dei_fold_rispetta_lo_schema(config: Config) -> None:
    folds = build_folds_table(config)
    validate_schema(folds, FOLDS)
    assert folds.get_column("fold").n_unique() == len(config.build_folds())


def test_uno_slot_puo_cambiare_split_fra_fold(config: Config) -> None:
    """Motivo per cui l'assegnazione autorevole non puo' stare in una sola colonna."""
    folds = build_folds_table(config)
    per_slot = folds.group_by("slot_index").agg(pl.col("split").n_unique().alias("n_split"))
    assert per_slot.get_column("n_split").max() > 1


def test_senza_slot_utilizzabili_nessun_campione_e_ammesso(config: Config) -> None:
    nessuno = np.zeros(config.time.n_slots, dtype=bool)
    folds = build_folds_table(config, nessuno)
    assert not bool(folds.get_column("is_sample_start").any())


def test_con_tutti_gli_slot_utilizzabili_i_campioni_coincidono_col_layout(
    config: Config,
) -> None:
    tutti = np.ones(config.time.n_slots, dtype=bool)
    folds = build_folds_table(config, tutti)
    attesi = sum(
        len(fold.sample_starts[split])
        for fold in config.build_folds()
        for split in ("train", "val", "test")
    )
    assert int(folds.get_column("is_sample_start").sum()) == attesi


# --------------------------------------------------------------------------- #
# Integrazione sui GRIB reali (saltata se assenti)
# --------------------------------------------------------------------------- #

REAL_CONFIG = Config.load(CONFIG_PATH)
REAL_MONTHS = available_months(REAL_CONFIG)
richiede_dati = pytest.mark.skipif(not REAL_MONTHS, reason="nessun GRIB scaricato")


@richiede_dati
def test_i_grib_reali_hanno_la_struttura_attesa() -> None:
    year, month = REAL_MONTHS[0]
    percorso = REAL_CONFIG.raw_dir / f"accumulated_{year:04d}-{month:02d}.grib"
    with open_grib(percorso) as grezzo:
        assert "step" in grezzo.dims
        piatto = flatten_accumulated(grezzo)
    tempi = pd.to_datetime(piatto.valid_time.values)
    assert len(tempi) == tempi.nunique()
    # Le corse coprono tutte le ore del giorno: la finestra di accumulo trova sempre
    # le ore che le servono.
    assert set(REAL_CONFIG.time.hourly_hours) <= set(tempi.hour)


@richiede_dati
def test_le_istantanee_reali_coprono_tutti_gli_slot() -> None:
    year, month = REAL_MONTHS[0]
    _, tempi = slot_positions(REAL_CONFIG, year, month)
    percorso = REAL_CONFIG.raw_dir / f"instantaneous_{year:04d}-{month:02d}.grib"
    with open_grib(percorso) as dataset:
        disponibili = set(pd.to_datetime(dataset.time.values))
    assert set(pd.to_datetime(as_naive_utc(tempi))) <= disponibili


# --------------------------------------------------------------------------- #
# Creazione dello store
# --------------------------------------------------------------------------- #


def test_lo_store_nasce_pigro_e_non_materializza_le_variabili(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un array pieno per variabile pesa 1,12 GiB su questo dominio.

    Con nove variabili di superficie l'inizializzazione ci stava per poco; aggiungendo
    quattro campi in quota chiedeva circa 15 GiB e si fermava con un errore di
    allocazione. Qui si vieta a `np.full` di produrre un array grande quanto lo store: se
    qualcuno tornasse a materializzarlo, il test lo dice invece di lasciare che il difetto
    ricompaia solo su una macchina con poca memoria.
    """
    vero_full = np.full
    n_slot = len(config.time.slot_times())

    def full_controllato(shape, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Un'allocazione per blocco e' corretta e necessaria: dask ne materializza uno
        # per volta mentre scrive. Vietata e' solo quella che copre l'intero asse
        # temporale, cioe' una variabile intera in memoria.
        if isinstance(shape, tuple) and len(shape) == 3 and shape[0] == n_slot:
            raise AssertionError(f"variabile materializzata in RAM: forma {shape}")
        return vero_full(shape, *args, **kwargs)

    monkeypatch.setattr(np, "full", full_controllato)
    percorso = initialize_store(config)

    with xr.open_zarr(percorso, consolidated=True) as store:
        atteso = (
            len(config.time.slot_times()),
            len(config.region.latitudes),
            len(config.region.longitudes),
        )
        for nome in dynamic_short_names(config):
            assert store[nome].shape == atteso
            assert store[nome].chunks[0][0] == config.paths.chunk_slots


def test_uno_store_esistente_non_viene_sovrascritto_per_sbaglio(config: Config) -> None:
    percorso = initialize_store(config)
    (percorso / "segno.txt").write_text("presente", encoding="utf-8")

    assert initialize_store(config) == percorso
    assert (percorso / "segno.txt").exists()


class TestFoldConPeriodoEscluso:
    """Il periodo escluso in testa deve spostare i fold, non svuotarli.

    Il difetto che questo previene ha una firma precisa: l'addestramento parte, non trova
    finestre e si ferma dicendo "nessuna finestra di train ammessa", senza nominare il
    periodo escluso come causa. Chi legge cerca dati mancanti per ore.
    """

    def _con_periodo(self, config: Config, inizio: str, fine: str) -> Config:
        configurazione = config.model_copy(
            update={
                "split": config.split.model_copy(
                    update={"holdout_start": inizio, "holdout_end": fine}
                )
            }
        )
        # `holdout_mask` legge il catalogo dal disco: senza scriverlo non c'e' nulla da
        # confrontare con le date.
        configurazione.tables_dir.mkdir(parents=True, exist_ok=True)
        # `ingested` vuoto basta: qui conta solo la colonna degli istanti, non quali mesi
        # siano davvero sul disco.
        catalogo = build_catalogue(configurazione, ingested=set())
        write_table(catalogo, SLOTS, configurazione.tables_dir)
        return configurazione

    def test_senza_periodo_i_fold_partono_dallo_slot_zero(self, config: Config) -> None:
        assert primo_slot_addestrabile(config) == 0
        assert config.build_folds()[0].bounds["train"][0] == 0

    def test_un_periodo_in_testa_sposta_il_primo_fold_oltre_di_esso(
        self, config: Config
    ) -> None:
        primo_mese = config.time.start_date
        fine_esclusione = date(primo_mese.year, primo_mese.month, 15)
        configurazione = self._con_periodo(
            config, primo_mese.isoformat(), fine_esclusione.isoformat()
        )

        scostamento = primo_slot_addestrabile(configurazione)
        assert scostamento > 0

        fold = configurazione.build_folds(first_slot=scostamento)[0]
        assert fold.bounds["train"][0] == scostamento
        # La garanzia vera: nessun campione di addestramento comincia dentro il periodo.
        escluso = holdout_mask(configurazione)
        assert escluso is not None
        assert not any(escluso[inizio] for inizio in fold.sample_starts["train"])

    def test_un_periodo_alla_fine_non_sposta_nulla(self, config: Config) -> None:
        """Le finestre che lo toccano vengono comunque rifiutate da `sample_starts`, quindi
        spostare l'inizio sarebbe una complicazione senza effetto."""
        ultimo = config.time.end_date
        configurazione = self._con_periodo(config, ultimo.isoformat(), ultimo.isoformat())

        assert primo_slot_addestrabile(configurazione) == 0

