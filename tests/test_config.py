"""Test della configurazione.

La configurazione e' l'unico punto in cui area, periodo, variabili e finestre sono
dichiarati: un errore qui si propaga silenziosamente a download, training e
inferenza. I test verificano che le incoerenze vengano rifiutate subito, non che lo
schema sia semplicemente parsabile.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from dwf.config import Config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def payload() -> dict[str, Any]:
    """Contenuto grezzo di `configs/default.yaml`, da modificare nei singoli test."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build(payload: dict[str, Any], project_root: Path | None = None) -> Config:
    data = copy.deepcopy(payload)
    data["project_root"] = project_root or CONFIG_PATH.parents[1]
    return Config.model_validate(data)


# --------------------------------------------------------------------------- #
# Caricamento di riferimento
# --------------------------------------------------------------------------- #


def test_configurazione_di_default_e_valida() -> None:
    config = Config.load(CONFIG_PATH)
    assert config.region.name == "euro_atlantic"


def test_griglia_di_default_corrisponde_ai_dati_esplorati() -> None:
    """261 x 401 e' la forma dei GRIB gia' scaricati (numberOfPoints 104661)."""
    config = Config.load(CONFIG_PATH)
    assert (config.region.n_lat, config.region.n_lon) == (261, 401)
    assert config.region.n_lat * config.region.n_lon == 104_661


def test_area_cds_e_nell_ordine_nord_ovest_sud_est() -> None:
    config = Config.load(CONFIG_PATH)
    assert config.region.cds_area == [75.0, -40.0, 10.0, 60.0]


def test_periodo_configurato_copre_mesi_parziali() -> None:
    config = Config.load(CONFIG_PATH)
    months = config.time.months()
    assert months[0] == (2024, 1)
    assert months[-1] == (2026, 8)
    assert len(months) == 32
    # L'ultimo mese si ferma al limite pubblicato da ERA5, non a fine mese.
    assert config.time.days_in_month(2026, 8) == list(range(1, 12))
    assert config.time.days_in_month(2024, 1) == list(range(1, 32))


def test_numero_di_slot_coerente_con_gli_istanti_generati() -> None:
    config = Config.load(CONFIG_PATH)
    times = config.time.slot_times()
    assert len(times) == config.time.n_slots
    assert times[0].date().isoformat() == config.time.start
    assert times[-1].date().isoformat() == config.time.end
    assert sorted(times) == times


def test_ore_orarie_richieste_per_le_cumulate() -> None:
    config = Config.load(CONFIG_PATH)
    assert config.time.hourly_hours == list(range(3, 23))
    assert config.time.accumulation_coverage == (20, 4)


def test_canali_di_uscita_derivano_dalle_teste() -> None:
    """gaussian(2) + hurdle(2) + fraction_of(1) = 5 canali per lead time, x 9 slot."""
    config = Config.load(CONFIG_PATH)
    assert config.target_names == ["t2m", "tp", "sf"]
    assert config.n_output_channels == 5 * 9


def test_serializzazione_yaml_e_rileggibile(tmp_path: Path) -> None:
    config = Config.load(CONFIG_PATH)
    target = tmp_path / "round_trip.yaml"
    target.write_text(config.to_yaml(), encoding="utf-8")
    reloaded = Config.load(target, project_root=config.project_root)
    assert reloaded.model_dump(exclude={"project_root"}) == config.model_dump(
        exclude={"project_root"}
    )


# --------------------------------------------------------------------------- #
# Regione
# --------------------------------------------------------------------------- #


def test_area_non_allineata_alla_griglia_e_rifiutata(payload: dict[str, Any]) -> None:
    payload["region"]["north"] = 74.9  # (74.9 - 10) / 0.25 non e' intero
    with pytest.raises(ValidationError, match="multiplo intero"):
        build(payload)


def test_griglia_non_multipla_del_nativo_e_rifiutata(payload: dict[str, Any]) -> None:
    payload["region"]["grid"] = 0.3
    with pytest.raises(ValidationError, match="multiplo della risoluzione nativa"):
        build(payload)


def test_estensione_invertita_e_rifiutata(payload: dict[str, Any]) -> None:
    payload["region"]["north"], payload["region"]["south"] = 10.0, 75.0
    with pytest.raises(ValidationError, match="maggiore di south"):
        build(payload)


def test_griglia_piu_grossa_riduce_i_punti(payload: dict[str, Any]) -> None:
    payload["region"]["grid"] = 1.0
    config = build(payload)
    assert (config.region.n_lat, config.region.n_lon) == (66, 101)


# --------------------------------------------------------------------------- #
# Tempo
# --------------------------------------------------------------------------- #


def test_periodo_invertito_e_rifiutato(payload: dict[str, Any]) -> None:
    payload["time"]["start"], payload["time"]["end"] = "2026-07-01", "2024-08-01"
    with pytest.raises(ValidationError, match="successivo a end"):
        build(payload)


def test_data_inesistente_e_rifiutata(payload: dict[str, Any]) -> None:
    """La regex accetta la forma: la validita' del giorno va controllata a parte."""
    payload["time"]["start"] = "2024-02-31"
    with pytest.raises(ValidationError, match="data non valida"):
        build(payload)


def test_formato_a_mese_non_e_piu_accettato(payload: dict[str, Any]) -> None:
    payload["time"]["start"] = "2024-08"
    with pytest.raises(ValidationError, match="should match pattern"):
        build(payload)


def test_slot_non_ordinati_sono_rifiutati(payload: dict[str, Any]) -> None:
    payload["time"]["slot_hours"] = [12, 6, 18]
    with pytest.raises(ValidationError, match="ordinato in modo crescente"):
        build(payload)


def test_finestra_di_accumulo_che_sconfina_e_rifiutata(payload: dict[str, Any]) -> None:
    payload["time"]["slot_hours"] = [0, 12, 18]
    with pytest.raises(ValidationError, match="fuori dall'intervallo"):
        build(payload)


def test_mese_non_valido_e_rifiutato(payload: dict[str, Any]) -> None:
    payload["time"]["start"] = "2024-13"
    with pytest.raises(ValidationError):
        build(payload)


# --------------------------------------------------------------------------- #
# Variabili e target
# --------------------------------------------------------------------------- #


def test_variabile_di_tipo_sbagliato_e_rifiutata(payload: dict[str, Any]) -> None:
    """`total_precipitation` e' cumulata: metterla tra le istantanee falsa l'aggregazione."""
    payload["variables"]["instantaneous"].append("total_precipitation")
    with pytest.raises(ValidationError, match="registrata come"):
        build(payload)


def test_variabile_sconosciuta_e_rifiutata(payload: dict[str, Any]) -> None:
    payload["variables"]["instantaneous"].append("temperatura_a_caso")
    with pytest.raises(ValidationError, match="non registrata"):
        build(payload)


def test_target_non_scaricato_e_rifiutato(payload: dict[str, Any]) -> None:
    payload["targets"].append({"name": "cape", "head": "gaussian"})
    with pytest.raises(ValidationError, match="non presenti tra le variabili"):
        build(payload)


def test_target_duplicato_e_rifiutato(payload: dict[str, Any]) -> None:
    payload["targets"].append({"name": "t2m", "head": "gaussian"})
    with pytest.raises(ValidationError, match="nomi duplicati"):
        build(payload)


def test_hurdle_senza_soglia_e_rifiutata(payload: dict[str, Any]) -> None:
    payload["targets"] = [{"name": "tp", "head": "hurdle"}]
    with pytest.raises(ValidationError, match="richiede 'threshold'"):
        build(payload)


def test_fraction_of_senza_riferimento_e_rifiutata(payload: dict[str, Any]) -> None:
    payload["targets"] = [{"name": "sf", "head": "fraction_of"}]
    with pytest.raises(ValidationError, match="richiede 'reference'"):
        build(payload)


def test_riferimento_non_previsto_e_rifiutato(payload: dict[str, Any]) -> None:
    payload["targets"] = [
        {"name": "t2m", "head": "gaussian"},
        {"name": "sf", "head": "fraction_of", "reference": "tp"},
    ]
    with pytest.raises(ValidationError, match="non e' tra i target previsti"):
        build(payload)


def test_riferimento_senza_soglia_e_rifiutato(payload: dict[str, Any]) -> None:
    payload["targets"] = [
        {"name": "tp", "head": "gaussian"},
        {"name": "sf", "head": "fraction_of", "reference": "tp"},
    ]
    with pytest.raises(ValidationError, match="deve avere una 'threshold'"):
        build(payload)


def test_soglia_su_testa_non_hurdle_e_rifiutata(payload: dict[str, Any]) -> None:
    payload["targets"] = [{"name": "t2m", "head": "gaussian", "threshold": 1.0}]
    with pytest.raises(ValidationError, match="ammesso solo con la testa 'hurdle'"):
        build(payload)


def test_riferimento_a_se_stessa_e_rifiutato(payload: dict[str, Any]) -> None:
    payload["targets"] = [{"name": "sf", "head": "fraction_of", "reference": "sf"}]
    with pytest.raises(ValidationError, match="non puo' essere se stessa"):
        build(payload)


# --------------------------------------------------------------------------- #
# Livelli di pressione
# --------------------------------------------------------------------------- #

# Quota, temperatura a due livelli e umidita': la coppia di temperature e' il caso
# scomodo, perche' nei GRIB condivide la short name `t`.
PRESSIONE = [
    {"variable": "geopotential", "level": 500},
    {"variable": "temperature", "level": 850},
    {"variable": "temperature", "level": 500},
    {"variable": "specific_humidity", "level": 700},
]


def test_nessun_livello_di_pressione_per_default() -> None:
    """Il layout dei canali del checkpoint esistente dipende da questa lista vuota."""
    config = Config.load(CONFIG_PATH)
    assert config.variables.pressure == []
    assert config.variables.pressure_specs == []
    assert config.variables.dynamic_short_names == [
        "t2m", "d2m", "msl", "u10", "v10", "tcc", "sd", "tp", "sf"
    ]


def test_due_livelli_della_stessa_variabile_hanno_nomi_distinti(
    payload: dict[str, Any],
) -> None:
    """`temperature` a 500 e a 850 hPa condividono la short name GRIB `t`."""
    payload["variables"]["pressure"] = PRESSIONE
    nomi = build(payload).variables.dynamic_short_names
    assert nomi[-4:] == ["z500", "t850", "t500", "q700"]
    assert len(nomi) == len(set(nomi))


def test_i_livelli_di_pressione_non_spostano_i_canali_esistenti(
    payload: dict[str, Any],
) -> None:
    """In coda: anteporli invaliderebbe ogni checkpoint addestrato prima."""
    senza = build(payload).variables.dynamic_short_names
    payload["variables"]["pressure"] = PRESSIONE
    con = build(payload).variables.dynamic_short_names
    assert con[: len(senza)] == senza


def test_le_variabili_sono_raggruppate_per_livello(payload: dict[str, Any]) -> None:
    """Il CDS restituisce il prodotto variabili x livelli: un gruppo per livello."""
    payload["variables"]["pressure"] = PRESSIONE
    gruppi = build(payload).variables.pressure_by_level()
    assert list(gruppi) == [500, 700, 850]
    assert gruppi[500] == ("geopotential", "temperature")
    assert gruppi[700] == ("specific_humidity",)


def test_livello_non_pubblicato_da_era5_e_rifiutato(payload: dict[str, Any]) -> None:
    """Un livello inesistente fa rifiutare l'intera richiesta dal CDS."""
    payload["variables"]["pressure"] = [{"variable": "temperature", "level": 512}]
    with pytest.raises(ValidationError, match="Livello di pressione non pubblicato"):
        build(payload)


def test_variabile_su_livelli_sconosciuta_e_rifiutata(payload: dict[str, Any]) -> None:
    payload["variables"]["pressure"] = [{"variable": "temperatura_a_caso", "level": 500}]
    with pytest.raises(ValidationError, match="non registrata"):
        build(payload)


def test_variabile_di_superficie_non_vale_come_livello(payload: dict[str, Any]) -> None:
    payload["variables"]["pressure"] = [{"variable": "2m_temperature", "level": 500}]
    with pytest.raises(ValidationError, match="non registrata"):
        build(payload)


def test_coppia_variabile_livello_duplicata_e_rifiutata(payload: dict[str, Any]) -> None:
    payload["variables"]["pressure"] = [
        {"variable": "temperature", "level": 850},
        {"variable": "temperature", "level": 850},
    ]
    with pytest.raises(ValidationError, match="pressure contiene duplicati"):
        build(payload)


def test_la_temperatura_in_quota_resta_in_celsius(payload: dict[str, Any]) -> None:
    """La conversione in lettura vale per ogni temperatura, non solo per quelle a 2 m."""
    payload["variables"]["pressure"] = PRESSIONE
    per_nome = {spec.short_name: spec for spec in build(payload).variables.pressure_specs}
    assert per_nome["t850"].working_units == "degC"
    assert per_nome["z500"].working_units == "m2 s-2"
    assert per_nome["q700"].non_negative


def test_i_livelli_di_pressione_sopravvivono_al_giro_yaml(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    payload["variables"]["pressure"] = PRESSIONE
    config = build(payload)
    target = tmp_path / "con_pressione.yaml"
    target.write_text(config.to_yaml(), encoding="utf-8")
    riletta = Config.load(target, project_root=config.project_root)
    assert riletta.variables.pressure == config.variables.pressure


# --------------------------------------------------------------------------- #
# Finestre, split, training
# --------------------------------------------------------------------------- #


def test_ritardo_di_tendenza_oltre_la_finestra_e_rifiutato(payload: dict[str, Any]) -> None:
    payload["features"]["tendency_lags"] = [1, 3, 21]
    with pytest.raises(ValidationError, match="non minore di"):
        build(payload)


def test_frazioni_di_split_che_saturano_sono_rifiutate(payload: dict[str, Any]) -> None:
    payload["split"]["train_fraction"] = 0.9
    payload["split"]["val_fraction"] = 0.15
    with pytest.raises(ValidationError, match=re.escape("deve essere < 1.0")):
        build(payload)


def test_crop_non_multiplo_di_otto_e_rifiutato(payload: dict[str, Any]) -> None:
    payload["training"]["crop_size"] = 100
    with pytest.raises(ValidationError, match="multiplo di 8"):
        build(payload)


def test_campo_sconosciuto_e_rifiutato(payload: dict[str, Any]) -> None:
    """`extra=forbid` impedisce che un refuso nello YAML passi inosservato."""
    payload["training"]["learnig_rate"] = 0.1
    with pytest.raises(ValidationError):
        build(payload)


# --------------------------------------------------------------------------- #
# Percorsi
# --------------------------------------------------------------------------- #


def test_percorsi_derivati_stanno_sotto_data_root(tmp_path: Path) -> None:
    config = Config.load(CONFIG_PATH, project_root=tmp_path)
    assert config.data_root == (tmp_path / config.paths.data_root).resolve()
    for path in (
        config.raw_dir,
        config.zarr_path,
        config.static_path,
        config.tables_dir,
        config.artifacts_dir,
    ):
        assert config.data_root in path.parents


def test_percorso_che_esce_da_data_root_e_rifiutato(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """Un sottopercorso con `..` non deve poter scrivere fuori dalla cartella dati."""
    payload["paths"]["raw_subdir"] = "../../fuori"
    config = build(payload, project_root=tmp_path)
    with pytest.raises(ValueError, match="fuori da data_root"):
        _ = config.raw_dir


def test_data_root_assoluto_e_rispettato(payload: dict[str, Any], tmp_path: Path) -> None:
    payload["paths"]["data_root"] = str(tmp_path / "altrove")
    config = build(payload)
    assert config.data_root == (tmp_path / "altrove").resolve()


# --------------------------------------------------------------------------- #
# Suddivisione temporale
# --------------------------------------------------------------------------- #


def test_i_fold_coprono_tutti_i_mesi_dell_anno() -> None:
    """Motivo per cui esiste la finestra mobile: un test contiguo sarebbe solo estivo."""
    config = Config.load(CONFIG_PATH)
    times = config.time.slot_times()
    coperti: set[int] = set()
    for fold in config.build_folds():
        start, stop = fold.bounds["test"]
        coperti.update(t.month for t in times[start:stop])
    assert coperti == set(range(1, 13))


def test_i_fold_valutano_i_mesi_nevosi() -> None:
    config = Config.load(CONFIG_PATH)
    times = config.time.slot_times()
    nevosi: set[int] = set()
    for fold in config.build_folds():
        start, stop = fold.bounds["test"]
        nevosi.update(t.month for t in times[start:stop] if t.month in (12, 1, 2, 3))
    assert nevosi == {12, 1, 2, 3}


def test_modalita_cronologica_produce_un_solo_fold(payload: dict[str, Any]) -> None:
    payload["split"]["mode"] = "chronological"
    config = build(payload)
    assert len(config.build_folds()) == 1


def test_modalita_di_split_sconosciuta_e_rifiutata(payload: dict[str, Any]) -> None:
    payload["split"]["mode"] = "casuale"
    with pytest.raises(ValidationError):
        build(payload)


def test_passo_maggiore_del_test_e_rifiutato(payload: dict[str, Any]) -> None:
    payload["split"]["step_days"] = 120
    payload["split"]["test_days"] = 90
    with pytest.raises(ValidationError, match="non verrebbero mai valutati"):
        build(payload)


def test_la_radice_dati_non_collide_con_la_cartella_del_repo() -> None:
    """Su Windows `data` verrebbe risolto nella `Data/` tracciata, mescolando i file.

    Il nome deve restare distinto perche' il layout sia identico su Windows, Linux e
    in Docker, e perche' i GB generati non finiscano accanto ai sorgenti tracciati.
    """
    config = Config.load(CONFIG_PATH)
    assert config.paths.data_root.lower() != "data"
    assert config.data_root.name == config.paths.data_root


def test_tutti_i_percorsi_restano_sotto_la_radice_dati() -> None:
    config = Config.load(CONFIG_PATH)
    for percorso in (config.raw_dir, config.zarr_path, config.tables_dir, config.static_path):
        assert config.data_root in percorso.parents or percorso == config.data_root


def test_le_prove_di_un_banco_non_condividono_la_cartella_del_modello() -> None:
    """Senza un ramo per esecuzione ogni prova sovrascrive il checkpoint della precedente.

    Il difetto era silenzioso: finche' le prove sono sequenziali e hanno lo stesso numero
    di canali, ciascuna rilegge quello che ha appena scritto e il risultato sembra
    corretto. Basta un secondo addestramento in parallelo, o un numero di canali diverso,
    perche' la valutazione misuri il modello sbagliato.
    """
    config = Config.load(CONFIG_PATH)
    prima = config.model_copy(
        update={"paths": config.paths.model_copy(update={"models_subdir": "bench/a"})}
    )
    seconda = config.model_copy(
        update={"paths": config.paths.model_copy(update={"models_subdir": "bench/b"})}
    )
    assert prima.fold_dir(0) != seconda.fold_dir(0)
    assert prima.fold_dir(0).parent.name == "a"
    # Senza ramo il percorso resta quello storico, cosi' i modelli gia' salvati
    # continuano a essere trovati dove sono.
    assert config.fold_dir(0) == config.models_dir / "fold_00"


def test_il_ramo_di_esecuzione_resta_sotto_la_cartella_dei_modelli() -> None:
    """Il valore arriva da riga di comando: non deve poter uscire dalla cartella."""
    config = Config.load(CONFIG_PATH)
    fuori = config.model_copy(
        update={"paths": config.paths.model_copy(update={"models_subdir": "../../fuga"})}
    )
    with pytest.raises(ValueError, match="models_subdir"):
        fuori.fold_dir(0)
def test_un_periodo_escluso_con_un_solo_estremo_e_rifiutato() -> None:
    """Un estremo solo non definisce un periodo: accettarlo significherebbe decidere in
    silenzio l'altro, e la scelta silenziosa cadrebbe su cosa entra in addestramento."""
    config = Config.load(CONFIG_PATH)
    for aggiornamento in ({"holdout_start": "2022-01-01"}, {"holdout_end": "2023-12-31"}):
        with pytest.raises(ValueError, match="insieme"):
            config.split.model_copy(update=aggiornamento).model_validate(
                {**config.split.model_dump(), **aggiornamento}
            )


def test_un_periodo_escluso_rovesciato_e_rifiutato() -> None:
    config = Config.load(CONFIG_PATH)
    rovesciato = {"holdout_start": "2023-12-31", "holdout_end": "2022-01-01"}
    with pytest.raises(ValueError, match="precede"):
        config.split.model_validate({**config.split.model_dump(), **rovesciato})


def test_un_periodo_escluso_coerente_e_accettato() -> None:
    config = Config.load(CONFIG_PATH)
    coerente = {"holdout_start": "2022-01-01", "holdout_end": "2023-12-31"}
    split = config.split.model_validate({**config.split.model_dump(), **coerente})
    assert split.holdout_start == "2022-01-01"
    assert split.holdout_end == "2023-12-31"
