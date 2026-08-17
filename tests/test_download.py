"""Test della costruzione e dell'esecuzione delle richieste CDS.

Nessun test contatta il Climate Data Store: il protocollo ``RetrieveClient`` permette
di sostituire il client con un finto che scrive un file locale. Cosi' si collaudano
ripresa, riprovi e scrittura atomica senza credenziali e senza attese in coda.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dwf.config import Config
from dwf.data.download import (
    DATASET,
    STATIC_REFERENCE,
    DownloadOutcome,
    DownloadTask,
    build_payload,
    build_tasks,
    outcomes_to_records,
    run_task,
    run_tasks,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Configurazione reale, con dati in una directory temporanea e senza attese.

    Il backoff di produzione e' 30 s: lasciandolo, i test sui riprovi dormirebbero
    novanta secondi ciascuno. Qui interessa che il riprovo avvenga, non quanto attenda.
    """
    payload = Config.load(CONFIG_PATH, project_root=tmp_path).model_dump()
    payload["download"]["retry_backoff_seconds"] = 0.0
    return Config.model_validate({**payload, "project_root": tmp_path})


class FakeClient:
    """Client finto: registra le chiamate e scrive un file non vuoto."""

    def __init__(self, *, fail_times: int = 0, write_empty: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []
        self.fail_times = fail_times
        self.write_empty = write_empty

    def retrieve(self, name: str, request: dict[str, Any], target: str | None = None) -> None:
        self.calls.append((name, request, target))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("errore simulato del servizio")
        assert target is not None
        contenuto = b"" if self.write_empty else b"GRIB-finto"
        Path(target).write_bytes(contenuto)


# --------------------------------------------------------------------------- #
# Costruzione dei task
# --------------------------------------------------------------------------- #


def test_i_task_coprono_statici_e_ogni_mese_per_famiglia(config: Config) -> None:
    tasks = build_tasks(config)
    n_mesi = len(config.time.months())
    assert len(tasks) == 1 + 2 * n_mesi
    assert tasks[0].kind == "static"
    for kind in ("instantaneous", "accumulated"):
        assert sum(1 for t in tasks if t.kind == kind) == n_mesi


def test_lo_statico_viene_richiesto_per_primo(config: Config) -> None:
    """Se fallisce non vale la pena accodare decine di mesi."""
    assert build_tasks(config)[0].kind == "static"


def test_le_famiglie_non_condividono_lo_stesso_file(config: Config) -> None:
    """Mescolare analisi e mean rate nello stesso GRIB rompe cfgrib."""
    tasks = build_tasks(config)
    percorsi = [t.target for t in tasks]
    assert len(set(percorsi)) == len(percorsi)


def test_le_istantanee_usano_solo_gli_slot_previsti(config: Config) -> None:
    task = next(t for t in build_tasks(config) if t.kind == "instantaneous")
    assert list(task.hours) == config.time.slot_hours


def test_le_cumulate_usano_le_ore_della_finestra(config: Config) -> None:
    task = next(t for t in build_tasks(config) if t.kind == "accumulated")
    assert list(task.hours) == config.time.hourly_hours


def test_i_file_stanno_sotto_la_directory_dati(config: Config) -> None:
    for task in build_tasks(config):
        assert config.data_root in task.target.parents


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #


def mese(config: Config, year: int, month: int) -> DownloadTask:
    """Task delle variabili istantanee per un mese specifico."""
    return next(
        task
        for task in build_tasks(config)
        if task.kind == "instantaneous" and task.year == year and task.month == month
    )


def test_il_payload_ritaglia_all_area_configurata(config: Config) -> None:
    task = next(t for t in build_tasks(config) if t.kind == "instantaneous")
    payload = build_payload(task, config)
    assert payload["area"] == config.region.cds_area
    assert payload["data_format"] == "grib"


def test_il_payload_del_primo_mese_parte_dal_giorno_richiesto(config: Config) -> None:
    task = mese(config, 2024, 1)
    assert build_payload(task, config)["day"][0] == "01"


def test_il_payload_dell_ultimo_mese_e_troncato(config: Config) -> None:
    """Chiedere giorni non ancora pubblicati fa rifiutare la richiesta dal CDS."""
    giorni = build_payload(mese(config, 2026, 8), config)["day"]
    assert giorni == [f"{day:02d}" for day in range(1, 12)]


def test_il_payload_statico_usa_l_istante_di_riferimento(config: Config) -> None:
    task = next(t for t in build_tasks(config) if t.kind == "static")
    payload = build_payload(task, config)
    year, month, day, _ = STATIC_REFERENCE
    assert payload["year"] == [f"{year:04d}"]
    assert payload["month"] == [f"{month:02d}"]
    assert payload["day"] == [f"{day:02d}"]


def test_nessun_parametro_grid_alla_risoluzione_nativa(config: Config) -> None:
    """Il CDS rifiuta `grid` in alcune installazioni: si invia solo se serve."""
    task = next(t for t in build_tasks(config) if t.kind == "instantaneous")
    assert "grid" not in build_payload(task, config)


def test_parametro_grid_presente_se_la_griglia_e_diversa(tmp_path: Path) -> None:
    payload = Config.load(CONFIG_PATH, project_root=tmp_path).model_dump()
    payload["region"]["grid"] = 0.5
    config = Config.model_validate({**payload, "project_root": tmp_path})
    task = next(t for t in build_tasks(config) if t.kind == "instantaneous")
    assert build_payload(task, config)["grid"] == ["0.5", "0.5"]


def test_task_mensile_senza_anno_e_rifiutato(config: Config) -> None:
    task = DownloadTask(
        kind="instantaneous",
        variables=("2m_temperature",),
        hours=(6,),
        year=None,
        month=None,
        target=config.raw_dir / "x.grib",
    )
    with pytest.raises(ValueError, match="senza anno o mese"):
        build_payload(task, config)


def test_task_fuori_dal_periodo_configurato_e_rifiutato(config: Config) -> None:
    task = DownloadTask(
        kind="instantaneous",
        variables=("2m_temperature",),
        hours=(6,),
        year=2030,
        month=5,
        target=config.raw_dir / "x.grib",
    )
    with pytest.raises(ValueError, match="Nessun giorno da richiedere"):
        build_payload(task, config)


# --------------------------------------------------------------------------- #
# Esecuzione
# --------------------------------------------------------------------------- #


def primo_task(config: Config) -> DownloadTask:
    return next(t for t in build_tasks(config) if t.kind == "instantaneous")


def test_download_riuscito_scrive_il_file(config: Config) -> None:
    task = primo_task(config)
    client = FakeClient()
    esito = run_task(task, config, client)
    assert esito.status == "downloaded"
    assert task.target.exists()
    assert esito.size_bytes > 0
    assert client.calls[0][0] == DATASET


def test_nessun_file_partial_resta_dopo_il_successo(config: Config) -> None:
    task = primo_task(config)
    run_task(task, config, FakeClient())
    assert not list(task.target.parent.glob("*.partial"))


def test_file_gia_presente_non_viene_riscaricato(config: Config) -> None:
    task = primo_task(config)
    task.target.parent.mkdir(parents=True, exist_ok=True)
    task.target.write_bytes(b"contenuto-precedente")
    client = FakeClient()
    esito = run_task(task, config, client)
    assert esito.status == "skipped"
    assert client.calls == []
    assert task.target.read_bytes() == b"contenuto-precedente"


def test_overwrite_forza_il_riscaricamento(config: Config) -> None:
    task = primo_task(config)
    task.target.parent.mkdir(parents=True, exist_ok=True)
    task.target.write_bytes(b"contenuto-precedente")
    esito = run_task(task, config, FakeClient(), overwrite=True)
    assert esito.status == "downloaded"
    assert task.target.read_bytes() == b"GRIB-finto"


def test_file_vuoto_preesistente_viene_riscaricato(config: Config) -> None:
    """Un file di 0 byte e' il residuo di un'interruzione, non un download valido."""
    task = primo_task(config)
    task.target.parent.mkdir(parents=True, exist_ok=True)
    task.target.write_bytes(b"")
    assert run_task(task, config, FakeClient()).status == "downloaded"


def test_errore_transitorio_viene_riprovato(config: Config) -> None:
    task = primo_task(config)
    client = FakeClient(fail_times=1)
    esito = run_task(task, config, client)
    assert esito.status == "downloaded"
    assert len(client.calls) == 2


def test_errore_persistente_produce_esito_failed(config: Config) -> None:
    task = primo_task(config)
    client = FakeClient(fail_times=99)
    esito = run_task(task, config, client)
    assert esito.status == "failed"
    assert len(client.calls) == config.download.max_retries
    assert "errore simulato" in esito.message
    # Nessun file parziale o troncato deve sopravvivere a un fallimento.
    assert not task.target.exists()
    assert not list(task.target.parent.glob("*.partial"))


def test_risposta_vuota_e_trattata_come_errore(config: Config) -> None:
    task = primo_task(config)
    esito = run_task(task, config, FakeClient(write_empty=True))
    assert esito.status == "failed"
    assert "vuoto" in esito.message
    assert not task.target.exists()


def test_esecuzione_in_sequenza_raccoglie_tutti_gli_esiti(config: Config) -> None:
    tasks = build_tasks(config)[:3]
    esiti = run_tasks(tasks, config, FakeClient())
    assert len(esiti) == 3
    assert all(e.status == "downloaded" for e in esiti)


def test_stop_on_error_interrompe_la_sequenza(config: Config) -> None:
    tasks = build_tasks(config)[:3]
    esiti = run_tasks(tasks, config, FakeClient(fail_times=99), stop_on_error=True)
    assert len(esiti) == 1
    assert esiti[0].status == "failed"


def test_senza_stop_on_error_la_sequenza_continua(config: Config) -> None:
    tasks = build_tasks(config)[:3]
    # Fallisce solo il primo tentativo di ciascuno dei primi due task.
    esiti = run_tasks(tasks, config, FakeClient(fail_times=1))
    assert len(esiti) == 3


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def test_i_record_del_manifest_descrivono_i_task(config: Config) -> None:
    task = primo_task(config)
    esito = DownloadOutcome(task=task, status="downloaded", size_bytes=10, seconds=1.5)
    record = outcomes_to_records([esito])[0]
    assert record["kind"] == "instantaneous"
    assert record["year"] == task.year
    assert record["filename"] == task.target.name
    assert record["n_variables"] == len(task.variables)
    assert record["n_hours"] == len(task.hours)
    assert record["status"] == "downloaded"
    assert record["recorded_at"].tzinfo is not None


def test_manifest_vuoto_su_lista_vuota() -> None:
    assert outcomes_to_records([]) == []
