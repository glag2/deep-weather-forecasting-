"""Test dell'aggiornamento incrementale dei dati.

Il caso difficile e' il mese in corso. ERA5 lo pubblica a pezzi, quindi un file che
copre meta' mese e' normale, non rotto. Le proprieta' da garantire sono tre e nessuna
fallirebbe rumorosamente se violata: i mesi non ancora accaduti non vanno chiesti, un
mese parziale va richiesto di nuovo **solo** quando si allunga, e un manifest scritto
prima che i giorni venissero contati non deve provocare un riscaricamento generale.
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from dwf.config import Config
from dwf.data.download import build_tasks, days_to_request
from dwf.data.freshness import (
    FALLBACK_LATENCY_DAYS,
    Availability,
    month_statuses,
    pending_months,
    query_availability,
    read_manifest,
    summarize_freshness,
)
from dwf.tables import DOWNLOADS, cast_to_schema, write_table

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    configurazione = Config.load(CONFIG_PATH, project_root=tmp_path)
    configurazione.raw_dir.mkdir(parents=True, exist_ok=True)
    configurazione.tables_dir.mkdir(parents=True, exist_ok=True)
    return configurazione


def scrivi_grib(config: Config, anno: int, mese: int, *generi: str) -> None:
    """Crea file segnaposto: la freschezza guarda presenza e dimensione, non contenuto."""
    for genere in generi or ("instantaneous", "accumulated"):
        percorso = config.raw_dir / f"{genere}_{anno:04d}-{mese:02d}.grib"
        percorso.write_bytes(b"GRIB")


def scrivi_manifest(config: Config, righe: list[dict]) -> None:
    completi = []
    for riga in righe:
        record = {
            "kind": "instantaneous",
            "year": None,
            "month": None,
            "filename": "x.grib",
            "n_variables": 1,
            "n_hours": 3,
            "n_days": None,
            "last_day": None,
            "status": "downloaded",
            "size_bytes": 1,
            "seconds": 1.0,
            "message": "",
            "recorded_at": datetime.now(UTC),
        }
        record.update(riga)
        completi.append(record)
    write_table(cast_to_schema(pl.DataFrame(completi), DOWNLOADS), DOWNLOADS, config.tables_dir)


def disponibilita(giorno: date) -> Availability:
    return Availability(giorno, "catalogue", datetime.now(UTC))


# --------------------------------------------------------------------------- #
# Frontiera di pubblicazione
# --------------------------------------------------------------------------- #


def apri_finto(payload: dict):
    def apri(_url: str, timeout: float = 0.0):
        return io.BytesIO(json.dumps(payload).encode())

    return apri


def test_la_frontiera_si_legge_dal_catalogo() -> None:
    intervallo = ["1940-01-01T00:00:00+00:00", "2026-08-11T00:00:00+00:00"]
    payload = {"extent": {"temporal": {"interval": [intervallo]}}}
    esito = query_availability(opener=apri_finto(payload))
    assert esito.end_date == date(2026, 8, 11)
    assert esito.is_authoritative


def test_senza_rete_si_stima_con_prudenza() -> None:
    def esplodi(_url: str, timeout: float = 0.0):
        raise urllib.error.URLError("nessuna rete")

    esito = query_availability(opener=esplodi, today=date(2026, 8, 20))
    assert not esito.is_authoritative
    # La stima deve stare indietro: chiedere un giorno inesistente fa fallire tutto.
    assert esito.end_date == date(2026, 8, 20 - FALLBACK_LATENCY_DAYS)


def test_un_catalogo_malformato_non_interrompe_il_lavoro() -> None:
    esito = query_availability(opener=apri_finto({"extent": {}}), today=date(2026, 8, 20))
    assert not esito.is_authoritative


def test_un_intervallo_aperto_ricade_nella_stima() -> None:
    payload = {"extent": {"temporal": {"interval": [["1940-01-01T00:00:00+00:00", None]]}}}
    esito = query_availability(opener=apri_finto(payload), today=date(2026, 8, 20))
    assert not esito.is_authoritative


# --------------------------------------------------------------------------- #
# Giorni da chiedere
# --------------------------------------------------------------------------- #


def test_senza_frontiera_si_chiede_tutto_il_periodo(config: Config) -> None:
    assert days_to_request(config, 2025, 3) == tuple(range(1, 32))


def test_la_frontiera_taglia_la_coda_del_mese(config: Config) -> None:
    assert days_to_request(config, 2025, 3, date(2025, 3, 10)) == tuple(range(1, 11))


def test_un_mese_interamente_futuro_non_ha_giorni(config: Config) -> None:
    assert days_to_request(config, 2025, 5, date(2025, 3, 10)) == ()


def test_i_mesi_futuri_non_diventano_richieste(config: Config) -> None:
    task = build_tasks(config, until=date(2024, 2, 15))
    mesi = {(attivita.year, attivita.month) for attivita in task if attivita.year}
    assert mesi == {(2024, 1), (2024, 2)}


def test_il_task_porta_con_se_i_giorni(config: Config) -> None:
    """I giorni stanno nel task: cosi' il manifest sa che cosa copre davvero il file."""
    task = build_tasks(config, until=date(2024, 2, 15))
    febbraio = next(a for a in task if (a.year, a.month) == (2024, 2))
    assert febbraio.days == tuple(range(1, 16))


# --------------------------------------------------------------------------- #
# Stato dei mesi
# --------------------------------------------------------------------------- #


def test_un_mese_senza_file_va_scaricato(config: Config) -> None:
    tutti = month_statuses(config, disponibilita(date(2024, 3, 31)))
    stati = {stato.label: stato for stato in tutti}
    assert stati["2024-01"].needs_download
    assert "mancano i file" in stati["2024-01"].reason


def test_i_mesi_oltre_la_frontiera_non_sono_attesi(config: Config) -> None:
    etichette = [stato.label for stato in month_statuses(config, disponibilita(date(2024, 3, 15)))]
    assert etichette == ["2024-01", "2024-02", "2024-03"]


def test_un_mese_completo_non_si_riscarica(config: Config) -> None:
    scrivi_grib(config, 2024, 1)
    scrivi_manifest(
        config,
        [
            {"kind": genere, "year": 2024, "month": 1, "n_days": 31, "last_day": 31}
            for genere in ("instantaneous", "accumulated")
        ],
    )
    stato = month_statuses(config, disponibilita(date(2024, 2, 15)))[0]
    assert not stato.needs_download
    assert stato.reason == "completo"


def test_un_mese_parziale_gia_allineato_non_si_riscarica(config: Config) -> None:
    """E' il caso normale del mese in corso: parziale, ma gia' aggiornato."""
    scrivi_grib(config, 2024, 2)
    scrivi_manifest(
        config,
        [
            {"kind": genere, "year": 2024, "month": 2, "n_days": 15, "last_day": 15}
            for genere in ("instantaneous", "accumulated")
        ],
    )
    stato = next(
        s for s in month_statuses(config, disponibilita(date(2024, 2, 15))) if s.label == "2024-02"
    )
    assert stato.partial
    assert not stato.needs_download


def test_un_mese_parziale_che_si_e_allungato_si_riscarica(config: Config) -> None:
    """Il giorno dopo ERA5 ha pubblicato altri giorni: il file va rifatto."""
    scrivi_grib(config, 2024, 2)
    scrivi_manifest(
        config,
        [
            {"kind": genere, "year": 2024, "month": 2, "n_days": 15, "last_day": 15}
            for genere in ("instantaneous", "accumulated")
        ],
    )
    stato = next(
        s for s in month_statuses(config, disponibilita(date(2024, 2, 22))) if s.label == "2024-02"
    )
    assert stato.needs_download
    assert "parziale" in stato.reason
    assert stato.days_available == 22


def test_basta_una_famiglia_indietro_per_riscaricare(config: Config) -> None:
    """Se le due famiglie coprono giorni diversi il mese non e' allineato."""
    scrivi_grib(config, 2024, 2)
    scrivi_manifest(
        config,
        [
            {"kind": "instantaneous", "year": 2024, "month": 2, "n_days": 20, "last_day": 20},
            {"kind": "accumulated", "year": 2024, "month": 2, "n_days": 15, "last_day": 15},
        ],
    )
    stato = next(
        s for s in month_statuses(config, disponibilita(date(2024, 2, 20))) if s.label == "2024-02"
    )
    assert stato.days_downloaded == 15
    assert stato.needs_download


def test_conta_l_ultima_registrazione_del_mese(config: Config) -> None:
    """Un mese riscaricato lascia piu' righe: vale la piu' recente, non la prima."""
    scrivi_grib(config, 2024, 2)
    vecchia = datetime(2024, 2, 16, tzinfo=UTC)
    nuova = datetime(2024, 2, 23, tzinfo=UTC)
    def righe(giorni: int, quando: datetime) -> list[dict]:
        return [
            {
                "kind": genere,
                "year": 2024,
                "month": 2,
                "n_days": giorni,
                "last_day": giorni,
                "recorded_at": quando,
            }
            for genere in ("instantaneous", "accumulated")
        ]

    scrivi_manifest(config, righe(15, vecchia) + righe(22, nuova))
    stato = next(
        s for s in month_statuses(config, disponibilita(date(2024, 2, 22))) if s.label == "2024-02"
    )
    assert stato.days_downloaded == 22
    assert not stato.needs_download


def test_un_file_vuoto_conta_come_mancante(config: Config) -> None:
    (config.raw_dir / "instantaneous_2024-01.grib").write_bytes(b"")
    (config.raw_dir / "accumulated_2024-01.grib").write_bytes(b"GRIB")
    stato = month_statuses(config, disponibilita(date(2024, 2, 15)))[0]
    assert "instantaneous" in stato.missing_kinds


# --------------------------------------------------------------------------- #
# Manifest di versioni precedenti
# --------------------------------------------------------------------------- #


def test_un_manifest_senza_i_giorni_resta_leggibile(config: Config) -> None:
    """Le colonne sui giorni sono state aggiunte dopo: lo storico non va perso."""
    vecchio = pl.DataFrame(
        {
            "kind": ["instantaneous"],
            "year": [2024],
            "month": [1],
            "filename": ["x.grib"],
            "n_variables": [1],
            "n_hours": [3],
            "status": ["downloaded"],
            "size_bytes": [10],
            "seconds": [1.0],
            "message": [""],
            "recorded_at": [datetime.now(UTC)],
        }
    )
    vecchio.write_parquet(config.tables_dir / DOWNLOADS.filename)
    manifest = read_manifest(config)
    assert manifest is not None
    assert manifest.get_column("n_days").to_list() == [None]


def test_una_riga_senza_giorni_non_provoca_riscaricamenti(config: Config) -> None:
    scrivi_grib(config, 2024, 1)
    scrivi_manifest(
        config,
        [{"kind": g, "year": 2024, "month": 1} for g in ("instantaneous", "accumulated")],
    )
    stato = month_statuses(config, disponibilita(date(2024, 2, 15)))[0]
    assert not stato.needs_download


def test_senza_manifest_i_file_presenti_bastano(config: Config) -> None:
    scrivi_grib(config, 2024, 1)
    stato = month_statuses(config, disponibilita(date(2024, 2, 15)))[0]
    assert not stato.needs_download


# --------------------------------------------------------------------------- #
# Riepilogo
# --------------------------------------------------------------------------- #


def test_il_riepilogo_ha_una_riga_per_mese_atteso(config: Config) -> None:
    tabella = summarize_freshness(config, disponibilita(date(2024, 4, 15)))
    assert tabella.get_column("month").to_list() == ["2024-01", "2024-02", "2024-03", "2024-04"]


def test_i_mesi_da_aggiornare_sono_quelli_segnalati(config: Config) -> None:
    scrivi_grib(config, 2024, 1)
    attesi = {stato.label for stato in pending_months(config, disponibilita(date(2024, 3, 15)))}
    assert attesi == {"2024-02", "2024-03"}
