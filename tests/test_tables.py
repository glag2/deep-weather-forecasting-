"""Test del layer tabellare Polars/Parquet.

Il punto centrale e' che lo schema sia verificato a runtime, non solo dichiarato: un
Parquet scritto da una versione precedente del codice deve essere rifiutato con un
messaggio che dice quale colonna non torna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from dwf.slots import GAP_LABEL
from dwf.tables import (
    ALL_SPECS,
    DOWNLOADS,
    METRICS,
    SLOTS,
    VARIABLES,
    build_slots_table,
    build_variables_table,
    cast_to_schema,
    empty_table,
    read_table,
    validate_schema,
    write_table,
)
from dwf.variables import spec_by_short_name

TIMES = [
    datetime(2024, 1, 1, 6, tzinfo=UTC),
    datetime(2024, 1, 1, 12, tzinfo=UTC),
    datetime(2024, 1, 1, 18, tzinfo=UTC),
    datetime(2024, 1, 2, 6, tzinfo=UTC),
]


def slots_frame() -> pl.DataFrame:
    return build_slots_table(
        times=TIMES,
        slot_of_day_values=[0, 1, 2, 0],
        splits=["train", "train", GAP_LABEL, "val"],
        usable=[True, True, False, True],
    )


# --------------------------------------------------------------------------- #
# Schemi
# --------------------------------------------------------------------------- #


def test_ogni_specifica_ha_nome_file_e_schema_non_vuoti() -> None:
    for spec in ALL_SPECS:
        assert spec.filename.endswith(".parquet")
        assert spec.schema
        assert spec.description


def test_i_nomi_dei_file_sono_univoci() -> None:
    nomi = [spec.filename for spec in ALL_SPECS]
    assert len(set(nomi)) == len(nomi)


def test_tabella_vuota_rispetta_lo_schema() -> None:
    for spec in ALL_SPECS:
        frame = empty_table(spec)
        assert frame.height == 0
        validate_schema(frame, spec)


def test_colonna_mancante_e_segnalata() -> None:
    frame = empty_table(SLOTS).drop("usable")
    with pytest.raises(ValueError, match="colonne mancanti"):
        validate_schema(frame, SLOTS)


def test_colonna_inattesa_e_segnalata() -> None:
    frame = empty_table(SLOTS).with_columns(extra=pl.lit(1))
    with pytest.raises(ValueError, match="colonne inattese"):
        validate_schema(frame, SLOTS)


def test_tipo_errato_e_segnalato_con_dettaglio() -> None:
    frame = empty_table(METRICS).with_columns(value=pl.col("value").cast(pl.Float32))
    with pytest.raises(ValueError, match="tipi errati"):
        validate_schema(frame, METRICS)


def test_ordine_diverso_delle_colonne_e_rifiutato() -> None:
    """L'ordine conta: i consumatori leggono per posizione oltre che per nome."""
    colonne = list(SLOTS.schema)
    frame = empty_table(SLOTS).select([*colonne[1:], colonne[0]])
    with pytest.raises(ValueError, match="ordine delle colonne"):
        validate_schema(frame, SLOTS)


# --------------------------------------------------------------------------- #
# Scrittura e rilettura
# --------------------------------------------------------------------------- #


def test_round_trip_conserva_valori_e_tipi(tmp_path: Path) -> None:
    frame = slots_frame()
    percorso = write_table(frame, SLOTS, tmp_path / "tables")
    assert percorso.exists()
    riletta = read_table(SLOTS, tmp_path / "tables")
    assert riletta.equals(frame)
    assert dict(riletta.schema) == dict(SLOTS.schema)


def test_la_scrittura_crea_la_directory(tmp_path: Path) -> None:
    destinazione = tmp_path / "a" / "b" / "tables"
    write_table(slots_frame(), SLOTS, destinazione)
    assert destinazione.is_dir()


def test_scrittura_di_frame_non_conforme_e_rifiutata(tmp_path: Path) -> None:
    frame = slots_frame().drop("usable")
    with pytest.raises(ValueError, match="non rispetta lo schema"):
        write_table(frame, SLOTS, tmp_path / "tables")


def test_lettura_di_tabella_assente_indica_la_fase_mancante(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as errore:
        read_table(SLOTS, tmp_path / "tables")
    messaggio = str(errore.value)
    assert SLOTS.filename in messaggio
    # L'errore deve dire quale fase produce la tabella, non solo che manca.
    assert SLOTS.description in messaggio


def test_parquet_con_schema_vecchio_e_rifiutato_in_lettura(tmp_path: Path) -> None:
    """Motivo per cui la validazione e' a runtime e non solo dichiarativa."""
    tables = tmp_path / "tables"
    tables.mkdir()
    slots_frame().drop("usable").write_parquet(SLOTS.path(tables))
    with pytest.raises(ValueError, match="colonne mancanti"):
        read_table(SLOTS, tables)


# --------------------------------------------------------------------------- #
# Conversione
# --------------------------------------------------------------------------- #


def test_cast_riordina_e_converte() -> None:
    frame = pl.DataFrame(
        {
            "value": [1.0],
            "model": ["persistence"],
            "split": ["test"],
            "fold": [0],
            "variable": ["t2m"],
            "lead_slot": [1],
            "month": [1],
            "metric": ["rmse"],
            "n_values": [10],
        }
    )
    convertita = cast_to_schema(frame, METRICS)
    assert list(convertita.columns) == list(METRICS.schema)
    validate_schema(convertita, METRICS)


def test_cast_con_colonna_mancante_e_esplicito() -> None:
    with pytest.raises(ValueError, match="Colonne mancanti"):
        cast_to_schema(pl.DataFrame({"model": ["x"]}), METRICS)


# --------------------------------------------------------------------------- #
# Costruttori
# --------------------------------------------------------------------------- #


def test_catalogo_slot_deriva_i_campi_temporali() -> None:
    frame = slots_frame()
    assert frame.height == len(TIMES)
    assert frame["slot_index"].to_list() == [0, 1, 2, 3]
    assert frame["hour"].to_list() == [6, 12, 18, 6]
    assert frame["day"].to_list() == [1, 1, 1, 2]
    assert frame["day_of_year"].to_list() == [1, 1, 1, 2]
    assert frame["source_month"].unique().to_list() == ["2024-01"]


def test_catalogo_slot_accetta_l_etichetta_di_gap() -> None:
    frame = slots_frame()
    assert GAP_LABEL in frame["split"].to_list()


def test_catalogo_slot_rifiuta_etichette_sconosciute() -> None:
    with pytest.raises(ValueError, match="Etichette di split non ammesse"):
        build_slots_table(
            times=TIMES,
            slot_of_day_values=[0, 1, 2, 0],
            splits=["train", "train", "casuale", "val"],
            usable=[True] * 4,
        )


def test_catalogo_slot_rifiuta_sequenze_disallineate() -> None:
    """Le sequenze sono allineate per posizione: una lunghezza diversa e' un bug."""
    with pytest.raises(ValueError, match="lunghezza diversa"):
        build_slots_table(
            times=TIMES,
            slot_of_day_values=[0, 1],
            splits=["train"] * 4,
            usable=[True] * 4,
        )


def test_registro_variabili_marca_i_target() -> None:
    specs = [spec_by_short_name(name) for name in ("t2m", "tp", "sf", "sd")]
    frame = build_variables_table(specs, targets=["t2m", "tp", "sf"])
    validate_schema(frame, VARIABLES)
    marcati = dict(zip(frame["short_name"], frame["is_target"], strict=True))
    assert marcati == {"t2m": True, "tp": True, "sf": True, "sd": False}


def test_registro_variabili_conserva_tipo_e_unita() -> None:
    frame = build_variables_table([spec_by_short_name("t2m")], targets=[])
    assert frame["kind"][0] == "instantaneous"
    assert frame["units"][0] == "K"


def test_manifest_dei_download_accetta_mesi_nulli() -> None:
    """I campi statici non hanno anno ne mese: il manifest deve ammettere il nullo."""
    frame = cast_to_schema(
        pl.DataFrame(
            {
                "kind": ["static", "instantaneous"],
                "year": [None, 2024],
                "month": [None, 1],
                "filename": ["static.grib", "instantaneous_2024-01.grib"],
                "n_variables": [2, 7],
                "n_hours": [1, 3],
                # I campi statici non coprono giorni: anche questi restano nulli.
                "n_days": [None, 31],
                "last_day": [None, 31],
                "status": ["downloaded", "skipped"],
                "size_bytes": [1024, 2048],
                "seconds": [1.0, 0.0],
                "message": ["", "file gia' presente"],
                "recorded_at": [datetime(2026, 8, 17, tzinfo=UTC)] * 2,
            }
        ),
        DOWNLOADS,
    )
    validate_schema(frame, DOWNLOADS)
    assert frame["year"].to_list() == [None, 2024]
