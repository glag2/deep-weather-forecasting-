"""Test della risoluzione delle credenziali CDS.

I test isolano l'ambiente del processo: senza `monkeypatch` leggerebbero le
credenziali reali dello sviluppatore e il loro esito dipenderebbe da fattori
estranei al codice. Nessun test scrive o legge un segreto vero.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dwf.credentials import (
    KEY_VARIABLE,
    RC_VARIABLE,
    URL_VARIABLE,
    MissingCredentialsError,
    credential_status,
    load_env_file,
    parse_env_file,
    require_credentials,
)

FAKE_URL = "https://example.invalid/api"
FAKE_KEY = "chiave-finta-per-test"


@pytest.fixture(autouse=True)
def ambiente_isolato(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Rimuove le credenziali reali e fa puntare il file rc a un percorso inesistente."""
    monkeypatch.delenv(URL_VARIABLE, raising=False)
    monkeypatch.delenv(KEY_VARIABLE, raising=False)
    monkeypatch.setenv(RC_VARIABLE, str(tmp_path / "rc-inesistente"))


# --------------------------------------------------------------------------- #
# Parsing del file .env
# --------------------------------------------------------------------------- #


def test_file_assente_non_solleva(tmp_path: Path) -> None:
    assert parse_env_file(tmp_path / "manca.env") == {}


def test_coppie_chiave_valore_sono_lette(tmp_path: Path) -> None:
    percorso = tmp_path / ".env"
    percorso.write_text(f"{URL_VARIABLE}={FAKE_URL}\n{KEY_VARIABLE}={FAKE_KEY}\n", encoding="utf-8")
    assert parse_env_file(percorso) == {URL_VARIABLE: FAKE_URL, KEY_VARIABLE: FAKE_KEY}


def test_commenti_e_righe_vuote_sono_ignorati(tmp_path: Path) -> None:
    percorso = tmp_path / ".env"
    percorso.write_text(
        f"# commento\n\n{URL_VARIABLE}={FAKE_URL}\nsenza_uguale\n", encoding="utf-8"
    )
    assert parse_env_file(percorso) == {URL_VARIABLE: FAKE_URL}


def test_virgolette_intorno_al_valore_sono_rimosse(tmp_path: Path) -> None:
    percorso = tmp_path / ".env"
    percorso.write_text(f'{KEY_VARIABLE}="{FAKE_KEY}"\n', encoding="utf-8")
    assert parse_env_file(percorso)[KEY_VARIABLE] == FAKE_KEY


def test_bom_iniziale_non_corrompe_la_prima_variabile(tmp_path: Path) -> None:
    """Su Windows Notepad e PowerShell 5.1 premettono un BOM al file.

    Senza lettura in utf-8-sig il BOM entra nel nome della prima variabile, che
    diventa invisibile: e' il difetto che ha reso `CDSAPI_URL` non risolvibile.
    """
    percorso = tmp_path / ".env"
    percorso.write_text(
        f"{URL_VARIABLE}={FAKE_URL}\n{KEY_VARIABLE}={FAKE_KEY}\n", encoding="utf-8-sig"
    )
    assert percorso.read_bytes().startswith(b"\xef\xbb\xbf")  # il BOM c'e' davvero
    assert parse_env_file(percorso) == {URL_VARIABLE: FAKE_URL, KEY_VARIABLE: FAKE_KEY}


def test_spazi_intorno_a_nome_e_valore_sono_ignorati(tmp_path: Path) -> None:
    percorso = tmp_path / ".env"
    percorso.write_text(f"  {URL_VARIABLE}  =  {FAKE_URL}  \n", encoding="utf-8")
    assert parse_env_file(percorso) == {URL_VARIABLE: FAKE_URL}


# --------------------------------------------------------------------------- #
# Applicazione all'ambiente
# --------------------------------------------------------------------------- #


def test_caricamento_riporta_solo_i_nomi(tmp_path: Path) -> None:
    percorso = tmp_path / ".env"
    percorso.write_text(f"{URL_VARIABLE}={FAKE_URL}\n{KEY_VARIABLE}={FAKE_KEY}\n", encoding="utf-8")
    applicati = load_env_file(percorso)
    assert sorted(applicati) == sorted([URL_VARIABLE, KEY_VARIABLE])
    # Il valore non deve comparire in cio' che viene restituito.
    assert all(FAKE_KEY not in voce for voce in applicati)


def test_ambiente_esistente_vince_sul_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In Docker un segreto iniettato dall'esterno non va sovrascritto dal file."""
    monkeypatch.setenv(KEY_VARIABLE, "valore-preesistente")
    percorso = tmp_path / ".env"
    percorso.write_text(f"{KEY_VARIABLE}=valore-del-file\n", encoding="utf-8")
    assert load_env_file(percorso) == []
    assert os.environ[KEY_VARIABLE] == "valore-preesistente"


def test_override_esplicito_sovrascrive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_VARIABLE, "valore-preesistente")
    percorso = tmp_path / ".env"
    percorso.write_text(f"{KEY_VARIABLE}=valore-del-file\n", encoding="utf-8")
    assert load_env_file(percorso, override=True) == [KEY_VARIABLE]
    assert os.environ[KEY_VARIABLE] == "valore-del-file"


def test_valore_vuoto_non_viene_applicato(tmp_path: Path) -> None:
    percorso = tmp_path / ".env"
    percorso.write_text(f"{KEY_VARIABLE}=\n", encoding="utf-8")
    assert load_env_file(percorso) == []


# --------------------------------------------------------------------------- #
# Stato e fail-closed
# --------------------------------------------------------------------------- #


def test_stato_completo_da_file(tmp_path: Path) -> None:
    percorso = tmp_path / ".env"
    percorso.write_text(f"{URL_VARIABLE}={FAKE_URL}\n{KEY_VARIABLE}={FAKE_KEY}\n", encoding="utf-8")
    stato = credential_status(percorso)
    assert stato.is_complete
    assert stato.has_url and stato.has_key


def test_stato_incompleto_se_manca_la_chiave(tmp_path: Path) -> None:
    percorso = tmp_path / ".env"
    percorso.write_text(f"{URL_VARIABLE}={FAKE_URL}\n", encoding="utf-8")
    stato = credential_status(percorso)
    assert not stato.is_complete
    assert stato.has_url and not stato.has_key


def test_stato_senza_alcuna_sorgente() -> None:
    stato = credential_status()
    assert not stato.is_complete
    assert stato.source == "nessuna sorgente trovata"


def test_file_rc_viene_letto_quando_l_ambiente_e_vuoto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = tmp_path / "cdsapirc"
    rc.write_text(f"url: {FAKE_URL}\nkey: {FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setenv(RC_VARIABLE, str(rc))
    stato = credential_status()
    assert stato.is_complete
    assert str(rc) in stato.source


def test_file_rc_con_bom_resta_leggibile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = tmp_path / "cdsapirc"
    rc.write_text(f"url: {FAKE_URL}\nkey: {FAKE_KEY}\n", encoding="utf-8-sig")
    monkeypatch.setenv(RC_VARIABLE, str(rc))
    assert credential_status().is_complete


def test_credenziali_mancanti_sollevano_errore_con_procedura() -> None:
    with pytest.raises(MissingCredentialsError) as errore:
        require_credentials()
    messaggio = str(errore.value)
    assert URL_VARIABLE in messaggio
    assert KEY_VARIABLE in messaggio
    # L'errore deve dire come rimediare, non solo che qualcosa manca.
    assert "how-to-api" in messaggio


def test_credenziali_complete_non_sollevano(tmp_path: Path) -> None:
    percorso = tmp_path / ".env"
    percorso.write_text(f"{URL_VARIABLE}={FAKE_URL}\n{KEY_VARIABLE}={FAKE_KEY}\n", encoding="utf-8")
    assert require_credentials(percorso).is_complete


def test_messaggio_di_errore_non_contiene_il_valore(tmp_path: Path) -> None:
    """Un errore non deve mai far finire un segreto nei log."""
    percorso = tmp_path / ".env"
    percorso.write_text(f"{KEY_VARIABLE}={FAKE_KEY}\n", encoding="utf-8")
    with pytest.raises(MissingCredentialsError) as errore:
        require_credentials(percorso)
    assert FAKE_KEY not in str(errore.value)
