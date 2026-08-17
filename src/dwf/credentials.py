"""Caricamento delle credenziali CDS.

Le credenziali arrivano da variabili d'ambiente o da un file ``.env`` non tracciato.
Il modulo non le stampa, non le registra e non le restituisce mai al chiamante:
espone solo se sono presenti e complete, e le rende disponibili a ``cdsapi``
attraverso l'ambiente del processo.

Nota di sicurezza verificata sul sorgente di ``cdsapi`` 0.7.7: con ``debug=True`` il
client registra un dizionario contenente la chiave. Il client va quindi costruito
sempre con ``debug=False`` (il default).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Nomi letti da `cdsapi.api.get_url_key_verify`, in ordine di precedenza.
URL_VARIABLE = "CDSAPI_URL"
KEY_VARIABLE = "CDSAPI_KEY"
RC_VARIABLE = "CDSAPI_RC"
DEFAULT_RC_NAME = ".cdsapirc"
DEFAULT_URL = "https://cds.climate.copernicus.eu/api"

_SETUP_HINT = (
    "Ottenere il Personal Access Token da https://cds.climate.copernicus.eu/how-to-api "
    "(occorre essere registrati e autenticati), accettare i Terms of Use del dataset su "
    "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels e scrivere "
    "CDSAPI_URL e CDSAPI_KEY in un file `.env` nella radice del progetto oppure in "
    "`~/.cdsapirc`. La procedura completa e' nel README. Il file non va committato."
)


class MissingCredentialsError(RuntimeError):
    """Le credenziali CDS non sono disponibili o sono incomplete."""


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    """Esito della risoluzione delle credenziali, senza alcun valore sensibile."""

    has_url: bool
    has_key: bool
    source: str

    @property
    def is_complete(self) -> bool:
        return self.has_url and self.has_key


def parse_env_file(path: Path) -> dict[str, str]:
    """Legge un file in formato ``CHIAVE=valore``, ignorando commenti e righe vuote.

    Implementazione minima e deliberata: evita una dipendenza aggiuntiva per un
    formato di tre righe. Non interpreta espansioni di variabili ne' virgolette
    annidate, che nel nostro caso non servono e sarebbero solo superficie d'errore.

    Il file e' letto con ``utf-8-sig``: su Windows sia Notepad sia
    ``Set-Content -Encoding utf8`` di PowerShell 5.1 premettono un BOM, che
    altrimenti finirebbe nel nome della prima variabile rendendola invisibile.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name:
            values[name] = value
    return values


def load_env_file(path: Path, *, override: bool = False) -> list[str]:
    """Porta le voci di un file ``.env`` nell'ambiente del processo.

    Restituisce i soli **nomi** delle variabili impostate, mai i valori. Per default
    l'ambiente esistente vince, cosi' che in Docker un segreto iniettato dall'esterno
    non venga sovrascritto da un file rimasto nell'immagine.
    """
    applied: list[str] = []
    for name, value in parse_env_file(path).items():
        if not value:
            continue
        if override or not os.environ.get(name):
            os.environ[name] = value
            applied.append(name)
    return applied


def credential_status(env_file: Path | None = None) -> CredentialStatus:
    """Verifica la disponibilita' delle credenziali senza esporne il contenuto."""
    if env_file is not None:
        load_env_file(env_file)

    if os.environ.get(URL_VARIABLE) and os.environ.get(KEY_VARIABLE):
        return CredentialStatus(has_url=True, has_key=True, source="variabili d'ambiente")

    rc_path = Path(os.environ.get(RC_VARIABLE) or Path.home() / DEFAULT_RC_NAME)
    if rc_path.exists():
        entries = _parse_rc_file(rc_path)
        return CredentialStatus(
            has_url=bool(entries.get("url")),
            has_key=bool(entries.get("key")),
            source=f"file {rc_path}",
        )

    return CredentialStatus(
        has_url=bool(os.environ.get(URL_VARIABLE)),
        has_key=bool(os.environ.get(KEY_VARIABLE)),
        source="nessuna sorgente trovata",
    )


def _parse_rc_file(path: Path) -> dict[str, str]:
    """Legge ``~/.cdsapirc``, che usa la sintassi ``chiave: valore``."""
    entries: dict[str, str] = {}
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError:
        return entries
    for raw_line in content.splitlines():
        name, separator, value = raw_line.partition(":")
        if separator and name.strip() in {"url", "key"}:
            entries[name.strip()] = value.strip()
    return entries


def require_credentials(env_file: Path | None = None) -> CredentialStatus:
    """Verifica le credenziali e solleva un errore leggibile se mancano.

    Default fail-closed: nessuna richiesta al CDS viene tentata senza credenziali
    complete, cosi' l'errore arriva subito e non dopo minuti di attesa in coda.
    """
    status = credential_status(env_file)
    if status.is_complete:
        return status
    missing = [
        name
        for name, present in ((URL_VARIABLE, status.has_url), (KEY_VARIABLE, status.has_key))
        if not present
    ]
    raise MissingCredentialsError(
        f"Credenziali CDS incomplete: manca {', '.join(missing)} "
        f"(sorgente esaminata: {status.source}). {_SETUP_HINT}"
    )
