"""Verifica l'accesso al CDS e riporta l'estensione temporale disponibile.

Da eseguire prima di accodare decine di richieste. Distingue i motivi per cui un
download fallisce, che altrimenti si confondono in un unico errore HTTP:

1. credenziali assenti o incomplete  -> controllo locale
2. token non valido                  -> ``check_authentication`` (restituisce 401)
3. Terms of Use non accettati        -> tentativo di download reale
4. data non ancora pubblicata        -> ``end_datetime`` dei metadati della collection

Due scelte deliberate:

- **la disponibilita' si legge dai metadati**, non sondando date a caso: la collection
  espone ``begin_datetime`` e ``end_datetime``, che sono la fonte autorevole;
- **le licenze non vengono accettate in blocco.** L'API non espone le licenze
  richieste da un singolo dataset (il campo ``licences`` della collection e' nullo) e
  ``get_licences()`` restituisce tutte le decine di licenze del portale. Accettarle
  tutte a nome dell'utente sarebbe un effetto collaterale inaccettabile, quindi
  l'unico test affidabile e' provare a scaricare, e l'accettazione avviene solo su
  una licenza indicata esplicitamente.

Uso:
    python scripts/check_cds_access.py
    python scripts/check_cds_access.py --accept-licence licence-to-use-copernicus-products
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from dwf.credentials import (
    KEY_VARIABLE,
    URL_VARIABLE,
    MissingCredentialsError,
    credential_status,
    require_credentials,
)

DATASET = "reanalysis-era5-single-levels"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Richiesta minima: un punto di griglia, un'ora, una variabile.
PROBE_AREA = [45.0, 10.0, 44.75, 10.25]
# Data storica sicuramente pubblicata: separa i problemi di accesso da quelli di
# disponibilita' temporale.
REFERENCE_DATE = date(2024, 1, 1)

# Licenze che valgono per i prodotti ERA5. Elencate solo per mostrarne lo stato:
# non vengono accettate automaticamente.
RELEVANT_LICENCES = (
    "licence-to-use-copernicus-products",
    "terms-of-use-cds",
)


def probe_payload(moment: date) -> dict[str, Any]:
    return {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": [f"{moment.year:04d}"],
        "month": [f"{moment.month:02d}"],
        "day": [f"{moment.day:02d}"],
        "time": ["00:00"],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": PROBE_AREA,
    }


def try_download(client: Any, moment: date) -> tuple[bool, str]:
    """Scarica un solo punto di griglia per la data indicata."""
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "probe.grib"
        try:
            client.retrieve(DATASET, probe_payload(moment), str(target))
        except Exception as exc:  # il client CDS solleva Exception generiche
            return False, f"{type(exc).__name__}: {exc}"
        if not target.exists() or target.stat().st_size == 0:
            return False, "il servizio ha restituito un file vuoto"
        return True, f"{target.stat().st_size} byte scaricati"


def report_licence_state(client: Any) -> set[str]:
    """Mostra lo stato delle licenze pertinenti a ERA5, senza modificarne nessuna."""
    try:
        accepted = {entry.get("id") for entry in client.get_accepted_licences()}
    except Exception as exc:
        print(f"  impossibile leggere le licenze accettate -> {type(exc).__name__}: {exc}")
        return set()
    for identifier in RELEVANT_LICENCES:
        state = "accettata" if identifier in accepted else "NON accettata"
        print(f"  {identifier}: {state}")
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument(
        "--accept-licence",
        metavar="ID",
        help="Accetta una singola licenza indicata per identificativo.",
    )
    parser.add_argument("--accept-revision", type=int, default=None)
    args = parser.parse_args()

    print("1. credenziali")
    status = credential_status(args.env_file if args.env_file.exists() else None)
    print(f"  url: {status.has_url}, key: {status.has_key} (sorgente: {status.source})")
    try:
        require_credentials()
    except MissingCredentialsError as exc:
        print(f"\nBLOCCATO: {exc}")
        raise SystemExit(1) from None

    from ecmwf.datastores import Client

    client = Client(url=os.environ[URL_VARIABLE], key=os.environ[KEY_VARIABLE])

    print("\n2. autenticazione")
    try:
        info = client.check_authentication()
    except Exception as exc:
        print(f"  FALLITA -> {type(exc).__name__}: {str(exc)[:250]}")
        print(
            "\nBLOCCATO: token non valido. Ricopiarlo da "
            "https://cds.climate.copernicus.eu/how-to-api (da autenticati) in `.env`."
        )
        raise SystemExit(1) from None
    print(f"  OK (ruolo: {info.get('role') if isinstance(info, dict) else '?'})")

    print("\n3. licenze pertinenti a ERA5")
    accepted = report_licence_state(client)
    if args.accept_licence:
        if args.accept_revision is None:
            print("\nBLOCCATO: --accept-licence richiede anche --accept-revision")
            raise SystemExit(2)
        if args.accept_licence in accepted:
            print(f"  {args.accept_licence} era gia' accettata, nessuna azione")
        else:
            client.accept_licence(args.accept_licence, revision=args.accept_revision)
            print(f"  accettata ora: {args.accept_licence} (rev {args.accept_revision})")

    print(f"\n4. estensione temporale di {DATASET}")
    collection = client.get_collection(DATASET)
    begin, end = collection.begin_datetime, collection.end_datetime
    print(f"  da {begin} a {end}")
    latest = end.date() if end is not None else None
    if latest is not None:
        lag = (datetime.now(UTC).date() - latest).days
        print(f"  ultima data disponibile: {latest} (latenza {lag} giorni)")

    print(f"\n5. prova di download reale ({REFERENCE_DATE})")
    ok, detail = try_download(client, REFERENCE_DATE)
    print(f"  {'OK' if ok else 'FALLITA'} -> {detail}")
    if not ok:
        print(
            "\nBLOCCATO. Se il messaggio cita licenza o autorizzazione, accettare i "
            "Terms of Use in fondo al form nella scheda Download di\n"
            f"  https://cds.climate.copernicus.eu/datasets/{DATASET}"
        )
        raise SystemExit(1)

    if latest is not None:
        print(f"\n6. prova di download sull'ultima data disponibile ({latest})")
        ok, detail = try_download(client, latest)
        print(f"  {'OK' if ok else 'FALLITA'} -> {detail}")
        if ok:
            print(f'\nDa impostare in configs/default.yaml ->  end: "{latest.isoformat()}"')


if __name__ == "__main__":
    main()
