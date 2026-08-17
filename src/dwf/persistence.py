"""Salvataggio e caricamento dei modelli in un formato che non puo' eseguire codice.

Il formato predefinito di PyTorch, usato da `torch.save` e `torch.load`, si appoggia a
**pickle**. Pickle non e' un formato dati: e' un linguaggio di serializzazione che
include l'opcode `REDUCE`, il quale invoca un chiamabile arbitrario indicato nel file
stesso. Caricare un checkpoint di provenienza ignota equivale quindi a eseguirne il
codice, con i permessi dell'utente. Non e' un difetto di PyTorch ma di pickle, e per
questo lo stesso rischio riguarda ogni formato che vi si appoggia.

Qui il rischio viene rimosso alla radice invece che mitigato:

- **i pesi** stanno in un archivio `.npz`, che e' uno zip di array `.npy`, e vengono
  letti con ``allow_pickle=False``. Con quel flag NumPy rifiuta gli array di tipo
  ``object``, che sono l'unico canale attraverso cui pickle potrebbe entrare. Nessun
  opcode viene mai interpretato;
- **i metadati** stanno in un file JSON. JSON descrive solo dati: non ha costrutti per
  riferirsi a funzioni o classi, quindi non esiste un payload eseguibile esprimibile.

La differenza rispetto a ``torch.load(weights_only=True)`` e' che quella opzione usa
comunque un unpickler, seppure ristretto a una lista di tipi ammessi: la sicurezza
dipende dalla tenuta di quella lista. Qui non c'e' nessun unpickler da aggirare.

Il modulo lavora deliberatamente con soli array NumPy e non importa torch: cio' che
attraversa il confine di fiducia sono numeri, mai oggetti.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

WEIGHTS_NAME = "weights.npz"
METADATA_NAME = "metadata.json"

# Tipi ammessi per i pesi. Esclude esplicitamente ``object``, che e' il tipo che
# NumPy serializza tramite pickle.
ALLOWED_KINDS = frozenset({"f", "i", "u", "b"})


class PersistenceError(RuntimeError):
    """Errore di salvataggio o caricamento di un modello."""


def _check_array(name: str, array: np.ndarray) -> np.ndarray:
    valori = np.asarray(array)
    if valori.dtype.kind not in ALLOWED_KINDS:
        raise PersistenceError(
            f"Il tensore {name!r} ha tipo {valori.dtype!r}, che NumPy serializzerebbe "
            f"con pickle. Sono ammessi solo tipi numerici o booleani."
        )
    if valori.dtype.hasobject:  # pragma: no cover - gia' escluso da ALLOWED_KINDS
        raise PersistenceError(f"Il tensore {name!r} contiene oggetti Python.")
    return valori


def save_weights(directory: Path, arrays: dict[str, np.ndarray]) -> Path:
    """Scrive i pesi in `.npz`, rifiutando qualunque array non numerico."""
    if not arrays:
        raise PersistenceError("Nessun tensore da salvare")
    controllati = {nome: _check_array(nome, valori) for nome, valori in arrays.items()}
    directory.mkdir(parents=True, exist_ok=True)
    percorso = directory / WEIGHTS_NAME
    # `savez` senza compressione: i pesi sono float densi, comprimerli costerebbe
    # tempo a ogni salvataggio per un guadagno trascurabile.
    with percorso.open("wb") as file:
        np.savez(file, **controllati)
    return percorso


def load_weights(directory: Path) -> dict[str, np.ndarray]:
    """Legge i pesi con `allow_pickle=False`: nessun codice puo' essere eseguito."""
    percorso = directory / WEIGHTS_NAME
    if not percorso.exists():
        raise PersistenceError(f"Pesi assenti: {percorso}")
    try:
        with np.load(percorso, allow_pickle=False) as archivio:
            return {nome: np.asarray(archivio[nome]) for nome in archivio.files}
    except ValueError as errore:
        # NumPy segnala cosi' il tentativo di leggere un array `object` con il flag
        # disattivato: e' esattamente il caso di un file manomesso.
        raise PersistenceError(
            f"Il file {percorso} contiene dati che richiederebbero pickle e non e' "
            f"stato caricato: {errore}"
        ) from errore


def save_metadata(directory: Path, metadata: dict[str, Any]) -> Path:
    """Scrive i metadati in JSON, verificando che siano davvero serializzabili."""
    directory.mkdir(parents=True, exist_ok=True)
    percorso = directory / METADATA_NAME
    try:
        testo = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as errore:
        raise PersistenceError(
            f"I metadati non sono rappresentabili in JSON: {errore}"
        ) from errore
    percorso.write_text(testo, encoding="utf-8")
    return percorso


def load_metadata(directory: Path) -> dict[str, Any]:
    """Legge i metadati JSON."""
    percorso = directory / METADATA_NAME
    if not percorso.exists():
        raise PersistenceError(f"Metadati assenti: {percorso}")
    try:
        contenuto = json.loads(percorso.read_text(encoding="utf-8"))
    except json.JSONDecodeError as errore:
        raise PersistenceError(f"Metadati illeggibili in {percorso}: {errore}") from errore
    if not isinstance(contenuto, dict):
        raise PersistenceError(
            f"I metadati in {percorso} non sono un oggetto JSON ma {type(contenuto).__name__}"
        )
    return contenuto


def save_model(
    directory: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> Path:
    """Salva pesi e metadati nella cartella indicata."""
    save_weights(directory, arrays)
    save_metadata(directory, metadata)
    return directory


def load_model(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Carica pesi e metadati senza mai interpretare codice serializzato."""
    return load_weights(directory), load_metadata(directory)


__all__ = [
    "METADATA_NAME",
    "WEIGHTS_NAME",
    "PersistenceError",
    "load_metadata",
    "load_model",
    "load_weights",
    "save_metadata",
    "save_model",
    "save_weights",
]
