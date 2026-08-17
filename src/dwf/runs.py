"""Esecuzioni di addestramento avviate dall'esterno, in processi separati.

L'addestramento dura ore. Farlo partire dentro il processo che serve le pagine
bloccherebbe l'interfaccia per tutto il tempo e la perderebbe al primo riavvio, quindi
ogni esecuzione e' un processo a se', con la propria cartella, la propria
configurazione salvata su file e il proprio registro.

Due conseguenze utili: l'esecuzione sopravvive alla chiusura della pagina, e resta
riproducibile da riga di comando, perche' quello che il sito lancia e' esattamente il
comando che un utente scriverebbe a mano.

Sulla validazione dei parametri: arrivano da un modulo web, quindi non sono attendibili.
Non vengono interpretati qui, vengono dati in pasto allo **stesso schema** che valida la
configurazione del progetto. Se lo schema li rifiuta, l'esecuzione non parte e l'errore
e' quello che si vedrebbe da terminale.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from dwf.config import Config

REGISTRO = "run.json"
DIARIO = "training.log"

# Un nome di esecuzione finisce in un percorso: si ammettono solo caratteri che non
# possono cambiarne il significato.
NOME_AMMESSO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class RunError(RuntimeError):
    """Errore nella gestione di un'esecuzione di addestramento."""


@dataclass(frozen=True)
class Esecuzione:
    """Una singola esecuzione, come risulta dal suo registro su disco."""

    nome: str
    cartella: Path
    avviata: datetime
    pid: int | None
    creazione_processo: float | None = None
    comando: list[str] = field(default_factory=list)
    parametri: dict[str, Any] = field(default_factory=dict)
    terminata: datetime | None = None
    esito: str | None = None

    @property
    def diario(self) -> Path:
        return self.cartella / DIARIO

    @property
    def attiva(self) -> bool:
        """Vera se il processo esiste ancora.

        Si controlla il sistema, non il registro: un'esecuzione interrotta da un riavvio
        lascerebbe un registro che dice "in corso" per sempre.
        """
        if self.pid is None or self.terminata is not None:
            return False
        return processo_vivo(self.pid, self.creazione_processo)

    @property
    def stato(self) -> str:
        """Esito dedotto dalle prove su disco, non da quello che il registro ricorda.

        Chi avvia il processo e' la pagina web, che non e' li' quando il processo
        finisce e non puo' quindi scriverne l'esito: fidarsi del registro farebbe
        risultare "interrotta" anche una corsa arrivata in fondo. Le epoche
        effettivamente scritte, confrontate con quelle chieste, lo dicono senza
        bisogno di un guardiano sempre acceso.
        """
        if self.attiva:
            return "in corso"
        if self.esito:
            return self.esito
        chieste = self.parametri.get("epochs")
        concluse = len(cronologia(self, int(self.parametri.get("fold", 0))))
        if chieste and concluse >= int(chieste):
            return "conclusa"
        if concluse:
            return f"interrotta a {concluse}/{chieste} epoche"
        return "interrotta prima della prima epoca"


def istante_di_creazione(pid: int) -> float | None:
    """Istante in cui il processo e' nato, usato come sua identita'."""
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def processo_vivo(pid: int, creazione: float | None = None) -> bool:
    """Se il processo esiste **ed e' ancora quello di prima**, senza inviargli nulla.

    Il solo numero non basta: i numeri di processo vengono riciclati, e una corsa
    conclusa da giorni tornerebbe a risultare in corso appena il sistema riassegna
    quel numero a qualcos'altro. La coppia numero piu' istante di nascita, invece,
    identifica un processo in modo stabile.
    """
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil e' fra le dipendenze
        return False

    try:
        processo = psutil.Process(pid)
        if processo.status() == psutil.STATUS_ZOMBIE:
            return False
        if creazione is not None and abs(processo.create_time() - creazione) > 1.0:
            return False
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------- #
# Percorsi
# --------------------------------------------------------------------------- #


def radice_esecuzioni(config: Config) -> Path:
    return config.models_dir / "runs"


def cartella_esecuzione(config: Config, nome: str) -> Path:
    """Cartella di un'esecuzione, verificando che il nome non esca dalla radice."""
    if not NOME_AMMESSO.match(nome):
        raise RunError(
            f"Nome non ammesso: {nome!r}. Sono ammessi lettere, cifre, trattino e "
            "trattino basso, fino a 64 caratteri."
        )
    radice = radice_esecuzioni(config)
    percorso = (radice / nome).resolve()
    if percorso != radice and radice.resolve() not in percorso.parents:
        raise RunError(f"Il nome porta fuori dalla cartella delle esecuzioni: {nome!r}")
    return percorso


def nome_proposto(prefisso: str = "run") -> str:
    return f"{prefisso}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"


# --------------------------------------------------------------------------- #
# Costruzione della configurazione
# --------------------------------------------------------------------------- #


def config_con_parametri(base: Config, nome: str, parametri: dict[str, Any]) -> Config:
    """Applica i parametri scelti dall'utente, validandoli con lo schema del progetto.

    I parametri accettati sono solo quelli dell'addestramento e la variante del modello:
    non si permette di ridefinire percorsi, area o variabili, perche' cambierebbero il
    significato dei dati invece che il modo di addestrare.
    """
    ammessi_training = {
        "epochs",
        "learning_rate",
        "batch_size",
        "crop_size",
        "samples_per_epoch",
        "seed",
        "grad_clip_norm",
        "num_workers",
    }
    ammessi_modello = {"variant", "anchor_diurnal", "base_channels", "depth", "blocks_per_level"}

    sconosciuti = set(parametri) - ammessi_training - ammessi_modello - {"loss_weights"}
    if sconosciuti:
        raise RunError(f"Parametri non modificabili: {sorted(sconosciuti)}")

    aggiornamenti_training = {k: v for k, v in parametri.items() if k in ammessi_training}
    aggiornamenti_modello = {k: v for k, v in parametri.items() if k in ammessi_modello}

    # `model_copy` **non rivalida**: accetterebbe zero epoche o un passo di apprendimento
    # negativo senza protestare, e il difetto emergerebbe solo a meta' addestramento.
    # Ricostruire dal dizionario fa passare i valori attraverso lo schema, che e' l'unico
    # punto in cui i vincoli sono dichiarati.
    base_training = base.training.model_dump()
    if "loss_weights" in parametri:
        base_training["loss_weights"] = {
            **base_training["loss_weights"],
            **dict(parametri["loss_weights"]),
        }
    training = type(base.training).model_validate({**base_training, **aggiornamenti_training})
    modello = type(base.model).model_validate({**base.model.model_dump(), **aggiornamenti_modello})
    percorsi = base.paths.model_copy(update={"models_subdir": f"runs/{nome}"})
    return base.model_copy(
        update={"training": training, "model": modello, "paths": percorsi}
    )


def salva_configurazione(config: Config, destinazione: Path) -> Path:
    """Scrive la configurazione dell'esecuzione, cosi' resta ripetibile da terminale."""
    destinazione.mkdir(parents=True, exist_ok=True)
    percorso = destinazione / "config.yaml"
    percorso.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return percorso


# --------------------------------------------------------------------------- #
# Avvio, lettura, arresto
# --------------------------------------------------------------------------- #


def avvia(
    base: Config,
    *,
    nome: str | None = None,
    fold: int = 0,
    parametri: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> Esecuzione:
    """Prepara la cartella, salva la configurazione e lancia il processo."""
    nome = nome or nome_proposto()
    cartella = cartella_esecuzione(base, nome)
    if cartella.exists() and (cartella / REGISTRO).exists():
        raise RunError(f"Esiste gia' un'esecuzione con questo nome: {nome}")

    config = config_con_parametri(base, nome, parametri or {})
    percorso_config = salva_configurazione(config, cartella)

    radice = project_root or base.project_root
    comando = [
        sys.executable,
        str(radice / "scripts" / "train_model.py"),
        "--config",
        str(percorso_config),
        "--fold",
        str(int(fold)),
    ]

    diario = cartella / DIARIO
    with diario.open("w", encoding="utf-8") as flusso:
        # Argomenti come lista e senza shell: nessun valore puo' essere reinterpretato
        # come comando.
        processo = subprocess.Popen(
            comando,
            stdout=flusso,
            stderr=subprocess.STDOUT,
            cwd=str(radice),
            shell=False,
        )

    esecuzione = Esecuzione(
        nome=nome,
        cartella=cartella,
        avviata=datetime.now(UTC),
        pid=processo.pid,
        creazione_processo=istante_di_creazione(processo.pid),
        comando=comando,
        parametri=dict(parametri or {}) | {"fold": int(fold)},
    )
    scrivi_registro(esecuzione)
    return esecuzione


def scrivi_registro(esecuzione: Esecuzione) -> None:
    esecuzione.cartella.mkdir(parents=True, exist_ok=True)
    (esecuzione.cartella / REGISTRO).write_text(
        json.dumps(
            {
                "nome": esecuzione.nome,
                "avviata": esecuzione.avviata.isoformat(),
                "pid": esecuzione.pid,
                "creazione_processo": esecuzione.creazione_processo,
                "comando": esecuzione.comando,
                "parametri": esecuzione.parametri,
                "terminata": esecuzione.terminata.isoformat() if esecuzione.terminata else None,
                "esito": esecuzione.esito,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def leggi(config: Config, nome: str) -> Esecuzione | None:
    cartella = cartella_esecuzione(config, nome)
    percorso = cartella / REGISTRO
    if not percorso.exists():
        return None
    dati = json.loads(percorso.read_text(encoding="utf-8"))
    return Esecuzione(
        nome=dati["nome"],
        cartella=cartella,
        avviata=datetime.fromisoformat(dati["avviata"]),
        pid=dati.get("pid"),
        creazione_processo=dati.get("creazione_processo"),
        comando=dati.get("comando", []),
        parametri=dati.get("parametri", {}),
        terminata=(
            datetime.fromisoformat(dati["terminata"]) if dati.get("terminata") else None
        ),
        esito=dati.get("esito"),
    )


def elenca(config: Config) -> list[Esecuzione]:
    radice = radice_esecuzioni(config)
    if not radice.exists():
        return []
    esecuzioni = []
    for cartella in sorted(radice.iterdir(), reverse=True):
        if not cartella.is_dir():
            continue
        try:
            esecuzione = leggi(config, cartella.name)
        except RunError:
            continue
        if esecuzione is not None:
            esecuzioni.append(esecuzione)
    return esecuzioni


def coda_del_diario(esecuzione: Esecuzione, righe: int = 40) -> str:
    if not esecuzione.diario.exists():
        return ""
    contenuto = esecuzione.diario.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(contenuto[-righe:])


def cronologia(esecuzione: Esecuzione, fold: int = 0) -> list[dict[str, Any]]:
    """Cronologia delle epoche gia' scritte, per seguire l'andamento durante la corsa."""
    percorso = esecuzione.cartella / f"fold_{fold:02d}" / "history.json"
    if not percorso.exists():
        return []
    try:
        return json.loads(percorso.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Il file viene riscritto a fine addestramento: una lettura a meta' e' normale.
        return []


def ferma(config: Config, nome: str) -> bool:
    """Chiede al processo di terminare. Restituisce vero se era in corso."""
    esecuzione = leggi(config, nome)
    if esecuzione is None or not esecuzione.attiva or esecuzione.pid is None:
        return False
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(esecuzione.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            os.kill(esecuzione.pid, signal.SIGTERM)
    except OSError as errore:
        raise RunError(f"Non e' stato possibile fermare l'esecuzione: {errore}") from errore

    scrivi_registro(
        Esecuzione(
            nome=esecuzione.nome,
            cartella=esecuzione.cartella,
            avviata=esecuzione.avviata,
            pid=esecuzione.pid,
            creazione_processo=esecuzione.creazione_processo,
            comando=esecuzione.comando,
            parametri=esecuzione.parametri,
            terminata=datetime.now(UTC),
            esito="fermata dall'utente",
        )
    )
    return True


def confronto(config: Config) -> list[dict[str, Any]]:
    """Una riga per esecuzione, con i parametri che la distinguono e il suo risultato."""
    righe: list[dict[str, Any]] = []
    for esecuzione in elenca(config):
        storia = cronologia(esecuzione, int(esecuzione.parametri.get("fold", 0)))
        valide = [
            r["val_loss"]
            for r in storia
            if isinstance(r.get("val_loss"), int | float) and r["val_loss"] == r["val_loss"]
        ]
        righe.append(
            {
                "esecuzione": esecuzione.nome,
                "stato": esecuzione.stato,
                "avviata": esecuzione.avviata.strftime("%Y-%m-%d %H:%M"),
                "epoche": len(storia),
                "migliore": min(valide) if valide else None,
                **{
                    chiave: valore
                    for chiave, valore in esecuzione.parametri.items()
                    if chiave != "loss_weights"
                },
            }
        )
    return righe


__all__ = [
    "DIARIO",
    "REGISTRO",
    "Esecuzione",
    "RunError",
    "avvia",
    "cartella_esecuzione",
    "coda_del_diario",
    "config_con_parametri",
    "confronto",
    "cronologia",
    "elenca",
    "ferma",
    "istante_di_creazione",
    "leggi",
    "nome_proposto",
    "processo_vivo",
    "radice_esecuzioni",
]
