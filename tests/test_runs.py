"""Test della gestione delle esecuzioni di addestramento.

I parametri arrivano da un modulo web, quindi la parte da verificare non e' che il caso
normale funzioni: e' che i casi ostili vengano fermati. Un nome che risale l'albero delle
cartelle, un parametro che ridefinisce i percorsi, un valore fuori intervallo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dwf.config import Config
from dwf.runs import (
    Esecuzione,
    RunError,
    cartella_esecuzione,
    config_con_parametri,
    elenca,
    istante_di_creazione,
    leggi,
    nome_proposto,
    processo_vivo,
    salva_configurazione,
    scrivi_registro,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config.load(CONFIG_PATH, project_root=tmp_path)


# --------------------------------------------------------------------------- #
# Nomi e percorsi
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "nome",
    ["../fuga", "..", "a/b", "a\\b", "", "con spazio", "x" * 65, ".nascosto"],
)
def test_i_nomi_pericolosi_sono_rifiutati(config: Config, nome: str) -> None:
    with pytest.raises(RunError):
        cartella_esecuzione(config, nome)


@pytest.mark.parametrize("nome", ["run_20260818_010203", "prova-1", "A", "x" * 64])
def test_i_nomi_normali_sono_accettati(config: Config, nome: str) -> None:
    percorso = cartella_esecuzione(config, nome)
    assert percorso.name == nome
    assert percorso.parent.name == "runs"


def test_il_nome_proposto_e_accettabile(config: Config) -> None:
    assert cartella_esecuzione(config, nome_proposto()).name.startswith("run_")


# --------------------------------------------------------------------------- #
# Parametri
# --------------------------------------------------------------------------- #


def test_i_parametri_di_addestramento_vengono_applicati(config: Config) -> None:
    modificata = config_con_parametri(
        config, "prova", {"epochs": 3, "learning_rate": 0.001, "seed": 7}
    )
    assert modificata.training.epochs == 3
    assert modificata.training.learning_rate == pytest.approx(0.001)
    assert modificata.training.seed == 7
    # L'originale non viene toccato: due esecuzioni non devono influenzarsi.
    assert config.training.epochs != 3 or config.training.seed != 7


def test_ogni_esecuzione_scrive_in_una_cartella_propria(config: Config) -> None:
    prima = config_con_parametri(config, "prima", {})
    seconda = config_con_parametri(config, "seconda", {})
    assert prima.fold_dir(0) != seconda.fold_dir(0)
    assert prima.fold_dir(0).parent.name == "prima"


def test_non_si_possono_ridefinire_i_percorsi(config: Config) -> None:
    """Cambiare i percorsi cambierebbe quali dati si stanno usando, non come si allena."""
    with pytest.raises(RunError, match="non modificabili"):
        config_con_parametri(config, "prova", {"data_root": "/altro"})


def test_non_si_possono_ridefinire_area_o_variabili(config: Config) -> None:
    with pytest.raises(RunError, match="non modificabili"):
        config_con_parametri(config, "prova", {"region": {"north": 90}})


@pytest.mark.parametrize(
    "parametri",
    [
        {"epochs": 0},
        {"epochs": -1},
        {"learning_rate": -0.1},
        {"batch_size": 0},
        {"crop_size": 0},
        {"variant": "inesistente"},
    ],
)
def test_un_valore_fuori_intervallo_viene_rifiutato_dallo_schema(
    config: Config, parametri: dict
) -> None:
    """La validazione e' quella del progetto: il sito non deve poterla aggirare.

    Il rischio concreto non e' teorico: ``model_copy`` applica gli aggiornamenti **senza
    rivalidarli**, quindi la strada comoda avrebbe accettato zero epoche e un passo di
    apprendimento negativo, e il problema sarebbe comparso a meta' addestramento.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        config_con_parametri(config, "prova", parametri)


def test_i_pesi_della_perdita_si_aggiornano_senza_azzerare_gli_altri(config: Config) -> None:
    nomi = list(config.training.loss_weights.model_dump())
    scelto = nomi[0]
    modificata = config_con_parametri(config, "prova", {"loss_weights": {scelto: 2.5}})
    aggiornati = modificata.training.loss_weights.model_dump()
    assert aggiornati[scelto] == pytest.approx(2.5)
    for altro in nomi[1:]:
        assert aggiornati[altro] == pytest.approx(
            config.training.loss_weights.model_dump()[altro]
        )


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #


def test_la_configurazione_salvata_si_rilegge(config: Config, tmp_path: Path) -> None:
    """Deve restare ripetibile da terminale: se non si rilegge, non lo e'."""
    modificata = config_con_parametri(config, "prova", {"epochs": 2})
    percorso = salva_configurazione(modificata, tmp_path / "esecuzione")
    riletta = Config.load(percorso, project_root=config.project_root)
    assert riletta.training.epochs == 2
    assert riletta.paths.models_subdir == "runs/prova"


def test_il_registro_si_rilegge(config: Config) -> None:
    from datetime import UTC, datetime

    esecuzione = Esecuzione(
        nome="prova",
        cartella=cartella_esecuzione(config, "prova"),
        avviata=datetime.now(UTC),
        pid=1,
        comando=["python", "train.py"],
        parametri={"epochs": 2, "fold": 0},
    )
    scrivi_registro(esecuzione)
    riletta = leggi(config, "prova")
    assert riletta is not None
    assert riletta.parametri == {"epochs": 2, "fold": 0}
    assert [e.nome for e in elenca(config)] == ["prova"]


def test_senza_esecuzioni_l_elenco_e_vuoto(config: Config) -> None:
    assert elenca(config) == []
    assert leggi(config, "inesistente") is None


def test_un_registro_che_dice_in_corso_non_basta(config: Config) -> None:
    """Un riavvio della macchina lascia registri che dichiarano un processo mai finito.

    Lo stato deve venire dal sistema, non da quello che il registro ricorda.
    """
    from datetime import UTC, datetime

    esecuzione = Esecuzione(
        nome="fantasma",
        cartella=cartella_esecuzione(config, "fantasma"),
        avviata=datetime.now(UTC),
        pid=999_999,
        parametri={},
    )
    scrivi_registro(esecuzione)
    riletta = leggi(config, "fantasma")
    assert riletta is not None
    assert not riletta.attiva
    assert riletta.stato.startswith("interrotta")


def test_una_corsa_arrivata_in_fondo_non_risulta_interrotta(config: Config) -> None:
    """Chi avvia la corsa e' la pagina web, che non c'e' piu' quando la corsa finisce.

    Senza dedurre l'esito dalle epoche scritte, ogni corsa completata verrebbe
    presentata come interrotta.
    """
    import json
    from datetime import UTC, datetime

    cartella = cartella_esecuzione(config, "completata")
    esecuzione = Esecuzione(
        nome="completata",
        cartella=cartella,
        avviata=datetime.now(UTC),
        pid=999_999,
        parametri={"epochs": 2, "fold": 0},
    )
    scrivi_registro(esecuzione)
    fold = cartella / "fold_00"
    fold.mkdir(parents=True, exist_ok=True)
    (fold / "history.json").write_text(
        json.dumps([{"epoch": 0, "val_loss": 3.0}, {"epoch": 1, "val_loss": 2.0}]),
        encoding="utf-8",
    )

    riletta = leggi(config, "completata")
    assert riletta is not None
    assert riletta.stato == "conclusa"

    (fold / "history.json").write_text(
        json.dumps([{"epoch": 0, "val_loss": 3.0}]), encoding="utf-8"
    )
    parziale = leggi(config, "completata")
    assert parziale is not None
    assert parziale.stato == "interrotta a 1/2 epoche"


def test_il_processo_corrente_risulta_vivo() -> None:
    import os

    assert processo_vivo(os.getpid())
    assert not processo_vivo(0)
    assert not processo_vivo(-1)


def test_un_numero_riciclato_non_fa_risultare_viva_una_corsa_vecchia() -> None:
    """I numeri di processo vengono riusati.

    Senza legare il numero all'istante di nascita, una corsa conclusa da giorni
    tornerebbe a risultare in corso appena il sistema riassegna quel numero a un
    processo qualsiasi.
    """
    import os

    corrente = os.getpid()
    nascita = istante_di_creazione(corrente)
    assert nascita is not None
    assert processo_vivo(corrente, nascita) is True
    # Stesso numero, nascita diversa: e' un altro processo.
    assert processo_vivo(corrente, nascita - 3600.0) is False
