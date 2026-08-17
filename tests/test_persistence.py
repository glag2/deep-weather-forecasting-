"""Verifiche del formato di salvataggio, comprese quelle di sicurezza.

Il punto centrale non e' che il salvataggio funzioni, ma che un file manomesso **non
possa eseguire codice**. Per dimostrarlo il test costruisce un payload realmente
dannoso e verifica due cose distinte: che con pickle attivo quel payload verrebbe
davvero eseguito, e che la nostra funzione di caricamento lo rifiuta senza eseguirlo.
Senza il primo controllo il secondo sarebbe una verifica vuota, superata anche da un
payload inerte.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dwf.persistence import (
    METADATA_NAME,
    WEIGHTS_NAME,
    PersistenceError,
    load_metadata,
    load_model,
    load_weights,
    save_metadata,
    save_model,
    save_weights,
)

# Percorso che il payload malevolo tenta di creare. Viene impostato dalla fixture e
# letto dal metodo `__reduce__`, che pickle invoca al momento del caricamento.
SENTINELLA: list[Path] = []


class CaricoMalevolo:
    """Oggetto che, se deserializzato con pickle, scrive un file.

    `__reduce__` dichiara a pickle come ricostruire l'oggetto: qui dichiara una
    chiamata a `Path.write_text`. E' la stessa tecnica con cui un checkpoint
    scaricato da terzi potrebbe eseguire qualunque comando.
    """

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (Path.write_text, (SENTINELLA[0], "codice eseguito"))


@pytest.fixture(autouse=True)
def _sentinella(tmp_path: Path):
    SENTINELLA.clear()
    SENTINELLA.append(tmp_path / "prova-di-esecuzione.txt")
    yield
    SENTINELLA.clear()


def _pesi_di_prova() -> dict[str, np.ndarray]:
    generatore = np.random.default_rng(0)
    return {
        "encoder.weight": generatore.normal(size=(8, 4, 3, 3)).astype(np.float32),
        "encoder.bias": np.zeros(8, dtype=np.float32),
        "contatore": np.array([3], dtype=np.int64),
    }


class TestSicurezza:
    def test_il_payload_e_davvero_pericoloso(self, tmp_path: Path) -> None:
        # Controllo di validita' del test: con pickle attivo il payload esegue.
        # Se questo test fallisse, quello successivo non proverebbe nulla.
        percorso = tmp_path / "malevolo.npy"
        np.save(percorso, np.array([CaricoMalevolo()], dtype=object), allow_pickle=True)
        assert not SENTINELLA[0].exists()

        np.load(percorso, allow_pickle=True)

        assert SENTINELLA[0].exists()
        assert SENTINELLA[0].read_text(encoding="utf-8") == "codice eseguito"

    def test_il_caricamento_rifiuta_il_payload_senza_eseguirlo(self, tmp_path: Path) -> None:
        cartella = tmp_path / "modello"
        cartella.mkdir()
        with (cartella / WEIGHTS_NAME).open("wb") as file:
            np.savez(file, malevolo=np.array([CaricoMalevolo()], dtype=object))

        with pytest.raises(PersistenceError, match="pickle"):
            load_weights(cartella)

        assert not SENTINELLA[0].exists(), "il payload e' stato eseguito"

    def test_il_salvataggio_rifiuta_array_di_oggetti(self, tmp_path: Path) -> None:
        with pytest.raises(PersistenceError, match="pickle"):
            save_weights(tmp_path, {"cattivo": np.array([CaricoMalevolo()], dtype=object)})

    def test_il_salvataggio_rifiuta_stringhe_arbitrarie_come_pesi(self, tmp_path: Path) -> None:
        # Le stringhe non sono eseguibili, ma non sono pesi: ammetterle allargherebbe
        # inutilmente la superficie del formato.
        with pytest.raises(PersistenceError):
            save_weights(tmp_path, {"testo": np.array(["a", "b"])})

    def test_i_metadati_non_possono_contenere_oggetti(self, tmp_path: Path) -> None:
        with pytest.raises(PersistenceError, match="JSON"):
            save_metadata(tmp_path, {"payload": CaricoMalevolo()})

    def test_i_metadati_sono_solo_dati_anche_se_sembrano_codice(self, tmp_path: Path) -> None:
        # Una stringa che somiglia a codice resta una stringa: JSON non ha modo di
        # trasformarla in una chiamata.
        sospetto = "__import__('os').system('echo attacco')"
        save_metadata(tmp_path, {"nota": sospetto})
        assert load_metadata(tmp_path)["nota"] == sospetto
        assert not SENTINELLA[0].exists()

    def test_il_file_dei_pesi_non_contiene_opcode_pickle(self, tmp_path: Path) -> None:
        # Verifica diretta sui byte: la firma di un flusso pickle e' il protocollo
        # `\x80` seguito dalla versione. Un archivio di soli array numerici non deve
        # contenerla.
        save_weights(tmp_path, _pesi_di_prova())
        contenuto = (tmp_path / WEIGHTS_NAME).read_bytes()
        assert b"\x80\x04" not in contenuto
        assert b"\x80\x05" not in contenuto


class TestSalvataggioECaricamento:
    def test_ciclo_completo_conserva_i_valori(self, tmp_path: Path) -> None:
        pesi = _pesi_di_prova()
        save_weights(tmp_path, pesi)
        riletti = load_weights(tmp_path)
        assert set(riletti) == set(pesi)
        for nome, valori in pesi.items():
            assert np.array_equal(riletti[nome], valori)

    def test_ciclo_completo_conserva_i_tipi(self, tmp_path: Path) -> None:
        pesi = _pesi_di_prova()
        save_weights(tmp_path, pesi)
        riletti = load_weights(tmp_path)
        for nome, valori in pesi.items():
            assert riletti[nome].dtype == valori.dtype

    def test_salva_e_carica_modello_completo(self, tmp_path: Path) -> None:
        metadati = {"fold": 0, "epoch": 13, "val_loss": 0.915, "in_channels": 245}
        save_model(tmp_path, _pesi_di_prova(), metadati)
        pesi, letti = load_model(tmp_path)
        assert letti == metadati
        assert "encoder.weight" in pesi

    def test_crea_la_cartella_se_manca(self, tmp_path: Path) -> None:
        destinazione = tmp_path / "a" / "b" / "c"
        save_model(destinazione, _pesi_di_prova(), {"fold": 1})
        assert (destinazione / WEIGHTS_NAME).exists()
        assert (destinazione / METADATA_NAME).exists()

    def test_rifiuta_di_salvare_senza_tensori(self, tmp_path: Path) -> None:
        with pytest.raises(PersistenceError, match="Nessun tensore"):
            save_weights(tmp_path, {})

    def test_errore_chiaro_se_i_pesi_mancano(self, tmp_path: Path) -> None:
        with pytest.raises(PersistenceError, match="Pesi assenti"):
            load_weights(tmp_path)

    def test_errore_chiaro_se_i_metadati_mancano(self, tmp_path: Path) -> None:
        with pytest.raises(PersistenceError, match="Metadati assenti"):
            load_metadata(tmp_path)

    def test_errore_chiaro_se_i_metadati_sono_corrotti(self, tmp_path: Path) -> None:
        (tmp_path / METADATA_NAME).write_text("{non json", encoding="utf-8")
        with pytest.raises(PersistenceError, match="illeggibili"):
            load_metadata(tmp_path)

    def test_rifiuta_metadati_che_non_sono_un_oggetto(self, tmp_path: Path) -> None:
        (tmp_path / METADATA_NAME).write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(PersistenceError, match="non sono un oggetto"):
            load_metadata(tmp_path)

    def test_rifiuta_valori_non_finiti_nei_metadati(self, tmp_path: Path) -> None:
        # `float('nan')` non e' JSON valido: ammetterlo produrrebbe file che altri
        # lettori rifiuterebbero.
        with pytest.raises(PersistenceError, match="JSON"):
            save_metadata(tmp_path, {"perdita": float("nan")})

    def test_i_metadati_sono_leggibili_da_qualunque_lettore_json(self, tmp_path: Path) -> None:
        save_metadata(tmp_path, {"fold": 2, "nome": "città"})
        contenuto = json.loads((tmp_path / METADATA_NAME).read_text(encoding="utf-8"))
        assert contenuto == {"fold": 2, "nome": "città"}

    def test_conserva_liste_annidate_nei_metadati(self, tmp_path: Path) -> None:
        # Il layout dei canali e' una lista di dizionari: deve sopravvivere intatta.
        canali = [{"channel_index": 0, "name": "t2m_t-0", "normalized": True}]
        save_metadata(tmp_path, {"channels": canali})
        assert load_metadata(tmp_path)["channels"] == canali
