"""Test della raccolta dati della dashboard.

La dashboard deve poter essere aperta su un progetto appena clonato, dove non esiste
ancora nessun artefatto, e dire che cosa manca invece di sollevare un'eccezione. Buona
parte di questi test verifica proprio il comportamento in assenza di dati, che e' il
caso piu' facile da rompere e il piu' fastidioso da scoprire a schermo.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from dwf.config import Config
from dwf.dashboard import (
    TECNOLOGIE,
    Riquadro,
    _accorcia,
    accuratezze_ingannevoli,
    copertura_mensile,
    curva_apprendimento,
    informazioni_modello,
    metriche,
    panoramica,
    per_scadenza,
    riepilogo_metriche,
    risorse,
    spazio_dati,
    struttura_fold,
)
from dwf.tables import METRICS, SLOTS, cast_to_schema

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Configurazione reale con radice dati vuota in una directory temporanea."""
    return Config.load(CONFIG_PATH, project_root=tmp_path)


def _scrivi_slots(config: Config, righe: list[dict]) -> None:
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    frame = cast_to_schema(pl.DataFrame(righe), SLOTS)
    frame.write_parquet(SLOTS.path(config.tables_dir))


def _slot(indice: int, istante: datetime, *, usable: bool) -> dict:
    return {
        "slot_index": indice,
        "valid_time": istante,
        "year": istante.year,
        "month": istante.month,
        "day": istante.day,
        "hour": istante.hour,
        "slot_of_day": indice % 3,
        "day_of_year": istante.timetuple().tm_yday,
        "source_month": f"{istante.year}-{istante.month:02d}",
        "split": "train",
        "usable": usable,
    }


# --------------------------------------------------------------------------- #
# Progetto senza artefatti
# --------------------------------------------------------------------------- #


class TestProgettoVuoto:
    def test_la_panoramica_non_solleva_senza_dati(self, config: Config) -> None:
        riquadri = panoramica(config)
        assert riquadri
        assert all(isinstance(r, Riquadro) for r in riquadri)

    def test_la_panoramica_dice_che_cosa_manca(self, config: Config) -> None:
        testo = " ".join(f"{r.valore} {r.nota}" for r in panoramica(config))
        assert "ingest_era5" in testo

    def test_la_copertura_e_assente_non_vuota(self, config: Config) -> None:
        assert copertura_mensile(config) is None

    def test_i_fold_sono_assenti(self, config: Config) -> None:
        assert struttura_fold(config) is None

    def test_le_metriche_sono_assenti(self, config: Config) -> None:
        assert metriche(config, 0) is None

    def test_il_modello_risulta_non_disponibile(self, config: Config) -> None:
        info = informazioni_modello(config, 0)
        assert info["disponibile"] is False
        assert "percorso" in info

    def test_lo_spazio_segnala_le_directory_assenti(self, config: Config) -> None:
        valori = {r.etichetta: r.valore for r in spazio_dati(config)}
        assert "assente" in valori.values()


# --------------------------------------------------------------------------- #
# Copertura dei dati
# --------------------------------------------------------------------------- #


class TestCopertura:
    def test_distingue_catalogati_e_presenti(self, config: Config) -> None:
        """Contare le righe direbbe un numero molto piu' grande del vero.

        `slots.parquet` cataloga l'intero periodo configurato: i mesi non ancora
        ingeriti ci sono comunque, con `usable` falso.
        """
        _scrivi_slots(
            config,
            [
                _slot(0, datetime(2024, 1, 1, 6, tzinfo=UTC), usable=True),
                _slot(1, datetime(2024, 1, 1, 12, tzinfo=UTC), usable=True),
                _slot(2, datetime(2024, 2, 1, 6, tzinfo=UTC), usable=False),
            ],
        )
        copertura = copertura_mensile(config)
        assert copertura is not None
        per_mese = {r["mese"]: r for r in copertura.to_dicts()}
        assert per_mese["2024-01"]["presenti"] == 2
        assert per_mese["2024-02"]["catalogati"] == 1
        assert per_mese["2024-02"]["presenti"] == 0
        assert per_mese["2024-02"]["frazione"] == 0.0

    def test_la_panoramica_conta_solo_gli_utilizzabili(self, config: Config) -> None:
        _scrivi_slots(
            config,
            [
                _slot(0, datetime(2024, 1, 1, 6, tzinfo=UTC), usable=True),
                _slot(1, datetime(2024, 1, 1, 12, tzinfo=UTC), usable=False),
                _slot(2, datetime(2024, 1, 1, 18, tzinfo=UTC), usable=False),
            ],
        )
        riquadro = next(r for r in panoramica(config) if "Slot" in r.etichetta)
        assert riquadro.valore == "1 / 3"

    def test_l_intervallo_riguarda_gli_slot_presenti(self, config: Config) -> None:
        """L'ultima data mostrata deve essere l'ultima **presente**, non l'ultima attesa."""
        _scrivi_slots(
            config,
            [
                _slot(0, datetime(2024, 1, 1, 6, tzinfo=UTC), usable=True),
                _slot(1, datetime(2026, 8, 1, 6, tzinfo=UTC), usable=False),
            ],
        )
        riquadro = next(r for r in panoramica(config) if "Slot" in r.etichetta)
        assert "2024-01-01" in riquadro.nota
        assert "2026" not in riquadro.nota


# --------------------------------------------------------------------------- #
# Metriche
# --------------------------------------------------------------------------- #


class TestMetriche:
    def _scrivi(self, config: Config, righe: list[dict]) -> None:
        config.fold_dir(0).mkdir(parents=True, exist_ok=True)
        cast_to_schema(pl.DataFrame(righe), METRICS).write_parquet(
            METRICS.path(config.fold_dir(0))
        )

    def _riga(self, **campi) -> dict:
        base = {
            "model": "dwf", "split": "test", "fold": 0, "variable": "t2m",
            "lead_slot": -1, "month": -1, "metric": "rmse", "value": 1.0,
            "n_values": 10,
        }
        return {**base, **campi}

    def test_filtra_per_fold_e_blocco(self, config: Config) -> None:
        self._scrivi(
            config,
            [
                self._riga(value=1.0),
                self._riga(split="val", value=2.0),
                self._riga(fold=1, value=3.0),
            ],
        )
        tabella = metriche(config, 0, "test")
        assert tabella is not None
        assert tabella.height == 1
        assert tabella["value"][0] == 1.0

    def test_il_riepilogo_prende_solo_le_righe_aggregate(self, config: Config) -> None:
        """Le righe per mese e per scadenza non vanno rimediate a mano.

        Fare la media delle medie pesarebbe male i mesi con meno casi: la valutazione
        produce gia' la riga aggregata, che e' quella corretta.
        """
        self._scrivi(
            config,
            [
                self._riga(value=1.0),
                self._riga(lead_slot=0, value=9.0),
                self._riga(month=3, value=9.0),
            ],
        )
        tabella = metriche(config, 0, "test")
        assert tabella is not None
        riepilogo = riepilogo_metriche(tabella)
        assert riepilogo.height == 1
        assert riepilogo["value"][0] == 1.0

    def test_il_riepilogo_mette_per_prima_la_temperatura(self, config: Config) -> None:
        """In ordine alfabetico la prima riga sarebbe l'accuratezza sulla neve.

        E' la metrica piu' fraintendibile del progetto e non deve aprire la pagina.
        """
        self._scrivi(
            config,
            [
                self._riga(variable="sf", metric="accuracy", value=0.85),
                self._riga(variable="tp", metric="brier", value=0.18),
                self._riga(variable="t2m", metric="rmse_celsius", value=2.9),
            ],
        )
        tabella = metriche(config, 0, "test")
        assert tabella is not None
        riepilogo = riepilogo_metriche(tabella)
        assert riepilogo["variable"].to_list() == ["t2m", "tp", "sf"]
        assert riepilogo["metric"][0] == "rmse_celsius"

    def test_il_riepilogo_mette_il_modello_prima_dei_riferimenti(
        self, config: Config
    ) -> None:
        self._scrivi(
            config,
            [
                self._riga(model="persistence", value=4.7),
                self._riga(model="dwf", value=2.9),
                self._riga(model="persistence_diurnal", value=3.2),
            ],
        )
        tabella = metriche(config, 0, "test")
        assert tabella is not None
        assert riepilogo_metriche(tabella)["model"].to_list() == [
            "dwf",
            "persistence_diurnal",
            "persistence",
        ]

    def test_segnala_l_accuratezza_battuta_dal_modello_muto(self, config: Config) -> None:
        """Con frequenza di base 0,09 un modello muto ottiene 0,91."""
        self._scrivi(
            config,
            [
                self._riga(variable="sf", metric="accuracy", value=0.847),
                self._riga(variable="sf", metric="base_rate", value=0.0908),
            ],
        )
        tabella = metriche(config, 0, "test")
        assert tabella is not None
        segnalate = accuratezze_ingannevoli(tabella)
        assert segnalate.height == 1
        assert segnalate["variable"][0] == "sf"
        assert segnalate["sempre_no"][0] == pytest.approx(0.9092)

    def test_non_segnala_un_accuratezza_migliore_del_modello_muto(
        self, config: Config
    ) -> None:
        self._scrivi(
            config,
            [
                self._riga(variable="tp", metric="accuracy", value=0.709),
                self._riga(variable="tp", metric="base_rate", value=0.345),
            ],
        )
        tabella = metriche(config, 0, "test")
        assert tabella is not None
        assert accuratezze_ingannevoli(tabella).height == 0

    def test_senza_frequenza_di_base_non_inventa_segnalazioni(
        self, config: Config
    ) -> None:
        """Mancando il termine di confronto, il silenzio e' l'unica risposta onesta."""
        self._scrivi(config, [self._riga(variable="sf", metric="accuracy", value=0.1)])
        tabella = metriche(config, 0, "test")
        assert tabella is not None
        assert accuratezze_ingannevoli(tabella).height == 0

    def test_l_andamento_per_scadenza_esclude_l_aggregato(self, config: Config) -> None:
        self._scrivi(
            config,
            [
                self._riga(value=99.0),
                self._riga(lead_slot=0, value=1.0),
                self._riga(lead_slot=1, value=2.0),
            ],
        )
        tabella = metriche(config, 0, "test")
        assert tabella is not None
        andamento = per_scadenza(tabella, "t2m", "rmse")
        assert andamento["lead_slot"].to_list() == [0, 1]
        assert 99.0 not in andamento["value"].to_list()


# --------------------------------------------------------------------------- #
# Modello e risorse
# --------------------------------------------------------------------------- #


class TestModelloERisorse:
    def test_legge_i_metadati_del_checkpoint(self, config: Config) -> None:
        import json

        destinazione = config.fold_dir(0)
        destinazione.mkdir(parents=True, exist_ok=True)
        (destinazione / "metadata.json").write_text(
            json.dumps({"n_parameters": 123}), encoding="utf-8"
        )
        (destinazione / "history.json").write_text(
            json.dumps([{"epoch": 0, "train_loss": 2.0, "val_loss": 1.0}]),
            encoding="utf-8",
        )
        info = informazioni_modello(config, 0)
        assert info["disponibile"] is True
        assert info["metadati"]["n_parameters"] == 123
        curva = curva_apprendimento(info)
        assert curva is not None
        assert curva.height == 1

    def test_la_curva_e_assente_senza_storia(self, config: Config) -> None:
        assert curva_apprendimento({"storia": []}) is None

    def test_le_risorse_hanno_valori_sensati(self) -> None:
        stato = risorse()
        assert stato["core"] >= 1
        assert 0.0 <= stato["memoria_percento"] <= 100.0
        assert stato["memoria_totale_gb"] > 0
        assert isinstance(stato["processi"], list)

    def test_il_comando_viene_accorciato(self) -> None:
        lungo = "python.exe " + " ".join(f"--opzione{i}" for i in range(40))
        assert len(_accorcia(lungo)) <= 70

    def test_l_elenco_delle_tecnologie_e_coerente(self) -> None:
        assert len(TECNOLOGIE) >= 8
        for nome, categoria, descrizione in TECNOLOGIE:
            assert nome and categoria and descrizione
            # La descrizione deve dire il ruolo nel progetto, non essere un'etichetta.
            assert len(descrizione) > 40


# --------------------------------------------------------------------------- #
# Coerenza fra artefatti
# --------------------------------------------------------------------------- #


def _scrivi_modello(
    config: Config, fold: int, *, canali: int, epoca: int, epoche_storia: int
) -> Path:
    """Cartella di modello minima, con i soli file che la verifica guarda."""
    import json

    import numpy as np

    destinazione = config.fold_dir(fold)
    destinazione.mkdir(parents=True, exist_ok=True)
    np.savez(destinazione / "weights.npz", peso=np.zeros(3, dtype=np.float32))
    (destinazione / "metadata.json").write_text(
        json.dumps({"in_channels": canali, "epoch": epoca}), encoding="utf-8"
    )
    (destinazione / "history.json").write_text(
        json.dumps([{"epoch": i} for i in range(epoche_storia)]), encoding="utf-8"
    )
    pl.DataFrame({"variable": ["t2m"], "mean": [0.0]}).write_parquet(
        destinazione / "norm_stats.parquet"
    )
    return destinazione


def test_la_coerenza_segnala_i_pesi_mancanti(config: Config) -> None:
    from dwf.dashboard import coerenza_artefatti

    problemi = coerenza_artefatti(config, 0)
    assert len(problemi) == 1
    assert "Pesi assenti" in problemi[0]


def test_la_coerenza_accetta_una_cartella_scritta_in_una_sola_esecuzione(
    config: Config,
) -> None:
    from dwf.dashboard import coerenza_artefatti
    from dwf.data.features import InputLayout

    canali = InputLayout.from_config(config).n_channels
    _scrivi_modello(config, 0, canali=canali, epoca=2, epoche_storia=3)
    assert coerenza_artefatti(config, 0) == []


def test_la_coerenza_rileva_i_canali_incompatibili(config: Config) -> None:
    from dwf.dashboard import coerenza_artefatti

    _scrivi_modello(config, 0, canali=999, epoca=0, epoche_storia=1)
    problemi = coerenza_artefatti(config, 0)
    assert any("999 canali" in p for p in problemi)


def test_la_coerenza_rileva_i_file_scritti_da_un_altra_esecuzione(
    config: Config, tmp_path: Path
) -> None:
    """Il guasto reale: un banco che scrive nella cartella del modello a scala piena.

    I pesi restano quelli buoni, ma statistiche e cronologia vengono da un'altra
    esecuzione. Nulla solleva un errore, e il modello caricato non e' piu' quello che
    si crede di avere. Le date dei file lo dicono.
    """
    import os
    import time

    from dwf.dashboard import coerenza_artefatti
    from dwf.data.features import InputLayout

    canali = InputLayout.from_config(config).n_channels
    destinazione = _scrivi_modello(config, 0, canali=canali, epoca=1, epoche_storia=2)

    adesso = time.time()
    os.utime(destinazione / "weights.npz", (adesso - 3600, adesso - 3600))
    os.utime(destinazione / "history.json", (adesso, adesso))

    problemi = coerenza_artefatti(config, 0)
    assert any("history.json" in p and "altra esecuzione" in p for p in problemi)


def test_la_coerenza_rileva_una_cronologia_piu_corta_dell_epoca_dichiarata(
    config: Config,
) -> None:
    from dwf.dashboard import coerenza_artefatti
    from dwf.data.features import InputLayout

    canali = InputLayout.from_config(config).n_channels
    _scrivi_modello(config, 0, canali=canali, epoca=16, epoche_storia=3)
    problemi = coerenza_artefatti(config, 0)
    assert any("epoca 16" in p and "3" in p for p in problemi)


def test_la_cache_cambia_nome_quando_i_pesi_cambiano(config: Config) -> None:
    """Senza legame con i pesi, la cache mostrerebbe la mappa di un modello superato."""
    import os
    import time

    from dwf.dashboard import _percorso_cache

    assert _percorso_cache(config, 0, "mappa") is None

    _scrivi_modello(config, 0, canali=1, epoca=0, epoche_storia=1)
    primo = _percorso_cache(config, 0, "mappa")
    assert primo is not None

    pesi = config.fold_dir(0) / "weights.npz"
    dopo = time.time() + 120
    os.utime(pesi, (dopo, dopo))
    assert _percorso_cache(config, 0, "mappa") != primo


def test_il_guadagno_sulla_persistenza_usa_il_riferimento_piu_forte() -> None:
    """Confrontarsi con la persistenza peggiore renderebbe il guadagno adulatorio."""
    from dwf.dashboard import guadagno_su_persistenza

    tabella = pl.DataFrame(
        {
            "model": ["dwf", "persistence", "persistence_diurnal"],
            "variable": ["t2m"] * 3,
            "metric": ["rmse_celsius"] * 3,
            "month": [-1, -1, -1],
            "lead_slot": [2, 2, 2],
            "value": [2.0, 4.0, 2.5],
        }
    )

    esito = guadagno_su_persistenza(tabella, "t2m", "rmse_celsius")

    assert esito.height == 1
    riga = esito.row(0, named=True)
    assert riga["riferimento"] == 2.5
    assert riga["guadagno_percento"] == pytest.approx(20.0)


def test_il_guadagno_e_negativo_se_il_modello_perde() -> None:
    from dwf.dashboard import guadagno_su_persistenza

    tabella = pl.DataFrame(
        {
            "model": ["dwf", "persistence_diurnal"],
            "variable": ["t2m"] * 2,
            "metric": ["rmse_celsius"] * 2,
            "month": [-1, -1],
            "lead_slot": [0, 0],
            "value": [3.0, 2.0],
        }
    )

    esito = guadagno_su_persistenza(tabella, "t2m", "rmse_celsius")

    assert esito.row(0, named=True)["guadagno_percento"] == pytest.approx(-50.0)


def test_il_guadagno_su_una_tabella_vuota_non_rompe_la_pagina() -> None:
    from dwf.dashboard import guadagno_su_persistenza

    vuota = pl.DataFrame(
        schema={
            "model": pl.Utf8,
            "variable": pl.Utf8,
            "metric": pl.Utf8,
            "month": pl.Int16,
            "lead_slot": pl.Int16,
            "value": pl.Float64,
        }
    )

    esito = guadagno_su_persistenza(vuota, "t2m", "rmse_celsius")

    assert esito.height == 0
    assert "guadagno_percento" in esito.columns
