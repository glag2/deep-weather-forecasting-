"""Il testo tecnico del report deve stare nella pagina e citare numeri veri.

Un documento generato non ha nessuno che lo rilegge: se una riga esce dal foglio o una
tabella riporta valori scritti a mano invece di quelli dell'ultima valutazione, nessuno se
ne accorge finche' non lo legge un committente.
"""

from __future__ import annotations

import polars as pl
import pytest

from dwf.documentation import (
    PREFISSO_VERBATIM,
    SEZIONI_TECNICHE,
    DocSection,
    measured_section,
    skill_lines,
)
from dwf.report import DOC_LINE_WIDTH, paginate, wrap_section


def _metriche(modello: dict[int, float], persistenza: dict[int, float]) -> pl.DataFrame:
    righe = []
    for nome, valori in (("dwf", modello), ("persistence_diurnal", persistenza)):
        for scadenza, valore in valori.items():
            righe.append(
                {
                    "model": nome, "split": "test", "fold": 0, "variable": "t2m",
                    "lead_slot": scadenza, "month": -1, "metric": "rmse_celsius",
                    "value": valore, "n_values": 100,
                }
            )
    return pl.DataFrame(righe)


class TestImpaginazione:
    def test_i_paragrafi_vanno_a_capo_entro_la_larghezza(self) -> None:
        sezione = DocSection("prova", ("parola " * 200,))

        righe = wrap_section(sezione)

        assert len(righe) > 1
        assert max(len(riga) for riga in righe) <= DOC_LINE_WIDTH

    def test_un_blocco_a_larghezza_fissa_non_viene_riformattato(self) -> None:
        """Una tabella riformattata non e' piu' una tabella: l'allineamento e' informazione."""
        tabella = f"{PREFISSO_VERBATIM}a    b    c\n{PREFISSO_VERBATIM}1    2    3"
        sezione = DocSection("prova", (tabella,))

        assert wrap_section(sezione) == ["a    b    c", "1    2    3"]

    def test_ogni_sezione_del_report_sta_nella_larghezza_della_pagina(self) -> None:
        """Il vero controllo: matplotlib non manda a capo e non segnala il testo fuori foglio."""
        troppo_lunghe = [
            (sezione.title, riga)
            for sezione in SEZIONI_TECNICHE
            for riga in wrap_section(sezione)
            if len(riga) > DOC_LINE_WIDTH
        ]

        assert not troppo_lunghe

    def test_le_pagine_non_perdono_righe(self) -> None:
        righe = [str(numero) for numero in range(100)]

        pagine = paginate(righe, lines_per_page=42)

        assert [riga for pagina in pagine for riga in pagina] == righe
        assert len(pagine) == 3

    def test_una_sezione_senza_corpo_e_un_errore(self) -> None:
        with pytest.raises(ValueError, match="vuota"):
            DocSection("titolo", ())


class TestNumeriMisurati:
    def test_il_guadagno_e_calcolato_dai_numeri_della_tabella(self) -> None:
        tabella = _metriche({0: 1.8, 1: 2.4}, {0: 2.4, 1: 2.4})

        righe = skill_lines(tabella)

        assert "25.0%" in righe[2]
        assert "0.0%" in righe[3]

    def test_senza_metriche_la_pagina_lo_dichiara(self) -> None:
        """Meglio una pagina che ammette di non avere la misura di una che ne mostra un'altra."""
        sezione = measured_section(None)

        assert "eseguire scripts/evaluate_model.py" in " ".join(sezione.body)

    def test_una_tabella_senza_persistenza_non_inventa_un_confronto(self) -> None:
        solo_modello = _metriche({0: 1.8}, {}).filter(pl.col("model") == "dwf")

        assert "confronto impossibile" in skill_lines(solo_modello)[0]

    def test_una_tabella_senza_le_colonne_attese_si_ferma(self) -> None:
        with pytest.raises(ValueError, match="incompleta"):
            skill_lines(pl.DataFrame({"model": ["dwf"]}))
