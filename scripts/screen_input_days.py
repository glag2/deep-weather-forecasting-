"""Quanti giorni di storico servono davvero in ingresso.

Il confronto non puo' essere fatto sul solo errore assoluto: cambiare la lunghezza
della finestra cambia **quali** finestre esistono, perche' una storia piu' lunga esclude
l'inizio del periodo. Ogni impostazione viene quindi misurata contro la persistenza
diurna calcolata **sulle sue stesse finestre**, e si confronta il guadagno rispetto a
quel riferimento, non l'errore grezzo.

Ogni impostazione e' ripetuta con piu' semi, perche' il rumore fra ripetizioni misurato
in `VARIANTS.md` (0,059 degC) e' dello stesso ordine delle differenze attese.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:  # pragma: no cover - avvio da riga di comando
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dwf.config import Config  # noqa: E402
from dwf.train import train_fold  # noqa: E402


def _carica_banco():
    """Riusa le funzioni di misura del banco invece di riscriverle."""
    percorso = Path(__file__).resolve().parent / "compare_variants.py"
    spec = importlib.util.spec_from_file_location("banco_varianti", percorso)
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    # Va registrato prima di eseguirlo: @dataclass risale a sys.modules per risolvere le
    # annotazioni, e su un modulo non registrato fallisce con un AttributeError opaco.
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


BANCO = _carica_banco()


@dataclass
class Esito:
    giorni: int
    slot: int
    canali: int
    riferimento: float
    rmse: list[float] = field(default_factory=list)
    secondi: list[float] = field(default_factory=list)
    finestre_train: int = 0
    ritardi: tuple[int, ...] = ()
    errore: str = ""

    @property
    def media(self) -> float:
        return statistics.fmean(self.rmse) if self.rmse else float("nan")

    @property
    def guadagno(self) -> float:
        """Quanto si guadagna sul riferimento della *propria* impostazione."""
        return self.riferimento - self.media


def ritardi_ammessi(base: Config, slot: int) -> tuple[int, ...]:
    """Tendenze che entrano nella finestra.

    Con 3 giorni di storico la tendenza a 3 giorni non esiste: non e' una scelta ma un
    vincolo strutturale della finestra corta, e va dichiarato invece di aggirato.
    """
    return tuple(lag for lag in base.features.tendency_lags if lag < slot)


def configura(base: Config, giorni: int, seme: int, args) -> Config:
    slot = giorni * base.time.slots_per_day
    finestre = base.windows.model_copy(update={"input_slots": slot})
    caratteristiche = base.features.model_copy(
        update={"tendency_lags": list(ritardi_ammessi(base, slot))}
    )
    allenamento = base.training.model_copy(
        update={
            "epochs": args.epochs,
            "crop_size": args.crop,
            "samples_per_epoch": args.samples,
            "seed": seme,
        }
    )
    percorsi = base.paths.model_copy(
        update={"artifacts_subdir": f"bench/giorni{giorni:02d}_seed{seme}"}
    )
    return base.model_copy(
        update={
            "windows": finestre,
            "features": caratteristiche,
            "training": allenamento,
            "paths": percorsi,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--days", type=int, nargs="+", default=[3, 7, 10, 14])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 101])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--crop", type=int, default=64)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--eval-windows", type=int, default=40)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "INPUT_DAYS.md")
    args = parser.parse_args()

    from dwf.data.dataset import sample_starts
    from dwf.data.features import InputLayout

    base = Config.load(args.config)
    esiti: list[Esito] = []

    for giorni in args.days:
        prima = configura(base, giorni, args.seeds[0], args)
        canali = InputLayout.from_config(prima).n_channels
        try:
            riferimento = BANCO.riferimento_diurno(prima, args.fold, args.eval_windows)
            finestre_train = len(sample_starts(prima, args.fold, "train"))
        except Exception as errore:
            traceback.print_exc()
            esiti.append(Esito(giorni, giorni * base.time.slots_per_day, canali,
                               float("nan"), errore=str(errore)[:100]))
            continue

        esito = Esito(giorni, giorni * base.time.slots_per_day, canali,
                      riferimento, finestre_train=finestre_train)
        esito.ritardi = ritardi_ammessi(base, esito.slot)
        print(f"\n[{giorni} giorni] {esito.slot} slot, {canali} canali, "
              f"tendenze {list(esito.ritardi)}, "
              f"{finestre_train} finestre di train, riferimento {riferimento:.3f}")

        for seme in args.seeds:
            config = configura(base, giorni, seme, args)
            avvio = time.perf_counter()
            try:
                risultato = train_fold(config, args.fold, verbose=False)
                durata = (time.perf_counter() - avvio) / max(1, len(risultato.history))
                rmse = BANCO.rmse_su_validazione(config, args.fold, args.eval_windows)
            except Exception as errore:
                traceback.print_exc()
                esito.errore = str(errore)[:100]
                break
            esito.rmse.append(rmse)
            esito.secondi.append(durata)
            print(f"    seme {seme}: RMSE {rmse:.3f} degC, {durata:.0f} s/passata")

        esiti.append(esito)

    scrivi(esiti, args)
    print(f"\nrelazione scritta in {args.output}")


def _dispersione_fra_semi(esiti: list[Esito]) -> float:
    """Dispersione fra ripetizioni, messa in comune fra le impostazioni.

    Ogni impostazione ha pochi semi, e la sua dispersione presa da sola sarebbe una
    stima troppo instabile per decidere qualcosa. Poiche' il rumore fra ripetizioni non
    dipende dalla lunghezza della finestra, le varianze si possono mettere in comune:
    si ottiene una stima con molti piu' gradi di liberta' a partire dagli stessi dati.
    """
    varianze = [statistics.variance(e.rmse) for e in esiti if len(e.rmse) > 1]
    if not varianze:
        return float("nan")
    return math.sqrt(statistics.fmean(varianze))


def scrivi(esiti: list[Esito], args) -> None:
    validi = [e for e in esiti if e.rmse]
    righe = [
        "# Quanti giorni di storico in ingresso",
        "",
        "Generato da `scripts/screen_input_days.py`.",
        "",
        "## Perche' non basta confrontare l'errore",
        "",
        "Allungare la finestra di ingresso cambia **quali** finestre esistono: una storia",
        "di 14 giorni scarta l'inizio del periodo, quindi il blocco di validazione non e'",
        "piu' lo stesso e gli errori grezzi non sono confrontabili.",
        "",
        "Ogni impostazione e' percio' misurata contro la persistenza diurna calcolata",
        "**sulle sue stesse finestre**, e si confronta il **guadagno** su quel",
        "riferimento. Ogni impostazione e' ripetuta con piu' semi, perche' il rumore fra",
        "ripetizioni e' dello stesso ordine delle differenze attese: il banco delle",
        "varianti lo misura in 0,059 degC, e qui viene stimato di nuovo sui propri dati",
        "invece di essere dato per noto.",
        "",
        "## Protocollo",
        "",
        f"Fold {args.fold}, {args.epochs} passate, ritaglio {args.crop}, "
        f"{args.samples} campioni per passata, semi {args.seeds}.",
        "",
        "## Risultati",
        "",
        "| giorni | slot | canali | tendenze | finestre | riferimento | RMSE | guadagno | s/ep |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for e in esiti:
        if not e.rmse:
            righe.append(
                f"| {e.giorni} | {e.slot} | {e.canali} | {list(e.ritardi)} | "
                f"{e.finestre_train} | - | fallita | - | - |"
            )
            continue
        secondi = statistics.fmean(e.secondi)
        righe.append(
            f"| {e.giorni} | {e.slot} | {e.canali} | {list(e.ritardi)} | "
            f"{e.finestre_train} | {e.riferimento:.3f} | {e.media:.3f} | "
            f"**{e.guadagno:+.3f}** | {secondi:.0f} |"
        )

    if validi:
        migliore = max(validi, key=lambda e: e.guadagno)
        ordinati = sorted(validi, key=lambda e: e.guadagno, reverse=True)
        margine = (
            ordinati[0].guadagno - ordinati[1].guadagno if len(ordinati) > 1 else float("nan")
        )
        righe += [
            "",
            "## Lettura",
            "",
            f"Guadagno maggiore: **{migliore.giorni} giorni** ({migliore.guadagno:+.3f} degC "
            f"sul proprio riferimento).",
            "",
        ]
        # Il margine e' una differenza fra **medie** di piu' semi, quindi non va
        # confrontato con la dispersione di una singola misura: quella sovrastima
        # l'incertezza della media e farebbe dichiarare reale meta' delle differenze
        # dovute al caso. Il metro giusto e' l'errore standard della differenza,
        # ricavato dalla dispersione fra semi misurata in questo stesso banco.
        dispersione = _dispersione_fra_semi(validi)
        soglia = float("nan")
        if dispersione == dispersione and len(ordinati) > 1:
            campioni_primo = max(len(ordinati[0].rmse), 1)
            campioni_secondo = max(len(ordinati[1].rmse), 1)
            soglia = 2.0 * dispersione * math.sqrt(1 / campioni_primo + 1 / campioni_secondo)

        if margine != margine or soglia != soglia:
            righe.append("Semi insufficienti per stimare l'incertezza del confronto.")
        elif margine < soglia:
            # Fra impostazioni che il banco non riesce a separare si sceglie la piu'
            # economica, non la piu' corta in assoluto: la piu' corta puo' benissimo
            # essere una di quelle nettamente peggiori.
            equivalenti = [
                e for e in validi if migliore.guadagno - e.guadagno < soglia
            ]
            scelta = min(equivalenti, key=lambda e: e.canali)
            elenco = ", ".join(
                f"{e.giorni} giorni" for e in sorted(equivalenti, key=lambda e: e.giorni)
            )
            # Con tutte le impostazioni equivalenti non esiste una categoria di escluse:
            # dirlo lo stesso significherebbe dichiarare peggiore un insieme vuoto.
            escluse = [e for e in validi if e not in equivalenti]
            coda = (
                "finestre utilizzabili. Le altre impostazioni restano fuori: non sono "
                "equivalenti, sono misurabilmente peggiori."
                if escluse
                else "finestre utilizzabili. Nessuna impostazione risulta esclusa: il banco "
                "non separa nessuna delle lunghezze provate, quindi qui non si sta "
                "scegliendo la migliore, si sta scegliendo la meno cara fra pari."
            )
            righe += [
                f"Il margine sul secondo classificato e' {margine:.3f} degC, **inferiore** ai "
                f"{soglia:.3f} degC di incertezza a due deviazioni standard (dispersione fra "
                f"semi {dispersione:.3f} degC, misurata qui): la differenza non e'",
                "distinguibile dal caso a questo budget.",
                "",
                f"Impostazioni non separabili dalla migliore: {elenco}. Fra queste si sceglie "
                f"la piu' economica, cioe' **{scelta.giorni} giorni** ({scelta.canali} canali), "
                "perche' a parita' di risultato misurabile costa meno memoria e lascia piu'",
                coda,
            ]
        else:
            righe += [
                f"Il margine sul secondo classificato e' {margine:.3f} degC, **superiore** ai "
                f"{soglia:.3f} degC di incertezza a due deviazioni standard (dispersione fra "
                f"semi {dispersione:.3f} degC): la differenza sopravvive al rumore.",
            ]
        righe += [
            "",
            "## Limiti",
            "",
            "1. Protocollo ridotto e un solo fold, come per il confronto fra varianti.",
            "2. Il numero di canali cresce linearmente con gli slot di ingresso: una",
            "   finestra lunga non e' solo piu' informativa, e' anche piu' difficile da",
            "   addestrare a parita' di campioni.",
            "3. Una finestra piu' lunga riduce le finestre disponibili, quindi in parte si",
            "   sta misurando anche la perdita di dati di addestramento.",
        ]
    args.output.write_text("\n".join(righe) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
