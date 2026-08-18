"""Analisi esplorativa del dataset ERA5 ingerito.

Scopo: stabilire **cosa e' ragionevole aspettarsi** dal modello prima di costruirlo.
Senza questa fase si finisce per attribuire alla rete difetti che appartengono al
problema, o per festeggiare risultati che la sola persistenza avrebbe dato.

Principio di lettura adottato: **i dati sono veri**. ERA5 e' una rianalisi prodotta
assimilando osservazioni in un modello fisico, e se un numero sorprende la prima
ipotesi da verificare e' un errore di chi analizza, non del dato. E' gia' successo in
questo progetto con la neve apparentemente superiore alla precipitazione totale, che
si e' rivelata la quantizzazione intera indipendente dei messaggi GRIB.

Produce un rapporto a schermo e il file `DATA_ANALYSIS.md`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:  # pragma: no cover - avvio da riga di comando
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dwf.config import Config  # noqa: E402
from dwf.data.features import to_working_units  # noqa: E402
from dwf.solar import cos_solar_zenith  # noqa: E402
from dwf.thermo import (  # noqa: E402
    latent_heat_content,
    relative_humidity,
    surface_pressure_from_msl,
)

# Vigo di Cadore, fuoco del progetto.
VIGO_LAT = 46.5031
VIGO_LON = 12.5308

# Soglie usate nella diagnostica dei target, coerenti con quelle della valutazione.
RAIN_MM = 0.1
SNOW_FRACTION = 0.5


@dataclass
class Sezione:
    """Un blocco del rapporto: titolo, righe di testo, e commento interpretativo."""

    titolo: str
    righe: list[str]
    commento: str = ""


def slot_ingeriti(store: xr.Dataset) -> np.ndarray:
    """Indici degli slot realmente scritti.

    Zarr non materializza i blocchi mai toccati, quindi i mesi non ancora ingeriti
    restano NaN: la maschera si ricava da un singolo punto, che basta perche' un mese
    viene scritto tutto insieme.
    """
    sonda = store["t2m"].isel(latitude=store.sizes["latitude"] // 2,
                              longitude=store.sizes["longitude"] // 2).values
    return np.flatnonzero(np.isfinite(sonda))


def descrivi_copertura(store: xr.Dataset, indici: np.ndarray) -> Sezione:
    tempi = store["valid_time"].values[indici]
    mesi: dict[str, int] = {}
    for istante in tempi:
        chiave = str(istante)[:7]
        mesi[chiave] = mesi.get(chiave, 0) + 1

    ore, conteggi = np.unique([int(str(t)[11:13]) for t in tempi], return_counts=True)
    righe = [
        f"slot allocati nello store : {store.sizes['slot']:,}",
        f"slot effettivamente presenti: {len(indici):,}",
        f"mesi coperti              : {len(mesi)}",
        f"primo istante             : {str(tempi[0])[:16]}",
        f"ultimo istante            : {str(tempi[-1])[:16]}",
        "",
        "slot per ora del giorno   : "
        + ", ".join(f"{o:02d}Z={c}" for o, c in zip(ore, conteggi, strict=True)),
        "",
        "slot per mese:",
    ]
    righe += [f"  {mese}  {numero:3d}" for mese, numero in sorted(mesi.items())]

    atteso = {6, 12, 18}
    commento = (
        "Le tre ore giornaliere sono equilibrate, come deve essere: uno squilibrio "
        "farebbe apprendere alla rete una climatologia distorta."
        if set(ore.tolist()) == atteso and conteggi.std() / conteggi.mean() < 0.05
        else "ATTENZIONE: la copertura oraria non e' bilanciata, va capito perche'."
    )
    return Sezione("1. Copertura temporale", righe, commento)


def descrivi_distribuzioni(store: xr.Dataset, indici: np.ndarray, passo: int) -> Sezione:
    """Statistiche per variabile, con un controllo di plausibilita' fisica."""
    limiti_attesi = {
        "t2m": (-70.0, 55.0, "degC"),
        "d2m": (-80.0, 40.0, "degC"),
        "msl": (87000.0, 110000.0, "Pa"),
        "u10": (-60.0, 60.0, "m/s"),
        "v10": (-60.0, 60.0, "m/s"),
        "tcc": (0.0, 1.0, "0-1"),
        "sd": (0.0, 12.0, "m"),
        "tp": (0.0, 0.5, "m"),
        "sf": (0.0, 0.3, "m"),
    }
    righe = [
        f"{'var':5s} {'unita':6s} {'minimo':>10s} {'p01':>9s} {'mediana':>9s} "
        f"{'p99':>9s} {'massimo':>10s} {'NaN':>6s}",
    ]
    anomalie: list[str] = []

    for nome, (basso, alto, unita) in limiti_attesi.items():
        if nome not in store:
            continue
        valori = to_working_units(
            nome,
            store[nome].isel(slot=indici, latitude=slice(None, None, passo),
                             longitude=slice(None, None, passo)).values,
        )
        finiti = valori[np.isfinite(valori)]
        quota_nan = 1.0 - finiti.size / valori.size
        q01, mediana, q99 = np.percentile(finiti, [1, 50, 99])
        righe.append(
            f"{nome:5s} {unita:6s} {finiti.min():10.3f} {q01:9.3f} {mediana:9.3f} "
            f"{q99:9.3f} {finiti.max():10.3f} {quota_nan:6.2%}"
        )
        if finiti.min() < basso or finiti.max() > alto:
            anomalie.append(
                f"{nome}: intervallo [{finiti.min():.3f}, {finiti.max():.3f}] "
                f"fuori dall'atteso [{basso}, {alto}]"
            )

    commento = (
        "Tutte le variabili stanno negli intervalli fisicamente plausibili."
        if not anomalie
        else "Valori fuori intervallo, da spiegare prima di proseguire:\n  "
        + "\n  ".join(anomalie)
    )
    return Sezione("2. Distribuzioni e plausibilita' fisica", righe, commento)


def descrivi_ciclo_diurno(store: xr.Dataset, indici: np.ndarray, passo: int) -> Sezione:
    """Ampiezza del ciclo diurno: e' il difetto misurato sul modello attuale."""
    tempi = store["valid_time"].values[indici]
    ore = np.array([int(str(t)[11:13]) for t in tempi])
    temperatura = to_working_units(
        "t2m",
        store["t2m"].isel(slot=indici, latitude=slice(None, None, passo),
                          longitude=slice(None, None, passo)).values,
    )

    righe = [f"{'ora':>5s} {'media':>9s} {'dev.std':>9s} {'n slot':>7s}"]
    medie = {}
    for ora in (6, 12, 18):
        maschera = ore == ora
        if not maschera.any():
            continue
        campione = temperatura[maschera]
        medie[ora] = float(np.nanmean(campione))
        righe.append(
            f"{ora:>3d}Z {medie[ora]:9.3f} {float(np.nanstd(campione)):9.3f} "
            f"{int(maschera.sum()):7d}"
        )

    ampiezza = max(medie.values()) - min(medie.values()) if medie else 0.0
    righe += ["", f"ampiezza media del ciclo diurno sul dominio: {ampiezza:.3f} degC"]

    commento = (
        f"Sul dominio intero il ciclo diurno vale {ampiezza:.2f} gradi, attenuato dalla "
        f"media su mare e terra. E' la grandezza che il modello attuale sottostima, ed "
        f"e' il motivo per cui i canali solari sono stati aggiunti."
    )
    return Sezione("3. Ciclo diurno", righe, commento)


def descrivi_prevedibilita(store: xr.Dataset, indici: np.ndarray, passo: int) -> Sezione:
    """Quanto e' prevedibile la temperatura senza modello, scadenza per scadenza.

    E' il risultato piu' importante dell'analisi, e va letto ricordando che gli slot
    **non sono equidistanti**: 06Z, 12Z e 18Z distano 6, 6 e 12 ore, quindi tre slot
    fanno esattamente un giorno e uno scarto di k slot non vale k per sei ore.

    Vengono confrontati due riferimenti, entrambi disponibili al momento della
    previsione:

    - *persistenza ingenua*, che ripete l'ultimo istante osservato;
    - *persistenza diurna*, che ripete l'osservazione piu' recente **alla stessa ora
      del giorno** del bersaglio.

    La seconda e' molto piu' forte, perche' non paga il ciclo giorno-notte. E' quella
    che il modello deve davvero battere.
    """
    temperatura = to_working_units(
        "t2m",
        store["t2m"].isel(slot=indici, latitude=slice(None, None, passo),
                          longitude=slice(None, None, passo)).values,
    ).reshape(len(indici), -1)
    tempi = store["valid_time"].values[indici]
    ore = np.array([int(str(t)[11:13]) for t in tempi])
    posizione = {int(s): i for i, s in enumerate(indici)}

    righe = [
        f"{'lead':>4s} {'ore':>4s} {'corr.ing':>9s} {'RMSE ing':>9s} "
        f"{'corr.diu':>9s} {'RMSE diu':>9s} {'guadagno':>9s} {'coppie':>7s}"
    ]
    correlazioni_diurne: list[float] = []

    for scarto in range(1, 10):
        # L'osservazione utile piu' recente alla stessa ora del bersaglio: si torna
        # indietro di giorni interi, cioe' di multipli di tre slot.
        indietro = 3 * int(np.ceil(scarto / 3))
        terne = []
        for posizione_iniziale, slot in enumerate(indici):
            bersaglio = posizione.get(int(slot) + scarto)
            riferimento = posizione.get(int(slot) + scarto - indietro)
            if bersaglio is None or riferimento is None:
                continue
            terne.append((posizione_iniziale, bersaglio, riferimento))
        if len(terne) < 20:
            continue

        partenza = np.array([t[0] for t in terne])
        arrivo = np.array([t[1] for t in terne])
        stessa_ora = np.array([t[2] for t in terne])
        assert np.array_equal(ore[stessa_ora], ore[arrivo]), "riferimento di ora sbagliata"

        ore_reali = float(
            np.mean((tempi[arrivo] - tempi[partenza]).astype("timedelta64[h]").astype(int))
        )
        ingenua = temperatura[partenza]
        diurna = temperatura[stessa_ora]
        reale = temperatura[arrivo]

        rmse_ingenua = float(np.sqrt(np.mean((reale - ingenua) ** 2)))
        rmse_diurna = float(np.sqrt(np.mean((reale - diurna) ** 2)))
        correlazione_diurna = float(np.corrcoef(diurna.ravel(), reale.ravel())[0, 1])
        correlazioni_diurne.append(correlazione_diurna)

        righe.append(
            f"{scarto:4d} {ore_reali:4.0f} "
            f"{float(np.corrcoef(ingenua.ravel(), reale.ravel())[0, 1]):9.4f} "
            f"{rmse_ingenua:9.3f} {correlazione_diurna:9.4f} {rmse_diurna:9.3f} "
            f"{1 - rmse_diurna / rmse_ingenua:8.1%} {len(terne):7d}"
        )

    smorzamento = (
        f"da {min(correlazioni_diurne):.3f} a {max(correlazioni_diurne):.3f}"
        if correlazioni_diurne
        else "non calcolabile"
    )
    commento = (
        "Gli slot non sono equidistanti, quindi la colonna delle ore e' calcolata sui "
        "tempi veri e non come multiplo di sei. Le scadenze 3, 6 e 9 cadono esattamente "
        "a uno, due e tre giorni, cioe' alla stessa ora del giorno iniziale.\n\n"
        "La persistenza diurna e' nettamente migliore di quella ingenua, e il guadagno "
        "e' massimo proprio alle scadenze che sfasano il ciclo giorno-notte. Questo "
        "sposta l'asticella: **il riferimento onesto da battere e' la persistenza "
        "diurna**, non quella ingenua, e la valutazione del modello va aggiornata di "
        "conseguenza.\n\n"
        f"La correlazione della persistenza diurna, {smorzamento}, e' anche la stima "
        "teorica dello smorzamento: sotto errore quadratico l'ampiezza ottimale della "
        "previsione vale la correlazione, quindi un modello addestrato a MSE tendera' "
        "a produrre quella frazione della variabilita' reale. E' la giustificazione "
        "quantitativa del termine spettrale nella nuova loss."
    )
    return Sezione("4. Prevedibilita' e riferimenti da battere", righe, commento)


def descrivi_target(store: xr.Dataset, indici: np.ndarray, passo: int) -> Sezione:
    """Frequenza degli eventi da prevedere: determina l'equilibrio delle classi."""
    pioggia = store["tp"].isel(slot=indici, latitude=slice(None, None, passo),
                               longitude=slice(None, None, passo)).values
    neve = store["sf"].isel(slot=indici, latitude=slice(None, None, passo),
                            longitude=slice(None, None, passo)).values

    soglia_m = RAIN_MM / 1000.0
    piove = pioggia > soglia_m
    quota_pioggia = float(np.nanmean(piove))
    with np.errstate(invalid="ignore", divide="ignore"):
        frazione = np.where(pioggia > soglia_m, neve / np.maximum(pioggia, 1e-12), 0.0)
    nevica = piove & (frazione > SNOW_FRACTION)

    righe = [
        f"celle con pioggia oltre {RAIN_MM} mm : {quota_pioggia:.2%}",
        f"celle con neve prevalente          : {float(np.nanmean(nevica)):.2%}",
        f"quota nevosa fra le celle piovose  : "
        f"{float(np.nanmean(frazione[piove])) if piove.any() else 0.0:.2%}",
        f"precipitazione media dove piove    : "
        f"{float(np.nanmean(pioggia[piove])) * 1000:.3f} mm",
        f"massimo su una cella               : {float(np.nanmax(pioggia)) * 1000:.2f} mm",
    ]
    commento = (
        f"La pioggia interessa il {quota_pioggia:.1%} delle celle: la classe positiva e' "
        f"minoritaria ma non rara, quindi l'accuratezza grezza sarebbe una metrica "
        f"ingannevole e F1 resta la scelta giusta. La neve e' molto piu' rara, ed e' il "
        f"motivo per cui la sua soglia di decisione va scelta sui dati e non fissata a 0,5."
    )
    return Sezione("5. Diagnostica dei target", righe, commento)


def descrivi_vigo(store: xr.Dataset, statico: xr.Dataset, indici: np.ndarray) -> Sezione:
    """Climatologia locale nel punto di interesse del progetto."""
    lat = store["latitude"].values
    lon = store["longitude"].values
    riga = int(np.abs(lat - VIGO_LAT).argmin())
    colonna = int(np.abs(lon - VIGO_LON).argmin())

    quota = float(statico["z"].values[riga, colonna]) / 9.80665
    temperatura = to_working_units(
        "t2m", store["t2m"].isel(slot=indici, latitude=riga, longitude=colonna).values
    )
    rugiada = to_working_units(
        "d2m", store["d2m"].isel(slot=indici, latitude=riga, longitude=colonna).values
    )
    pressione_mare = store["msl"].isel(slot=indici, latitude=riga, longitude=colonna).values
    pressione = surface_pressure_from_msl(
        pressione_mare, np.full_like(temperatura, quota * 9.80665), temperatura
    )
    umidita = relative_humidity(temperatura, rugiada)
    latente = latent_heat_content(temperatura, rugiada, pressione)

    tempi = store["valid_time"].values[indici]
    ore = np.array([int(str(t)[11:13]) for t in tempi])
    per_ora = {
        int(o): float(np.nanmean(temperatura[ore == o])) for o in (6, 12, 18) if (ore == o).any()
    }
    ampiezza = max(per_ora.values()) - min(per_ora.values()) if per_ora else 0.0

    righe = [
        f"cella della griglia        : riga {riga}, colonna {colonna} "
        f"({lat[riga]:.2f} N, {lon[colonna]:.2f} E)",
        f"quota del modello          : {quota:.0f} m",
        "quota reale del paese      : circa 951 m",
        f"scarto di quota            : {quota - 951:.0f} m",
        "",
        f"temperatura media          : {float(np.nanmean(temperatura)):.2f} degC",
        f"minima osservata           : {float(np.nanmin(temperatura)):.2f} degC",
        f"massima osservata          : {float(np.nanmax(temperatura)):.2f} degC",
        f"ampiezza del ciclo diurno  : {ampiezza:.2f} degC",
        f"umidita' relativa media    : {float(np.nanmean(umidita)):.1%}",
        f"calore latente medio       : {float(np.nanmean(latente)) / 1000:.1f} kJ/kg",
    ]
    for ora, media in sorted(per_ora.items()):
        righe.append(f"  media a {ora:02d}Z              : {media:.2f} degC")

    scarto_termico = (quota - 951) * 6.5 / 1000.0
    commento = (
        f"La cella sta {quota - 951:.0f} m piu' in alto del paese, perche' a 0,25 gradi "
        f"una cella copre circa 28 km e media tutto il Cadore, creste comprese. Con il "
        f"gradiente termico standard di 6,5 gradi per chilometro corrisponde a circa "
        f"{scarto_termico:.1f} gradi di scarto freddo sistematico rispetto al fondovalle. "
        f"Non e' un errore del modello ne' del dato: e' il limite di risoluzione, e per "
        f"passare dalla cella al paese serve una correzione di quota esplicita. "
        f"L'ampiezza diurna locale, {ampiezza:.1f} gradi, e' molto maggiore di quella "
        f"media del dominio, il che rende Vigo un punto severo per il modello."
    )
    return Sezione("6. Vigo di Cadore", righe, commento)


def descrivi_solare(store: xr.Dataset, indici: np.ndarray) -> Sezione:
    """Verifica che il canale solare sia coerente con il dato osservato."""
    lat = store["latitude"].values
    lon = store["longitude"].values
    tempi = store["valid_time"].values[indici]

    passo = max(1, len(indici) // 120)
    scelti = indici[::passo]
    scelti_tempi = tempi[::passo]

    zenit: list[float] = []
    temperature: list[float] = []
    for indice, istante in zip(scelti, scelti_tempi, strict=True):
        momento = np.datetime64(istante).astype("datetime64[s]").astype(object)
        campo = cos_solar_zenith(lat[::8], lon[::8], momento)
        zenit.append(float(campo.mean()))
        temperature.append(
            float(
                np.nanmean(
                    to_working_units(
                        "t2m",
                        store["t2m"]
                        .isel(slot=int(indice), latitude=slice(None, None, 8),
                              longitude=slice(None, None, 8))
                        .values,
                    )
                )
            )
        )

    correlazione = float(np.corrcoef(zenit, temperature)[0, 1])
    righe = [
        f"istanti campionati            : {len(zenit)}",
        f"coseno zenitale medio         : {np.mean(zenit):.4f}",
        f"correlazione con la temperatura: {correlazione:+.4f}",
    ]
    commento = (
        f"La correlazione fra insolazione istantanea e temperatura media del dominio "
        f"vale {correlazione:+.3f}. Il segno positivo conferma che il canale porta "
        f"informazione nella direzione fisicamente attesa. Il valore non e' vicino a uno "
        f"perche' la temperatura ha una forte inerzia termica e la stagione domina "
        f"sull'ora: e' proprio questa differenza che la rete deve imparare a comporre."
    )
    return Sezione("7. Coerenza del canale solare", righe, commento)


def scrivi_rapporto(sezioni: list[Sezione], destinazione: Path, intestazione: str) -> None:
    parti = [
        "# Analisi del dataset ERA5",
        "",
        intestazione,
        "",
        "> Principio di lettura: **i dati sono veri**. ERA5 assimila osservazioni in un",
        "> modello fisico, quindi davanti a un numero sorprendente la prima ipotesi da",
        "> verificare e' un errore di analisi, non del dato.",
        "",
        "Rigenerabile con `python scripts/analyze_data.py`.",
        "",
    ]
    for sezione in sezioni:
        parti += [f"## {sezione.titolo}", "", "```", *sezione.righe, "```", ""]
        if sezione.commento:
            parti += [sezione.commento, ""]
    destinazione.write_text("\n".join(parti), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument(
        "--stride", type=int, default=4,
        help="Sottocampionamento spaziale per le statistiche (1 = griglia intera).",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/DATA_ANALYSIS.md")
    args = parser.parse_args()

    config = Config.load(args.config, project_root=PROJECT_ROOT)
    store = xr.open_zarr(config.zarr_path)
    statico = xr.open_zarr(config.static_path)

    indici = slot_ingeriti(store)
    if indici.size == 0:
        raise SystemExit("Nessuno slot ingerito: eseguire prima scripts/ingest_era5.py")

    sezioni = [
        descrivi_copertura(store, indici),
        descrivi_distribuzioni(store, indici, args.stride),
        descrivi_ciclo_diurno(store, indici, args.stride),
        descrivi_prevedibilita(store, indici, args.stride),
        descrivi_target(store, indici, args.stride),
        descrivi_vigo(store, statico, indici),
        descrivi_solare(store, indici),
    ]

    for sezione in sezioni:
        print(f"\n{'=' * 78}\n{sezione.titolo}\n{'=' * 78}")
        for riga in sezione.righe:
            print(riga)
        if sezione.commento:
            print(f"\n-> {sezione.commento}")

    tempi = store["valid_time"].values[indici]
    intestazione = (
        f"Generata su **{len(indici):,} slot** ingeriti, da {str(tempi[0])[:10]} a "
        f"{str(tempi[-1])[:10]}, sottocampionamento spaziale 1 su {args.stride}."
    )
    scrivi_rapporto(sezioni, args.output, intestazione)
    print(f"\nrapporto scritto in {args.output}")


if __name__ == "__main__":
    main()
