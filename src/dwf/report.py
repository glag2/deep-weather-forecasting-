"""Report PDF di una previsione: mappe, andamenti e focus locale.

Una previsione e' un insieme di array `(scadenza, latitudine, longitudine)`: leggibile
da codice, non da una persona. Questo modulo produce il documento che si guarda, con la
stessa scala di colore su tutte le scadenze di una variabile, perche' una scala per
pannello farebbe sembrare uguali giorni molto diversi.

Due scelte vanno dichiarate. Le mappe sono `imshow` con estensione geografica e nessuna
linea di costa: `cartopy` non e' fra le dipendenze del progetto e aggiungerla per
disegnare contorni non sarebbe giustificato. Le latitudini di ERA5 sono **decrescenti**,
quindi la riga 0 e' il nord: da qui `origin="upper"`, altrimenti le mappe risulterebbero
capovolte senza che nulla segnali l'errore.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from dwf.predict import summarize
from dwf.thermo import latent_heat_content

if TYPE_CHECKING:  # pragma: no cover - solo per i tipi
    from collections.abc import Sequence

    from dwf.predict import Forecast

# A4 orizzontale in pollici: le mappe sono piu' larghe che alte, come il dominio.
PAGE_SIZE_INCHES = (11.69, 8.27)

# Pannelli per pagina: una griglia 3 x 3 copre esattamente le 9 scadenze previste.
PANEL_ROWS = 3
PANEL_COLUMNS = 3

# Coda esclusa dagli estremi della scala di colore. Con gli estremi assoluti un solo
# punto anomalo appiattirebbe tutto il resto della mappa in una sola tinta.
DEFAULT_QUANTILE = 0.02

# Pressione standard al livello del mare, usata solo come ripiego quando il campo di
# pressione non viene fornito: sull'acqua e' quasi esatta, in quota sovrastima.
STANDARD_PRESSURE_PA = 101325.0

# Vigo di Cadore nella griglia a 0.25 gradi con origine a 75 N / 40 W.
VIGO_ROW = 114
VIGO_COLUMN = 210
VIGO_NAME = "Vigo di Cadore"

# Quota della cella di griglia contro quota reale del paese. La cella media l'intero
# Cadore, creste comprese, e risulta 512 m piu' alta del fondovalle abitato.
VIGO_MODEL_ELEVATION_M = 1463.0
VIGO_REAL_ELEVATION_M = 951.0

# Gradiente termico verticale standard dell'atmosfera, in K per metro.
STANDARD_LAPSE_RATE_K_PER_M = 0.0065


class ReportError(RuntimeError):
    """Errore nella costruzione del report."""


@dataclass(frozen=True, slots=True)
class MapFrame:
    """Come collocare un array sulla mappa: estensione e verso delle righe.

    `extent` e' nell'ordine richiesto da matplotlib, `(ovest, est, sud, nord)`, e cade
    sui **bordi** delle celle, non sui centri: con i centri meta' cella resterebbe fuori
    dalla mappa a ogni lato.
    """

    extent: tuple[float, float, float, float]
    origin: str


def map_frame(latitudes: np.ndarray, longitudes: np.ndarray) -> MapFrame:
    """Riquadro geografico dedotto dagli assi, con il verso corretto delle righe."""
    lat = np.asarray(latitudes, dtype=np.float64)
    lon = np.asarray(longitudes, dtype=np.float64)
    if lat.size < 2 or lon.size < 2:
        raise ReportError(
            f"Servono almeno due punti per asse per dedurre il passo della griglia: "
            f"ricevuti {lat.size} in latitudine e {lon.size} in longitudine"
        )

    passo_lat = abs(float(lat[1] - lat[0])) / 2.0
    passo_lon = abs(float(lon[1] - lon[0])) / 2.0
    sud, nord = float(lat.min()) - passo_lat, float(lat.max()) + passo_lat
    ovest, est = float(lon.min()) - passo_lon, float(lon.max()) + passo_lon

    # Riga 0 al nord (latitudini decrescenti, come le fornisce ERA5) significa origine
    # in alto; con latitudini crescenti l'origine va in basso.
    origine = "upper" if lat[0] > lat[-1] else "lower"
    return MapFrame(extent=(ovest, est, sud, nord), origin=origine)


def colour_scale(
    field: np.ndarray, *, quantile: float = DEFAULT_QUANTILE, symmetric: bool = False
) -> tuple[float, float]:
    """Estremi di colore condivisi da tutte le scadenze di una variabile.

    Condividerli e' il punto: e' l'unico modo in cui il confronto fra pannelli dice
    qualcosa sull'evoluzione invece che sulla normalizzazione di ciascuno.
    """
    if not 0.0 <= quantile < 0.5:
        raise ReportError(f"quantile deve stare in [0, 0.5): ricevuto {quantile}")
    valori = np.asarray(field, dtype=np.float64)
    finiti = valori[np.isfinite(valori)]
    if finiti.size == 0:
        raise ReportError("Il campo non contiene valori finiti: nessuna scala calcolabile")

    if symmetric:
        estremo = float(np.quantile(np.abs(finiti), 1.0 - quantile))
        estremo = max(estremo, 1e-6)
        return -estremo, estremo

    minimo = float(np.quantile(finiti, quantile))
    massimo = float(np.quantile(finiti, 1.0 - quantile))
    if massimo <= minimo:
        # Campo costante o quasi: senza un margine matplotlib disegnerebbe una tinta unica
        # e la barra dei colori non avrebbe tacche.
        margine = max(abs(minimo) * 1e-3, 1e-6)
        return minimo - margine, massimo + margine
    return minimo, massimo


def check_grid(forecast: Forecast) -> tuple[int, int, int]:
    """Verifica che i campi combacino con gli assi e restituisce `(scadenze, righe, colonne)`.

    Un campo trasposto o con una scadenza in meno non fa fallire il disegno: produce
    mappe plausibili e sbagliate, quindi va intercettato prima.
    """
    n_lead = len(forecast.valid_times)
    n_lat = int(np.asarray(forecast.latitudes).size)
    n_lon = int(np.asarray(forecast.longitudes).size)
    attesa = (n_lead, n_lat, n_lon)

    campi = {
        "t2m_mean": forecast.t2m_mean,
        "t2m_std": forecast.t2m_std,
        "precip_probability": forecast.precip_probability,
        "precip_amount": forecast.precip_amount,
        "snow_probability": forecast.snow_probability,
    }
    for nome, campo in campi.items():
        forma = tuple(np.asarray(campo).shape)
        if forma != attesa:
            raise ReportError(
                f"Il campo {nome!r} ha forma {forma}, attesa {attesa} "
                f"(scadenze x latitudini x longitudini)"
            )
    return attesa


def elevation_bias_kelvin(
    *,
    model_elevation_m: float = VIGO_MODEL_ELEVATION_M,
    real_elevation_m: float = VIGO_REAL_ELEVATION_M,
    lapse_rate_k_per_m: float = STANDARD_LAPSE_RATE_K_PER_M,
) -> float:
    """Errore sistematico di temperatura dovuto al dislivello fra cella e paese.

    Positivo significa che il modello e' **piu' freddo** del luogo reale, perche' la
    cella sta piu' in alto: e' un errore della rappresentazione del terreno, non della
    rete, e va dichiarato accanto ai numeri locali.
    """
    return (model_elevation_m - real_elevation_m) * lapse_rate_k_per_m


def local_series(
    forecast: Forecast, *, row: int = VIGO_ROW, column: int = VIGO_COLUMN
) -> pl.DataFrame:
    """Previsione in una singola cella di griglia, una riga per scadenza."""
    _, n_lat, n_lon = check_grid(forecast)
    if not 0 <= row < n_lat or not 0 <= column < n_lon:
        raise ReportError(
            f"Cella ({row}, {column}) fuori dalla griglia {n_lat} x {n_lon}: "
            f"la previsione non copre quel punto"
        )

    return pl.DataFrame(
        {
            "lead_slot": np.arange(len(forecast.valid_times), dtype=np.int16),
            "valid_time": list(forecast.valid_times),
            "t2m_mean_celsius": forecast.t2m_mean[:, row, column].astype(np.float64),
            "t2m_std_celsius": forecast.t2m_std[:, row, column].astype(np.float64),
            "precip_probability": forecast.precip_probability[:, row, column].astype(
                np.float64
            ),
            "precip_amount_mm": forecast.precip_amount[:, row, column].astype(np.float64),
            "snow_probability": forecast.snow_probability[:, row, column].astype(np.float64),
        }
    )


def latent_heat_fields(
    forecast: Forecast,
    *,
    dewpoint_celsius: np.ndarray | None = None,
    pressure_pa: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Calore latente per scadenza, con l'ipotesi usata per l'umidita'.

    La rete non prevede il punto di rugiada: senza un campo osservato da persistere,
    l'unica ipotesi difendibile e' l'aria satura, che da' il **massimo** contenuto
    latente compatibile con la temperatura prevista. L'etichetta restituita dichiara
    quale delle due ipotesi e' in uso, cosi' la pagina non presenta come previsione un
    limite superiore.
    """
    n_lead, n_lat, n_lon = check_grid(forecast)
    temperatura = np.asarray(forecast.t2m_mean, dtype=np.float32)

    if dewpoint_celsius is None:
        rugiada = temperatura
        ipotesi = "aria satura (limite superiore: la rugiada non e' prevista)"
    else:
        rugiada = _broadcast_field(
            dewpoint_celsius, (n_lead, n_lat, n_lon), name="dewpoint_celsius"
        )
        ipotesi = "rugiada dell'ultimo istante osservato, persistita sulle scadenze"

    if pressure_pa is None:
        pressione = np.full_like(temperatura, STANDARD_PRESSURE_PA)
    else:
        pressione = _broadcast_field(pressure_pa, (n_lead, n_lat, n_lon), name="pressure_pa")

    return latent_heat_content(temperatura, rugiada, pressione), ipotesi


def _broadcast_field(
    values: np.ndarray, shape: tuple[int, int, int], *, name: str
) -> np.ndarray:
    """Porta un campo bidimensionale o tridimensionale alla forma della previsione."""
    campo = np.asarray(values, dtype=np.float32)
    if campo.shape == shape:
        return campo
    if campo.shape == shape[1:]:
        return np.broadcast_to(campo, shape).astype(np.float32)
    raise ReportError(
        f"{name} ha forma {campo.shape}: attesa {shape} oppure {shape[1:]} "
        f"(un solo istante da replicare sulle scadenze)"
    )


# --------------------------------------------------------------------------- #
# Pagine
# --------------------------------------------------------------------------- #


def _new_page(title: str) -> Figure:
    """Figura di una pagina, costruita senza `pyplot` per non dipendere da un backend."""
    figura = Figure(figsize=PAGE_SIZE_INCHES)
    figura.suptitle(title, fontsize=14)
    return figura


def _lead_label(forecast: Forecast, lead: int) -> str:
    istante = forecast.valid_times[lead]
    ore = int((istante - forecast.init_time).total_seconds() // 3600)
    return f"+{ore} h  {istante:%d/%m %H} UTC"


def _map_page(
    pdf: PdfPages,
    forecast: Forecast,
    field: np.ndarray,
    *,
    title: str,
    colourbar_label: str,
    colourmap: str,
    limits: tuple[float, float],
    caption: str | None = None,
) -> None:
    """Una pagina con un pannello per scadenza e una sola barra dei colori."""
    riquadro = map_frame(forecast.latitudes, forecast.longitudes)
    figura = _new_page(title)
    assi = figura.subplots(PANEL_ROWS, PANEL_COLUMNS).ravel()
    minimo, massimo = limits
    immagine = None

    for lead, asse in enumerate(assi):
        if lead >= field.shape[0]:
            asse.set_visible(False)
            continue
        immagine = asse.imshow(
            field[lead],
            extent=riquadro.extent,
            origin=riquadro.origin,
            cmap=colourmap,
            vmin=minimo,
            vmax=massimo,
            interpolation="nearest",
            aspect="auto",
        )
        asse.set_title(_lead_label(forecast, lead), fontsize=8)
        asse.tick_params(labelsize=6)

    if immagine is not None:
        barra = figura.colorbar(immagine, ax=assi.tolist(), fraction=0.03, pad=0.02)
        barra.set_label(colourbar_label, fontsize=8)
        barra.ax.tick_params(labelsize=7)
    if caption is not None:
        figura.text(0.02, 0.015, caption, fontsize=7, style="italic")
    pdf.savefig(figura)


def _summary_lines(forecast: Forecast) -> list[str]:
    """Riepilogo per scadenza in colonne a larghezza fissa, leggibile in monospazio."""
    intestazione = (
        f"{'lead':>4}  {'istante (UTC)':<16}  {'T2m C':>7}  {'sigma C':>8}  "
        f"{'P(pioggia)':>11}  {'mm':>7}  {'P(neve)':>8}"
    )
    righe = [intestazione, "-" * len(intestazione)]
    for riga in summarize(forecast).iter_rows(named=True):
        istante: datetime = riga["valid_time"]
        righe.append(
            f"{riga['lead_slot']:>4}  {istante:%d/%m/%Y %H}  "
            f"{riga['t2m_mean_celsius']:>7.2f}  {riga['t2m_std_celsius']:>8.2f}  "
            f"{riga['precip_probability_mean']:>11.3f}  "
            f"{riga['precip_amount_mm_mean']:>7.3f}  "
            f"{riga['snow_probability_mean']:>8.3f}"
        )
    return righe


def _cover_page(
    pdf: PdfPages,
    forecast: Forecast,
    *,
    fold: int | None,
    n_parameters: int | None,
    latent_heat_note: str,
) -> None:
    n_lead, n_lat, n_lon = check_grid(forecast)
    riquadro = map_frame(forecast.latitudes, forecast.longitudes)
    ovest, est, sud, nord = riquadro.extent

    metadati = [
        f"ultimo istante osservato (inizializzazione): {forecast.init_time:%Y-%m-%d %H} UTC",
        f"scadenze previste: {n_lead}, "
        f"da {forecast.valid_times[0]:%Y-%m-%d %H} a {forecast.valid_times[-1]:%Y-%m-%d %H} UTC",
        f"dominio: {n_lat} x {n_lon} punti, "
        f"da {nord:.2f} N a {sud:.2f} N e da {ovest:.2f} E a {est:.2f} E",
        f"fold: {'non dichiarato' if fold is None else fold}",
        f"parametri del modello: "
        f"{'non dichiarati' if n_parameters is None else format(n_parameters, ',')}",
        f"calore latente: {latent_heat_note}",
        "natura della previsione: hindcast verificabile, perche' ERA5 pubblica con "
        "circa sei giorni di ritardo",
    ]

    figura = _new_page("Previsione a tre giorni - riepilogo")
    figura.text(0.06, 0.88, "\n".join(metadati), fontsize=9, va="top", family="monospace")
    figura.text(
        0.06,
        0.60,
        "\n".join(_summary_lines(forecast)),
        fontsize=8,
        va="top",
        family="monospace",
    )
    figura.text(
        0.06,
        0.05,
        "Le medie sono sul dominio intero: mescolano Atlantico e Alpi, quindi servono a "
        "confrontare le scadenze fra loro, non a descrivere un luogo.",
        fontsize=7,
        style="italic",
    )
    pdf.savefig(figura)


def _domain_mean(field: np.ndarray) -> np.ndarray:
    return np.asarray(field, dtype=np.float64).mean(axis=(1, 2))


def _trends_page(pdf: PdfPages, forecast: Forecast) -> None:
    scadenze = np.arange(len(forecast.valid_times))
    etichette = [f"{istante:%d/%m %H}" for istante in forecast.valid_times]
    temperatura = _domain_mean(forecast.t2m_mean)
    incertezza = _domain_mean(forecast.t2m_std)

    figura = _new_page("Andamento medio sul dominio e incertezza")
    superiore, centrale, inferiore = figura.subplots(3, 1, sharex=True)

    superiore.plot(scadenze, temperatura, marker="o", color="tab:red")
    superiore.fill_between(
        scadenze,
        temperatura - incertezza,
        temperatura + incertezza,
        color="tab:red",
        alpha=0.2,
        label="+/- 1 deviazione standard prevista",
    )
    superiore.set_ylabel("T2m media (gradi Celsius)", fontsize=8)
    superiore.legend(fontsize=7)

    centrale.plot(
        scadenze, _domain_mean(forecast.precip_probability), marker="o", color="tab:blue"
    )
    centrale.set_ylabel("P(pioggia) media", fontsize=8)
    quantita = centrale.twinx()
    quantita.plot(
        scadenze,
        _domain_mean(forecast.precip_amount),
        marker="s",
        linestyle="--",
        color="tab:cyan",
    )
    quantita.set_ylabel("mm attesi (media)", fontsize=8)
    quantita.tick_params(labelsize=7)

    inferiore.plot(
        scadenze, _domain_mean(forecast.snow_probability), marker="o", color="tab:purple"
    )
    inferiore.set_ylabel("P(neve) media", fontsize=8)
    inferiore.set_xticks(scadenze)
    inferiore.set_xticklabels(etichette, fontsize=7)
    inferiore.set_xlabel("istante previsto (UTC)", fontsize=8)

    for asse in (superiore, centrale, inferiore):
        asse.grid(alpha=0.3)
        asse.tick_params(labelsize=7)

    figura.text(
        0.02,
        0.015,
        "L'incertezza e' quella dichiarata dalla testa gaussiana, mediata sul dominio: "
        "cresce con la scadenza se il modello e' calibrato.",
        fontsize=7,
        style="italic",
    )
    pdf.savefig(figura)


def _local_page(
    pdf: PdfPages,
    forecast: Forecast,
    *,
    row: int,
    column: int,
    place: str,
) -> None:
    serie = local_series(forecast, row=row, column=column)
    latitudine = float(np.asarray(forecast.latitudes)[row])
    longitudine = float(np.asarray(forecast.longitudes)[column])
    scadenze = serie.get_column("lead_slot").to_numpy()
    etichette = [f"{istante:%d/%m %H}" for istante in serie.get_column("valid_time")]
    temperatura = serie.get_column("t2m_mean_celsius").to_numpy()
    incertezza = serie.get_column("t2m_std_celsius").to_numpy()
    scarto = elevation_bias_kelvin()

    figura = _new_page(
        f"{place} - cella ({row}, {column}) a {latitudine:.2f} N / {longitudine:.2f} E"
    )
    superiore, centrale, inferiore = figura.subplots(3, 1, sharex=True)

    superiore.plot(scadenze, temperatura, marker="o", color="tab:red", label="T2m prevista")
    superiore.fill_between(
        scadenze,
        temperatura - incertezza,
        temperatura + incertezza,
        color="tab:red",
        alpha=0.2,
    )
    superiore.plot(
        scadenze,
        temperatura + scarto,
        marker="^",
        linestyle=":",
        color="tab:orange",
        label=f"corretta di +{scarto:.1f} K per il dislivello",
    )
    superiore.axhline(0.0, color="grey", linewidth=0.8)
    superiore.set_ylabel("gradi Celsius", fontsize=8)
    superiore.legend(fontsize=7)

    centrale.bar(
        scadenze,
        serie.get_column("precip_amount_mm").to_numpy(),
        color="tab:cyan",
        width=0.6,
    )
    centrale.set_ylabel("mm attesi", fontsize=8)
    probabilita = centrale.twinx()
    probabilita.plot(
        scadenze,
        serie.get_column("precip_probability").to_numpy(),
        marker="o",
        color="tab:blue",
    )
    probabilita.set_ylabel("P(pioggia)", fontsize=8)
    probabilita.set_ylim(0.0, 1.0)
    probabilita.tick_params(labelsize=7)

    inferiore.plot(
        scadenze,
        serie.get_column("snow_probability").to_numpy(),
        marker="o",
        color="tab:purple",
    )
    inferiore.set_ylabel("P(neve)", fontsize=8)
    inferiore.set_ylim(0.0, 1.0)
    inferiore.set_xticks(scadenze)
    inferiore.set_xticklabels(etichette, fontsize=7)
    inferiore.set_xlabel("istante previsto (UTC)", fontsize=8)

    for asse in (superiore, centrale, inferiore):
        asse.grid(alpha=0.3)
        asse.tick_params(labelsize=7)

    dislivello = VIGO_MODEL_ELEVATION_M - VIGO_REAL_ELEVATION_M
    figura.subplots_adjust(bottom=0.16)
    figura.text(
        0.02,
        0.015,
        f"La cella di griglia ha quota {VIGO_MODEL_ELEVATION_M:.0f} m, il paese sta a "
        f"circa {VIGO_REAL_ELEVATION_M:.0f} m: {dislivello:.0f} m di scarto, perche' a "
        f"0.25 gradi la cella media tutto il Cadore, creste comprese.\n"
        f"Con il gradiente standard di "
        f"{STANDARD_LAPSE_RATE_K_PER_M * 1000:.1f} K/km ne risulta un errore "
        f"sistematico di circa {scarto:.1f} K: la previsione e' piu' fredda del paese. "
        f"La curva punteggiata mostra la temperatura corretta di quel solo scarto; "
        f"resta un errore di rappresentazione del terreno, non una correzione validata.",
        fontsize=7,
        style="italic",
    )
    pdf.savefig(figura)


# --------------------------------------------------------------------------- #
# Documento
# --------------------------------------------------------------------------- #


def write_report(
    forecast: Forecast,
    destination: str | Path,
    *,
    fold: int | None = None,
    n_parameters: int | None = None,
    dewpoint_celsius: np.ndarray | None = None,
    pressure_pa: np.ndarray | None = None,
    focus_row: int = VIGO_ROW,
    focus_column: int = VIGO_COLUMN,
    focus_place: str = VIGO_NAME,
) -> Path:
    """Scrive il report PDF della previsione e restituisce il percorso prodotto.

    `dewpoint_celsius` e `pressure_pa` sono facoltativi e servono solo alla pagina del
    calore latente: se mancano si usa l'ipotesi di aria satura, dichiarata nel documento.
    """
    check_grid(forecast)
    percorso = Path(destination).expanduser()
    percorso.parent.mkdir(parents=True, exist_ok=True)

    latente, ipotesi_latente = latent_heat_fields(
        forecast, dewpoint_celsius=dewpoint_celsius, pressure_pa=pressure_pa
    )

    with PdfPages(percorso) as pdf:
        _cover_page(
            pdf,
            forecast,
            fold=fold,
            n_parameters=n_parameters,
            latent_heat_note=ipotesi_latente,
        )
        _map_page(
            pdf,
            forecast,
            forecast.t2m_mean,
            title="Temperatura a 2 m prevista",
            colourbar_label="gradi Celsius",
            colourmap="RdBu_r",
            limits=colour_scale(forecast.t2m_mean),
            caption="Scala condivisa fra i pannelli: le differenze visibili sono "
            "differenze di temperatura, non di normalizzazione.",
        )
        _map_page(
            pdf,
            forecast,
            forecast.precip_probability,
            title="Probabilita' di precipitazione",
            colourbar_label="probabilita'",
            colourmap="Blues",
            limits=(0.0, 1.0),
        )
        _map_page(
            pdf,
            forecast,
            forecast.precip_amount,
            title="Precipitazione attesa",
            colourbar_label="millimetri per slot",
            colourmap="YlGnBu",
            limits=(0.0, max(colour_scale(forecast.precip_amount, quantile=0.001)[1], 1e-3)),
            caption="Valore atteso: quantita' condizionata pesata dalla probabilita' "
            "che precipiti davvero.",
        )
        _map_page(
            pdf,
            forecast,
            forecast.snow_probability,
            title="Probabilita' di neve",
            colourbar_label="probabilita'",
            colourmap="BuPu",
            limits=(0.0, 1.0),
            caption="P(neve) = P(precipitazione) x quota nevosa: non puo' superare la "
            "probabilita' di precipitazione.",
        )
        _map_page(
            pdf,
            forecast,
            latente / 1000.0,
            title="Contenuto di calore latente",
            colourbar_label="kJ per kg d'aria",
            colourmap="magma",
            limits=colour_scale(latente / 1000.0),
            caption=f"Ipotesi sull'umidita': {ipotesi_latente}.",
        )
        _trends_page(pdf, forecast)
        _local_page(
            pdf, forecast, row=focus_row, column=focus_column, place=focus_place
        )

    return percorso


def report_path(directory: str | Path, forecast: Forecast) -> Path:
    """Percorso convenzionale del report, con l'istante di inizializzazione nel nome."""
    return Path(directory) / f"forecast_report_{forecast.init_time:%Y%m%d_%H}.pdf"


def domain_mean_series(forecast: Forecast, fields: Sequence[str]) -> pl.DataFrame:
    """Medie sul dominio per scadenza, le stesse mostrate nella pagina degli andamenti."""
    disponibili = {
        "t2m_mean": forecast.t2m_mean,
        "t2m_std": forecast.t2m_std,
        "precip_probability": forecast.precip_probability,
        "precip_amount": forecast.precip_amount,
        "snow_probability": forecast.snow_probability,
    }
    colonne: dict[str, object] = {
        "lead_slot": np.arange(len(forecast.valid_times), dtype=np.int16)
    }
    for nome in fields:
        if nome not in disponibili:
            raise ReportError(
                f"Campo sconosciuto: {nome!r}. Disponibili: {sorted(disponibili)}"
            )
        colonne[nome] = _domain_mean(disponibili[nome])
    return pl.DataFrame(colonne)


__all__ = [
    "PAGE_SIZE_INCHES",
    "STANDARD_LAPSE_RATE_K_PER_M",
    "VIGO_COLUMN",
    "VIGO_MODEL_ELEVATION_M",
    "VIGO_NAME",
    "VIGO_REAL_ELEVATION_M",
    "VIGO_ROW",
    "MapFrame",
    "ReportError",
    "check_grid",
    "colour_scale",
    "domain_mean_series",
    "elevation_bias_kelvin",
    "latent_heat_fields",
    "local_series",
    "map_frame",
    "report_path",
    "write_report",
]
